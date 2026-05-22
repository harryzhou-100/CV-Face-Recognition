#!/usr/bin/env python3
"""
Compare FP32 vs dynamic-INT8 ResNet50 backbone:
file size, LFW subset per-image latency, verification accuracy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE2_OPT = Path(__file__).resolve().parents[1]
RECOGNITION_SCRIPTS = PROJECT_ROOT / "stage2_recognition" / "scripts"
sys.path.insert(0, str(STAGE2_OPT / "scripts"))
sys.path.insert(0, str(RECOGNITION_SCRIPTS))

from insightface_preprocess import insightface_blob_from_image  # noqa: E402
from lfw_eval_common import (  # noqa: E402
    best_threshold,
    insightface_lfw_10fold,
    load_official_lfw_pairs,
    pairwise_sq_dist,
)
from sklearn.metrics import accuracy_score  # noqa: E402
from model_utils import (  # noqa: E402
    DEFAULT_CKPT,
    load_fp32_backbone,
    load_quantized_backbone,
    resolve_checkpoint,
)

DEFAULT_QUANT = STAGE2_OPT / "models" / "resnet50_int8_dynamic.pth"
DEFAULT_LFW = PROJECT_ROOT / "stage2_recognition" / "data" / "lfw_flat"
DEFAULT_PAIRS = PROJECT_ROOT / "stage2_recognition" / "data/sklearn_cache/lfw_home/pairs.txt"
DEFAULT_REPORT = STAGE2_OPT / "reports" / "quantization_comparison.md"


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def embed_image(model: torch.nn.Module, path: Path, device: torch.device) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    blob = insightface_blob_from_image(img, 112)
    x = torch.from_numpy(blob).to(device)
    with torch.no_grad():
        out = model(x).cpu().numpy()[0]
    return out.astype(np.float32)


def benchmark_latency(
    model: torch.nn.Module,
    image_paths: list[Path],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    usable = [p for p in image_paths if cv2.imread(str(p)) is not None]
    if not usable:
        return float("nan")

    for p in usable[:warmup]:
        embed_image(model, p, device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for _ in range(repeats):
        for p in usable:
            embed_image(model, p, device)
            n += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (elapsed / n) * 1000.0


def evaluate_lfw_subset(
    model: torch.nn.Module,
    pairs: list[tuple[Path, Path, int]],
    device: torch.device,
) -> tuple[float, float]:
    cache: dict[str, np.ndarray | None] = {}
    for p1, p2, _ in pairs:
        for p in (p1, p2):
            k = str(p)
            if k not in cache:
                cache[k] = embed_image(model, p, device)

    emb1, emb2, labels = [], [], []
    for p1, p2, y in pairs:
        e1, e2 = cache[str(p1)], cache[str(p2)]
        if e1 is None or e2 is None:
            continue
        emb1.append(e1)
        emb2.append(e2)
        labels.append(y)

    emb1_arr = np.stack(emb1)
    emb2_arr = np.stack(emb2)
    y = np.array(labels, dtype=bool)
    dist = pairwise_sq_dist(emb1_arr, emb2_arr)
    if len(y) == 6000:
        acc, std, _ = insightface_lfw_10fold(emb1_arr, emb2_arr, y)
        return acc, std
    thr, acc = best_threshold(y.astype(int), -dist)
    _ = thr
    return float(acc), 0.0


def save_fp32_stub(ckpt: Path, out: Path) -> None:
    """Save backbone-only FP32 weights for fair size comparison with INT8 bundle."""
    raw = torch.load(ckpt, map_location="cpu")
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    backbone = {k: v for k, v in state.items() if k.startswith("backbone.")}
    torch.save({"state_dict": backbone}, out)


def build_table(rows: list[dict]) -> str:
    lines = [
        "# ResNet50 动态量化对比（交付表）",
        "",
        "| 指标 | FP32（原始） | INT8 动态量化 | 变化 |",
        "|------|-------------|---------------|------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['metric']} | {r['fp32']} | {r['int8']} | {r['delta']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--quantized", type=Path, default=DEFAULT_QUANT)
    parser.add_argument("--lfw-root", type=Path, default=DEFAULT_LFW)
    parser.add_argument("--pairs-file", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--max-pairs", type=int, default=6000, help="LFW pairs for accuracy eval")
    parser.add_argument("--bench-images", type=int, default=500, help="Images for latency benchmark")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if not args.quantized.exists():
        raise FileNotFoundError(
            f"Quantized weights missing: {args.quantized}. Run: python3 stage2_optimization/scripts/quantize.py"
        )

    ckpt = resolve_checkpoint(args.checkpoint)
    fp32_device = torch.device(
        args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    )
    int8_device = torch.device("cpu")  # PyTorch dynamic INT8 linear runs on CPU only

    fp32_stub = STAGE2_OPT / "models" / "resnet50_fp32_backbone_only.pth"
    if not fp32_stub.exists():
        print("[compare] Exporting backbone-only FP32 stub for size metric...", flush=True)
        save_fp32_stub(ckpt, fp32_stub)

    print("[compare] Loading FP32 model...", flush=True)
    fp32_model = load_fp32_backbone(ckpt, device=fp32_device)
    print("[compare] Loading INT8 model...", flush=True)
    int8_model = load_quantized_backbone(args.quantized, ckpt, device=int8_device)

    lfw_images = sorted(args.lfw_root.glob("*.jpg"))
    bench_paths = lfw_images[: args.bench_images]

    pairs = load_official_lfw_pairs(args.pairs_file, args.lfw_root)
    if args.max_pairs < len(pairs):
        pairs = pairs[: args.max_pairs]

    print(f"[compare] FP32 latency ({fp32_device})...")
    fp32_ms = benchmark_latency(fp32_model, bench_paths, fp32_device, args.warmup, args.repeats)
    print(f"[compare] INT8 latency ({int8_device})...")
    int8_ms = benchmark_latency(int8_model, bench_paths, int8_device, args.warmup, args.repeats)

    print(f"[compare] FP32 LFW accuracy ({fp32_device})...")
    fp32_acc, fp32_std = evaluate_lfw_subset(fp32_model, pairs, fp32_device)
    print(f"[compare] INT8 LFW accuracy ({int8_device})...")
    int8_acc, int8_std = evaluate_lfw_subset(int8_model, pairs, int8_device)

    fp32_size = file_size_mb(fp32_stub)
    int8_size = file_size_mb(args.quantized)
    size_ratio = (1.0 - int8_size / fp32_size) * 100 if fp32_size > 0 else 0.0
    speed_ratio = (1.0 - int8_ms / fp32_ms) * 100 if fp32_ms > 0 else 0.0
    acc_delta = (int8_acc - fp32_acc) * 100

    rows = [
        {
            "metric": "权重文件大小 (MB)",
            "fp32": f"{fp32_size:.2f}",
            "int8": f"{int8_size:.2f}",
            "delta": f"↓ {size_ratio:.1f}%",
        },
        {
            "metric": "单张推理耗时 (ms/张)",
            "fp32": f"{fp32_ms:.3f}",
            "int8": f"{int8_ms:.3f}",
            "delta": f"{'↓' if speed_ratio >= 0 else '↑'} {abs(speed_ratio):.1f}%",
        },
        {
            "metric": f"LFW 子集准确率 (%)（{len(pairs)} 对）",
            "fp32": f"{fp32_acc * 100:.2f}" + (f" ± {fp32_std * 100:.2f}" if fp32_std > 0 else ""),
            "int8": f"{int8_acc * 100:.2f}" + (f" ± {int8_std * 100:.2f}" if int8_std > 0 else ""),
            "delta": f"{acc_delta:+.2f} pp",
        },
    ]

    table_md = build_table(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    meta = (
        f"\n- 基准权重: `{ckpt}`\n"
        f"- 量化权重: `{args.quantized}`\n"
        f"- LFW: `{args.lfw_root}` ({len(pairs)} pairs)\n"
        f"- 测速样本: {len(bench_paths)} 张, FP32 device={fp32_device}, INT8 device={int8_device}\n"
    )
    args.report.write_text(table_md + meta, encoding="utf-8")

    print("\n" + table_md)
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
