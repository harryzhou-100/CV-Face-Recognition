#!/usr/bin/env python3
"""Full LFW verification with InsightFace w600k_r50 ONNX on GPU (batched)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from onnx2torch import convert as onnx2torch_convert
from sklearn.preprocessing import normalize as sk_normalize
from tqdm import tqdm

STAGE2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE2_ROOT / "scripts"))
from lfw_eval_common import (  # noqa: E402
    STAGE2_ROOT as _,
    best_threshold,
    build_pairs,
    compute_stats,
    group_by_person,
    insightface_lfw_10fold,
    list_lfw_images,
    load_official_lfw_pairs,
    write_report,
)

DEFAULT_ONNX = Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"
FALLBACK_ONNX = STAGE2_ROOT / "models/w600k_r50.onnx"


def resolve_onnx(path: Path | None) -> Path:
    if path and path.exists():
        return path
    if DEFAULT_ONNX.exists():
        return DEFAULT_ONNX
    if FALLBACK_ONNX.exists():
        return FALLBACK_ONNX
    raise FileNotFoundError(
        "w600k_r50.onnx not found. Run: insightface app or copy buffalo_l/w600k_r50.onnx"
    )


def create_session(onnx_path: Path, device_id: int) -> ort.InferenceSession:
    available = ort.get_available_providers()
    sess_providers: list = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        try:
            sess = ort.InferenceSession(
                str(onnx_path),
                providers=[
                    ("CUDAExecutionProvider", {"device_id": device_id}),
                    "CPUExecutionProvider",
                ],
            )
            if "CUDAExecutionProvider" in sess.get_providers():
                print(f"[onnx] CUDAExecutionProvider active (device {device_id})")
                return sess
        except Exception as exc:
            print(f"[onnx] CUDA init failed ({exc}), falling back to CPU")
    print("[onnx] Using CPUExecutionProvider")
    return ort.InferenceSession(str(onnx_path), providers=sess_providers)


def embed_all_torch(
    torch_model: torch.nn.Module,
    image_paths: list[Path],
    batch_size: int,
    device: torch.device,
    image_size: int = 112,
    tta_flip: bool = False,
) -> dict[str, np.ndarray | None]:
    """Run ONNX graph on PyTorch CUDA (uses L40S when ORT CUDA/cuDNN unavailable)."""
    cache: dict[str, np.ndarray | None] = {}
    batch_imgs: list[np.ndarray] = []
    batch_keys: list[str] = []
    mean, std = 127.5, 127.5

    def _blob(im: np.ndarray) -> np.ndarray:
        return cv2.dnn.blobFromImage(
            im, 1.0 / std, (image_size, image_size), (mean, mean, mean), swapRB=True
        )

    @torch.no_grad()
    def flush() -> None:
        if not batch_imgs:
            return
        blob_list = []
        for im in batch_imgs:
            blob_list.append(_blob(im))
            if tta_flip:
                blob_list.append(_blob(cv2.flip(im, 1)))
        blobs = np.concatenate(blob_list, axis=0)
        out = torch_model(torch.from_numpy(blobs).to(device)).cpu().numpy()
        step = 2 if tta_flip else 1
        for i, key in enumerate(batch_keys):
            if tta_flip:
                v = out[i * 2].astype(np.float32) + out[i * 2 + 1].astype(np.float32)
                v = sk_normalize(v.reshape(1, -1))[0]
            else:
                v = sk_normalize(out[i].astype(np.float32).reshape(1, -1))[0]
            cache[key] = v

    for path in tqdm(image_paths, desc="LFW embed (PyTorch CUDA)"):
        img = cv2.imread(str(path))
        if img is None:
            cache[str(path)] = None
            continue
        batch_imgs.append(img)
        batch_keys.append(str(path))
        if len(batch_imgs) >= batch_size:
            flush()
            batch_imgs, batch_keys = [], []
    flush()
    return cache


def embed_all_batched(
    model: ArcFaceONNX,
    image_paths: list[Path],
    batch_size: int,
    tta_flip: bool = False,
) -> dict[str, np.ndarray | None]:
    cache: dict[str, np.ndarray | None] = {}
    batch_imgs: list[np.ndarray] = []
    batch_keys: list[str] = []

    for path in tqdm(image_paths, desc="LFW embed (ONNX GPU)"):
        img = cv2.imread(str(path))
        if img is None:
            cache[str(path)] = None
            continue
        batch_imgs.append(img)
        batch_keys.append(str(path))
        if len(batch_imgs) >= batch_size:
            _flush_batch(model, batch_imgs, batch_keys, cache, tta_flip)
            batch_imgs, batch_keys = [], []

    if batch_imgs:
        _flush_batch(model, batch_imgs, batch_keys, cache, tta_flip)
    return cache


def _flush_batch(
    model: ArcFaceONNX,
    imgs: list[np.ndarray],
    keys: list[str],
    cache: dict[str, np.ndarray | None],
    tta_flip: bool = False,
) -> None:
    if not tta_flip:
        feats = model.get_feat(imgs)
        for key, feat in zip(keys, feats):
            cache[key] = sk_normalize(feat.astype(np.float32).reshape(1, -1))[0]
        return
    for key, im in zip(keys, imgs):
        f1 = model.get_feat(im)[0].astype(np.float32)
        f2 = model.get_feat(cv2.flip(im, 1))[0].astype(np.float32)
        cache[key] = sk_normalize((f1 + f2).reshape(1, -1))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="LFW verification with InsightFace ONNX on GPU")
    parser.add_argument("--lfw-root", type=Path, default=STAGE2_ROOT / "data/lfw_flat")
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--pairs-each", type=int, default=3000)
    parser.add_argument(
        "--pairs-file",
        type=Path,
        default=STAGE2_ROOT / "data/sklearn_cache/lfw_home/pairs.txt",
        help="Official LFW pairs.txt (recommended for InsightFace-aligned benchmark)",
    )
    parser.add_argument(
        "--random-pairs",
        action="store_true",
        help="Use random same/diff pairs instead of official pairs.txt",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=("auto", "onnx", "torch"),
        default="torch",
        help="auto: try ORT CUDA then PyTorch CUDA; torch: onnx2torch on GPU",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=True,
        help="Flip TTA: sum(original, mirrored) then L2-normalize (InsightFace default)",
    )
    parser.add_argument("--no-tta", action="store_true", help="Disable flip TTA")
    parser.add_argument(
        "--report",
        type=Path,
        default=STAGE2_ROOT / "reports/lfw_verification_report.md",
    )
    args = parser.parse_args()
    if args.no_tta:
        args.tta = False

    lfw_root = args.lfw_root
    if not lfw_root.exists():
        nested = STAGE2_ROOT / "data/lfw" / "lfw"
        lfw_root = nested if nested.exists() else STAGE2_ROOT / "data/lfw/lfw"

    onnx_path = resolve_onnx(args.onnx)
    images = list_lfw_images(lfw_root)
    by_person = group_by_person(lfw_root, images)
    if args.random_pairs:
        pairs = build_pairs(by_person, args.pairs_each, 42)
        pair_source = f"random seed=42, {args.pairs_each} per class"
    elif args.pairs_file.exists():
        pairs = load_official_lfw_pairs(args.pairs_file, lfw_root)
        pair_source = str(args.pairs_file.resolve())
    else:
        pairs = build_pairs(by_person, args.pairs_each, 42)
        pair_source = "random (pairs.txt missing)"
    print(f"[pairs] {len(pairs)} pairs from {pair_source}")
    unique_paths = sorted({p for pair in pairs for p in pair[:2]}, key=str)

    use_torch = args.backend == "torch"
    ort_cuda_ok = False
    if args.backend in ("auto", "onnx"):
        session = create_session(onnx_path, args.device_id)
        ort_cuda_ok = "CUDAExecutionProvider" in session.get_providers()
        use_torch = args.backend == "torch" or (args.backend == "auto" and not ort_cuda_ok)

    if use_torch:
        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        print(f"[torch] onnx2torch on {device}")
        torch_model = onnx2torch_convert(str(onnx_path)).to(device).eval()
        cache = embed_all_torch(
            torch_model, unique_paths, args.batch_size, device, tta_flip=args.tta
        )
        backend_label = f"PyTorch CUDA (onnx2torch, {onnx_path.name})"
    else:
        model = ArcFaceONNX(model_file=str(onnx_path), session=session)
        print(
            f"[model] {onnx_path.name} input_size={model.input_size} "
            f"mean={model.input_mean} std={model.input_std}"
        )
        cache = embed_all_batched(model, unique_paths, args.batch_size, tta_flip=args.tta)
        backend_label = "ONNX Runtime (InsightFace w600k_r50)"
    if args.tta:
        backend_label += " + TTA flip"

    emb1_list, emb2_list, labels = [], [], []
    skipped = 0
    for p1, p2, y in pairs:
        e1, e2 = cache.get(str(p1)), cache.get(str(p2))
        if e1 is None or e2 is None:
            skipped += 1
            continue
        emb1_list.append(e1)
        emb2_list.append(e2)
        labels.append(y)

    emb1 = np.stack(emb1_list, axis=0)
    emb2 = np.stack(emb2_list, axis=0)
    y_true = np.array(labels, dtype=bool)
    scores_arr = np.sum(np.square(emb1 - emb2), axis=1)
    thr, _ = best_threshold(y_true.astype(int), -scores_arr)

    acc_10fold, acc_std, fold_accs = insightface_lfw_10fold(emb1, emb2, y_true)
    print(f"[10-fold InsightFace] mean={acc_10fold * 100:.2f}% std={acc_std * 100:.2f}%")

    stats = compute_stats(
        y_true.astype(int),
        -scores_arr,
        thr,
        lfw_dir=str(lfw_root.resolve()),
        model_desc=str(onnx_path.resolve()),
        n_images=len(images),
        n_persons=len(by_person),
        skipped=skipped,
        accuracy_10fold=acc_10fold,
        accuracy_10fold_std=acc_std,
        fold_accs=fold_accs,
        pair_source=pair_source,
        metric="L2 squared distance (InsightFace official)",
    )
    write_report(
        stats,
        args.report,
        backend=backend_label,
        train_note="WebFace600K 预训练（InsightFace buffalo_l / w600k_r50）",
    )
    print(f"LFW Accuracy: {stats['accuracy'] * 100:.2f}%")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
