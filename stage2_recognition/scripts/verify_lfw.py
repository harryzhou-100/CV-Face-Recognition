#!/usr/bin/env python3
"""LFW face verification with PyTorch ResNet50+ArcFace (InsightFace-aligned preprocessing)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE2_ROOT / "scripts"))
sys.path.insert(0, str(STAGE2_ROOT / "models"))

from arcface_head import ResNet50ArcFace  # noqa: E402
from insightface_preprocess import image_to_torch_tensor  # noqa: E402
from lfw_eval_common import (  # noqa: E402
    build_pairs,
    best_threshold,
    compute_stats,
    group_by_person,
    list_lfw_images,
    write_report,
)


def load_model(ckpt_path: Path, device: torch.device) -> ResNet50ArcFace:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ResNet50ArcFace(
        num_classes=ckpt["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)
    return model


@torch.no_grad()
def embed(
    model: ResNet50ArcFace,
    path: Path,
    size: int,
    device: torch.device,
) -> np.ndarray | None:
    t = image_to_torch_tensor(path, size, device)
    if t is None:
        return None
    return model.forward_features(t).cpu().numpy()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lfw-root",
        type=Path,
        default=STAGE2_ROOT / "data/lfw_flat",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=STAGE2_ROOT / "work_dirs/resnet50_arcface_frozen/best.pth",
    )
    parser.add_argument("--pairs-each", type=int, default=3000)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument(
        "--report",
        type=Path,
        default=STAGE2_ROOT / "reports/lfw_verification_report_pytorch.md",
    )
    args = parser.parse_args()

    lfw_root = args.lfw_root
    if not lfw_root.exists():
        nested = STAGE2_ROOT / "data/lfw" / "lfw"
        lfw_root = nested if nested.exists() else STAGE2_ROOT / "data/lfw/lfw"

    images = list_lfw_images(lfw_root)
    by_person = group_by_person(lfw_root, images)
    pairs = build_pairs(by_person, args.pairs_each, 42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    labels, scores = [], []
    skipped = 0
    cache: dict[str, np.ndarray | None] = {}
    for p1, p2, y in pairs:
        k1, k2 = str(p1), str(p2)
        if k1 not in cache:
            cache[k1] = embed(model, p1, args.image_size, device)
        if k2 not in cache:
            cache[k2] = embed(model, p2, args.image_size, device)
        e1, e2 = cache[k1], cache[k2]
        if e1 is None or e2 is None:
            skipped += 1
            continue
        labels.append(y)
        scores.append(float(np.dot(e1, e2)))

    y_true = np.array(labels)
    scores_arr = np.array(scores)
    thr, _ = best_threshold(y_true, scores_arr)
    stats = compute_stats(
        y_true,
        scores_arr,
        thr,
        lfw_dir=str(lfw_root.resolve()),
        model_desc=str(args.checkpoint.resolve()),
        n_images=len(images),
        n_persons=len(by_person),
        skipped=skipped,
    )
    write_report(
        stats,
        args.report,
        backend="PyTorch IResNet50-ArcFace (InsightFace-aligned preprocess)",
        train_note="MS-Celeb-1M（MS1MV3 子集）微调",
    )
    (STAGE2_ROOT / "reports" / "lfw_metrics_pytorch.json").write_text(
        __import__("json").dumps(stats, indent=2),
        encoding="utf-8",
    )
    print(f"LFW Accuracy: {stats['accuracy'] * 100:.2f}%")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
