#!/usr/bin/env python3
"""Run inference on val images and save categorized visualizations."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE2_ROOT = Path(__file__).resolve().parents[1]
CONFIG = STAGE2_ROOT / "configs" / "retinanet_r50_fpn_wider_face.py"
OUTPUT_ROOT = STAGE2_ROOT / "outputs"
DATA_ROOT = STAGE2_ROOT / "data" / "wider_face"
WORK_DIR = STAGE2_ROOT / "work_dirs" / "retinanet_r50_wider_face"

CATEGORIES = ("easy", "medium", "hard", "multi_face", "no_face", "general")


def find_checkpoint(work_dir: Path) -> Path:
    for p in sorted(work_dir.glob("epoch_*.pth"), reverse=True):
        return p
    for p in sorted(work_dir.glob("best_*.pth"), reverse=True):
        return p
    raise FileNotFoundError(work_dir)


def face_difficulty(bh: float, img_h: float) -> str:
    if img_h <= 0:
        return "general"
    scale = bh / img_h
    if scale >= 0.08:
        return "easy"
    if scale >= 0.03:
        return "medium"
    return "hard"


def categorize_image(n_det: int, n_gt: int, hardest: str) -> str:
    if n_det == 0:
        return "no_face"
    if n_det >= 3:
        return "multi_face"
    return hardest


def draw_boxes(img: np.ndarray, boxes, scores, color=(0, 255, 0)) -> np.ndarray:
    out = img.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-images", type=int, default=120)
    parser.add_argument("--score-thr", type=float, default=0.3)
    args = parser.parse_args()

    import os

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    from mmdet.apis import init_detector, inference_detector
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)

    default_ckpt = WORK_DIR / "epoch_6.pth"
    ckpt = args.checkpoint or (
        default_ckpt if default_ckpt.exists() else find_checkpoint(WORK_DIR)
    )
    model = init_detector(str(args.config), str(ckpt), device="cuda:0")

    ann_path = DATA_ROOT / "annotations" / "instances_val.json"
    with ann_path.open() as f:
        coco = json.load(f)

    img_dir = DATA_ROOT / "WIDER_val" / "images"
    id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
    id_to_h = {im["id"]: im["height"] for im in coco["images"]}
    gt_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        gt_by_img.setdefault(ann["image_id"], []).append(ann)

    image_ids = list(id_to_file.keys())
    random.seed(42)
    random.shuffle(image_ids)
    image_ids = image_ids[: args.num_images]

    for cat in CATEGORIES:
        (OUTPUT_ROOT / cat).mkdir(parents=True, exist_ok=True)

    manifest = []
    for img_id in image_ids:
        rel = id_to_file[img_id]
        img_path = img_dir / rel
        if not img_path.exists():
            continue
        result = inference_detector(model, str(img_path))
        pred = result.pred_instances
        keep = pred.scores.cpu().numpy() >= args.score_thr
        boxes = pred.bboxes.cpu().numpy()[keep]
        scores = pred.scores.cpu().numpy()[keep]

        gts = gt_by_img.get(img_id, [])
        img_h = id_to_h.get(img_id, 1)
        hardest = "easy"
        for g in gts:
            d = face_difficulty(g["bbox"][3], img_h)
            if d == "hard":
                hardest = "hard"
            elif d == "medium" and hardest != "hard":
                hardest = "medium"
        cat = categorize_image(len(boxes), len(gts), hardest)
        also_save_general = cat != "general"

        img = cv2.imread(str(img_path))
        vis = draw_boxes(img, boxes, scores)
        out_name = rel.replace("/", "__")
        out_path = OUTPUT_ROOT / cat / out_name
        cv2.imwrite(str(out_path), vis)
        if also_save_general and cat != "general":
            cv2.imwrite(str(OUTPUT_ROOT / "general" / out_name), vis)
        manifest.append(
            {
                "image": rel,
                "category": cat,
                "num_det": int(len(boxes)),
                "num_gt": len(gts),
                "output": str(out_path.relative_to(STAGE2_ROOT)),
            }
        )

    manifest_path = OUTPUT_ROOT / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(manifest)} visualizations under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
