#!/usr/bin/env python3
"""Run landmark inference and affine face alignment on 300W test/val images."""

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
CONFIG = STAGE2_ROOT / "configs" / "hrnetv2_w18_300w_256x256.py"
WORK_DIR = STAGE2_ROOT / "work_dirs" / "hrnetv2_w18_300w"
DATA_ROOT = STAGE2_ROOT / "data" / "300w"
OUT_ALIGNED = STAGE2_ROOT / "outputs" / "aligned"
OUT_COMPARE = STAGE2_ROOT / "outputs" / "comparisons"
OUT_LM = STAGE2_ROOT / "outputs" / "landmarks"

sys.path.insert(0, str(STAGE2_ROOT / "scripts"))
from face_align_utils import align_face, draw_landmarks_68  # noqa: E402


def find_checkpoint(work_dir: Path) -> Path:
    for p in sorted(work_dir.glob("best_NME_*.pth")):
        return p
    for p in sorted(work_dir.glob("epoch_*.pth"), reverse=True):
        return p
    raise FileNotFoundError(work_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--ann-file", type=Path, default=DATA_ROOT / "annotations" / "face_landmarks_300w_test.json")
    parser.add_argument("--num-images", type=int, default=80)
    parser.add_argument("--output-size", type=int, default=256)
    args = parser.parse_args()

    import os

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    from mmpose.apis import inference_topdown, init_model
    from mmpose.utils import register_all_modules

    register_all_modules(init_default_scope=True)

    ckpt = args.checkpoint or find_checkpoint(WORK_DIR)
    model = init_model(str(args.config), str(ckpt), device="cuda:0")

    ann = json.loads(args.ann_file.read_text(encoding="utf-8"))
    img_root = DATA_ROOT / "images"
    ids = [im["id"] for im in ann["images"]]
    random.seed(42)
    random.shuffle(ids)
    ids = ids[: args.num_images]

    id_to_file = {im["id"]: im["file_name"] for im in ann["images"]}
    id_to_bbox = {}
    for a in ann["annotations"]:
        if a["image_id"] in ids:
            id_to_bbox[a["image_id"]] = a

    OUT_ALIGNED.mkdir(parents=True, exist_ok=True)
    OUT_COMPARE.mkdir(parents=True, exist_ok=True)
    OUT_LM.mkdir(parents=True, exist_ok=True)

    manifest = []
    for img_id in ids:
        rel = id_to_file[img_id]
        img_path = img_root / rel
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ann_item = id_to_bbox.get(img_id)
        if not ann_item:
            continue
        # MMPose top-down needs bbox xyxy
        kpts = np.array(ann_item["keypoints"], dtype=np.float32).reshape(-1, 3)
        xy = kpts[:, :2]
        x1, y1 = xy.min(axis=0)
        x2, y2 = xy.max(axis=0)
        pad = 0.2 * max(x2 - x1, y2 - y1)
        bbox = np.array([[x1 - pad, y1 - pad, x2 + pad, y2 + pad]], dtype=np.float32)

        results = inference_topdown(model, img, bboxes=bbox)
        if not results:
            continue
        pred = results[0].pred_instances
        pts = pred.keypoints[0]
        if hasattr(pts, "cpu"):
            pts = pts.cpu().numpy()
        pts = np.asarray(pts, dtype=np.float32).reshape(68, 2)

        aligned, _ = align_face(img, pts, output_size=(args.output_size, args.output_size))
        name = rel.replace("/", "__")
        p_aligned = OUT_ALIGNED / name
        p_lm = OUT_LM / name
        p_cmp = OUT_COMPARE / name

        cv2.imwrite(str(p_aligned), aligned)
        cv2.imwrite(str(p_lm), draw_landmarks_68(img, pts))
        # side-by-side: original | landmarks | aligned
        h, w = img.shape[:2]
        thumb_h = 256
        scale = thumb_h / h
        thumb_w = int(w * scale)
        orig_r = cv2.resize(img, (thumb_w, thumb_h))
        lm_r = cv2.resize(draw_landmarks_68(img, pts), (thumb_w, thumb_h))
        cmp_img = np.hstack([orig_r, lm_r, aligned])
        cv2.imwrite(str(p_cmp), cmp_img)
        manifest.append({"image": rel, "aligned": str(p_aligned.relative_to(STAGE2_ROOT))})

    (STAGE2_ROOT / "outputs" / "alignment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Aligned {len(manifest)} faces -> {OUT_ALIGNED}")


if __name__ == "__main__":
    main()
