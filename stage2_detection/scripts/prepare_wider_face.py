#!/usr/bin/env python3
"""Download WIDER FACE and convert annotations to COCO format for MMDetection."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

STAGE2_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = STAGE2_ROOT / "data" / "wider_face"
REPO_ID = "CUHK-CSE/wider_face"

HF_FILES = {
    "annot": "data/wider_face_split.zip",
    "train": "data/WIDER_train.zip",
    "val": "data/WIDER_val.zip",
}


def download_hf_assets(cache_dir: Path) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, relpath in HF_FILES.items():
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=relpath,
            repo_type="dataset",
            local_dir=str(cache_dir / "hf_cache"),
        )
        paths[key] = Path(local)
        print(f"[download] {key}: {local}")
    return paths


def extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    print(f"[extract] {zip_path.name} -> {dest}")


def parse_wider_gt(gt_file: Path) -> dict[str, list[list[float]]]:
    """Parse WIDER txt annotation -> {rel_image_path: [[x,y,w,h], ...]}."""
    boxes: dict[str, list[list[float]]] = {}
    with gt_file.open("r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            img_rel = line
            n_line = f.readline()
            if not n_line:
                break
            n_faces = int(n_line.strip())
            face_boxes: list[list[float]] = []
            # WIDER writes one placeholder line when n_faces == 0
            lines_to_read = n_faces if n_faces > 0 else 1
            for _ in range(lines_to_read):
                parts = f.readline().strip().split()
                if n_faces == 0:
                    continue
                if len(parts) < 4:
                    continue
                x, y, w, h = map(float, parts[:4])
                if w <= 0 or h <= 0:
                    continue
                if len(parts) >= 8 and int(float(parts[7])) == 1:
                    continue
                face_boxes.append([x, y, w, h])
            boxes[img_rel] = face_boxes
    return boxes


def to_coco(
    split: str,
    img_root: Path,
    gt_file: Path,
    out_json: Path,
) -> None:
    ann_map = parse_wider_gt(gt_file)
    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    for img_rel, bboxes in ann_map.items():
        img_path = img_root / img_rel
        if not img_path.exists():
            continue
        import cv2

        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        images.append(
            {
                "id": img_id,
                "file_name": img_rel.replace("\\", "/"),
                "width": w,
                "height": h,
            }
        )
        for box in bboxes:
            x, y, bw, bh = box
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            x = max(0, x)
            y = max(0, y)
            bw = x2 - x
            bh = y2 - y
            if bw < 1 or bh < 1:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x), float(y), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        img_id += 1

    coco = {
        "info": {"description": f"WIDER FACE {split} (COCO format)"},
        "licenses": [],
        "categories": [{"id": 1, "name": "face", "supercategory": "person"}],
        "images": images,
        "annotations": annotations,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(
        f"[coco] {split}: images={len(images)} "
        f"annotations={len(annotations)} -> {out_json}"
    )


def layout_dataset(raw_root: Path) -> None:
    """Normalize directory layout under data/wider_face."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("WIDER_train", "WIDER_val", "wider_face_split"):
        src = raw_root / name
        dst = DATA_ROOT / name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
        elif src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume zips already extracted under data/wider_face",
    )
    args = parser.parse_args()

    raw_root = STAGE2_ROOT / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        paths = download_hf_assets(STAGE2_ROOT / "data" / "downloads")
        for key, zp in paths.items():
            extract_zip(zp, raw_root)

    layout_dataset(raw_root)

    ann_dir = DATA_ROOT / "wider_face_split"
    train_gt = ann_dir / "wider_face_train_bbx_gt.txt"
    val_gt = ann_dir / "wider_face_val_bbx_gt.txt"
    if not train_gt.exists() or not val_gt.exists():
        raise FileNotFoundError(f"Missing annotation files under {ann_dir}")

    ann_out = DATA_ROOT / "annotations"
    to_coco(
        "train",
        DATA_ROOT / "WIDER_train" / "images",
        train_gt,
        ann_out / "instances_train.json",
    )
    to_coco(
        "val",
        DATA_ROOT / "WIDER_val" / "images",
        val_gt,
        ann_out / "instances_val.json",
    )

    # Symlink-friendly list files for tooling
    train_list = DATA_ROOT / "train.txt"
    val_list = DATA_ROOT / "val.txt"
    for split, gt_path, out_list in (
        ("train", train_gt, train_list),
        ("val", val_gt, val_list),
    ):
        ann_map = parse_wider_gt(gt_path)
        with out_list.open("w", encoding="utf-8") as f:
            for rel in sorted(ann_map.keys()):
                f.write(rel.replace("\\", "/") + "\n")
        print(f"[list] {split}: {len(ann_map)} entries -> {out_list}")


if __name__ == "__main__":
    main()
