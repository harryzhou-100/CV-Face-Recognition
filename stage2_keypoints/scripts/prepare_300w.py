#!/usr/bin/env python3
"""Prepare 300W dataset for MMPose (annotations + images)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

STAGE2_ROOT = Path(__file__).resolve().parents[1]
DATA_300W = STAGE2_ROOT / "data" / "300w"

ANNOT_URL = "https://download.openmmlab.com/mmpose/datasets/300w_annotations.tar"
# VGG mirror with 300W sub-datasets (afw/helen/lfpw/ibug images)
IMAGES_URL = "http://www.robots.ox.ac.uk/~vgg/research/DVE/data/datasets/300w.tar.gz"

IBUG_ZIPS = {
    "afw": "https://ibug.doc.ic.ac.uk/download/annotations/afw.zip",
    "helen": "https://ibug.doc.ic.ac.uk/download/annotations/helen.zip",
    "lfpw": "https://ibug.doc.ic.ac.uk/download/annotations/lfpw.zip",
    "ibug": "https://ibug.doc.ic.ac.uk/download/annotations/ibug.zip",
}


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] exists: {dest}")
        return dest
    print(f"[download] {url}")
    subprocess.run(
        ["wget", "-c", url, "-O", str(dest)],
        check=True,
    )
    return dest


def extract_tar(tar_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(dest)
    print(f"[extract] {tar_path.name} -> {dest}")


def layout_from_vgg_tar(raw_dir: Path, images_dst: Path) -> None:
    """Map VGG 300w.tar.gz layout to MMPose images/ tree."""
    images_dst.mkdir(parents=True, exist_ok=True)
    roots = list(raw_dir.rglob("afw"))
    root = roots[0].parent if roots else raw_dir
    for name in ("afw", "helen", "lfpw", "ibug"):
        src = root / name
        if src.exists():
            dst = images_dst / name
            if not dst.exists():
                shutil.copytree(src, dst)
                print(f"[layout] {src} -> {dst}")
    # Challenge test images live under nested 300W/{01_Indoor,02_Outdoor}
    for test_src in raw_dir.rglob("300W"):
        if (test_src / "01_Indoor").exists():
            dst = images_dst / "Test"
            dst.mkdir(parents=True, exist_ok=True)
            for sub in ("01_Indoor", "02_Outdoor"):
                s = test_src / sub
                d = dst / sub
                if s.exists() and not d.exists():
                    shutil.copytree(s, d)
                    print(f"[layout] {s} -> {d}")
            break
    # Fix ibug filename with stray space
    ibug = images_dst / "ibug"
    for p in ibug.glob("* *"):
        new = ibug / p.name.replace(" ", "")
        if not new.exists():
            p.rename(new)


def layout_from_ibug_zips(zip_dir: Path, images_dst: Path) -> None:
    images_dst.mkdir(parents=True, exist_ok=True)
    for name, url in IBUG_ZIPS.items():
        zp = zip_dir / f"{name}.zip"
        if not zp.exists():
            try:
                download(url, zp)
            except subprocess.CalledProcessError:
                print(f"[warn] failed to download {name}")
                continue
        out = images_dst / name
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(out)
        print(f"[extract] {name}.zip -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--images-source", choices=["vgg", "ibug", "auto"], default="auto")
    args = parser.parse_args()

    DATA_300W.mkdir(parents=True, exist_ok=True)
    dl_dir = STAGE2_ROOT / "data" / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)

    ann_tar = dl_dir / "300w_annotations.tar"
    download(ANNOT_URL, ann_tar)
    ann_root = DATA_300W
    extract_tar(ann_tar, ann_root)
    # annotations may unpack to 300w/annotations or annotations/
    if (DATA_300W / "300w" / "annotations").exists():
        inner = DATA_300W / "300w"
        for sub in ("annotations", "images"):
            src = inner / sub
            if src.exists():
                dst = DATA_300W / sub
                if not dst.exists():
                    shutil.move(str(src), str(dst))

    if args.skip_images:
        return

    images_dst = DATA_300W / "images"
    if images_dst.exists() and any(images_dst.iterdir()):
        print(f"[skip] images already present under {images_dst}")
        return

    if args.images_source in ("vgg", "auto"):
        vgg_tar = dl_dir / "300w.tar.gz"
        try:
            download(IMAGES_URL, vgg_tar)
            raw = dl_dir / "300w_raw"
            extract_tar(vgg_tar, raw)
            layout_from_vgg_tar(raw, images_dst)
            if any(images_dst.iterdir()):
                return
        except subprocess.CalledProcessError as e:
            print(f"[warn] VGG mirror failed: {e}")

    if args.images_source in ("ibug", "auto"):
        layout_from_ibug_zips(dl_dir, images_dst)

    n_imgs = sum(1 for _ in images_dst.rglob("*") if _.suffix.lower() in {".jpg", ".png", ".jpeg"})
    print(f"[done] images found: {n_imgs}")


if __name__ == "__main__":
    main()
