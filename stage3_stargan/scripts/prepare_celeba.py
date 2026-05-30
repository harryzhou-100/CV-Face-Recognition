#!/usr/bin/env python3
"""Download CelebA (aligned images + attributes) for StarGAN training."""
import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

# Mirrors (Google Drive is often rate-limited)
MIRRORS = {
    "img_align_celeba.zip": "https://cseweb.ucsd.edu/~weijian/static/datasets/celeba/img_align_celeba.zip",
    "list_attr_celeba.txt": "https://raw.githubusercontent.com/KaiserW/bald-recognition/master/dataset/celeba/list_attr_celeba.txt",
    "list_eval_partition.txt": "https://raw.githubusercontent.com/KaiserW/bald-recognition/master/dataset/celeba/list_eval_partition.txt",
}


def wget_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-c", "-O", str(dest), url]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="stage3_stargan/data/celeba",
        help="CelebA root directory",
    )
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    img_dir = root / "img_align_celeba"
    attr_file = root / "list_attr_celeba.txt"
    part_file = root / "list_eval_partition.txt"

    if not attr_file.exists():
        print("Downloading list_attr_celeba.txt ...")
        wget_download(MIRRORS["list_attr_celeba.txt"], attr_file)

    if not part_file.exists():
        print("Downloading list_eval_partition.txt ...")
        wget_download(MIRRORS["list_eval_partition.txt"], part_file)

    if not img_dir.exists() or len(list(img_dir.glob("*.jpg"))) < 1000:
        zip_path = root / "img_align_celeba.zip"
        if not zip_path.exists() or zip_path.stat().st_size < 1_000_000_000:
            if zip_path.exists():
                zip_path.unlink()
            print("Downloading img_align_celeba.zip (~1.3GB)...")
            wget_download(MIRRORS["img_align_celeba.zip"], zip_path)
        print("Extracting images...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        extracted = root / "img_align_celeba"
        if not extracted.exists():
            for d in root.iterdir():
                if d.is_dir() and "celeba" in d.name.lower() and d != extracted:
                    shutil.move(str(d), str(extracted))
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)

    n_imgs = len(list(img_dir.glob("*.jpg")))
    print(f"CelebA ready: {n_imgs} images at {img_dir}")
    print(f"Attributes: {attr_file}")
    print(f"Partition: {part_file}")


if __name__ == "__main__":
    main()
