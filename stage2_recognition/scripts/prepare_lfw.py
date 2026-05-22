#!/usr/bin/env python3
"""Download and prepare LFW for verification."""

from __future__ import annotations

import argparse
import subprocess
import tarfile
from pathlib import Path

STAGE2_ROOT = Path(__file__).resolve().parents[1]
LFW_URL = "https://ndownloader.figshare.com/files/5976018"
OUT = STAGE2_ROOT / "data" / "lfw"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flat-dir",
        type=Path,
        default=STAGE2_ROOT / "data" / "lfw_flat",
        help="Also export flat jpg dir for simple pair scripts",
    )
    parser.add_argument(
        "--use-sklearn",
        action="store_true",
        help="Download via sklearn.datasets.fetch_lfw_people",
    )
    args = parser.parse_args()

    if args.use_sklearn:
        from sklearn.datasets import fetch_lfw_people

        home = str(STAGE2_ROOT / "data" / "sklearn_cache")
        fetch_lfw_people(
            data_home=home, funneled=True, download_if_missing=True
        )
        lfw_nested = Path(home) / "lfw_home" / "lfw_funneled"
    else:
        dl = STAGE2_ROOT / "data" / "downloads" / "lfw.tgz"
        dl.parent.mkdir(parents=True, exist_ok=True)
        if not dl.exists():
            subprocess.run(["wget", "-c", LFW_URL, "-O", str(dl)], check=True)
        OUT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dl, "r:gz") as tf:
            tf.extractall(OUT)
        lfw_nested = OUT / "lfw" / "lfw"
        if not lfw_nested.exists():
            lfw_nested = OUT / "lfw"
    print(f"[done] LFW at {lfw_nested}")

    # Flat export for compatibility with existing pair builder
    args.flat_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in lfw_nested.rglob("*.jpg"):
        dst = args.flat_dir / f"{img.parent.name}_{img.name}"
        if not dst.exists():
            dst.symlink_to(img.resolve())
        n += 1
    print(f"[flat] {n} symlinks -> {args.flat_dir}")


if __name__ == "__main__":
    main()
