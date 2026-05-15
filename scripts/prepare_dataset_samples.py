"""Copy 10-20 random images from host data/ into project samples/ for quick tests."""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def copy_samples(src_images: list[Path], dst_dir: Path, count: int, seed: int) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for old in dst_dir.iterdir():
        if old.is_file():
            old.unlink()

    if not src_images:
        return 0

    rng = random.Random(seed)
    picks = rng.sample(src_images, min(count, len(src_images)))
    for src in picks:
        shutil.copy2(src, dst_dir / src.name)
    return len(picks)


def find_celeba_images(data_root: Path) -> Path | None:
    candidates = [
        data_root / "celeba" / "img_align_celeba",
        data_root / "celeba" / "Img" / "img_align_celeba",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return None


def find_lfw_images(data_root: Path) -> Path | None:
    candidates = [
        data_root / "lfw" / "lfw",
        data_root / "lfw",
    ]
    for path in candidates:
        if path.is_dir() and any(path.glob("*/*.jpg")):
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy dataset samples into the project")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Host dataset root (default: ../data next to project)",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs" / "samples",
        help="Project sample output directory (default: docs/samples)",
    )
    parser.add_argument("-n", "--count", type=int, default=15, help="Images per dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    celeba_src = find_celeba_images(args.data_root)
    lfw_src = find_lfw_images(args.data_root)

    if celeba_src is None and lfw_src is None:
        raise SystemExit(
            f"No CelebA/LFW images found under {args.data_root}. "
            "See docs/DATA_SETUP.md for download instructions."
        )

    if celeba_src:
        n = copy_samples(
            collect_images(celeba_src),
            args.samples_dir / "celeba",
            args.count,
            args.seed,
        )
        print(f"CelebA: copied {n} images -> {args.samples_dir / 'celeba'}")
    else:
        print("CelebA: skipped (img_align_celeba not found)")

    if lfw_src:
        n = copy_samples(
            collect_images(lfw_src),
            args.samples_dir / "lfw",
            args.count,
            args.seed + 1,
        )
        print(f"LFW: copied {n} images -> {args.samples_dir / 'lfw'}")
    else:
        print("LFW: skipped (lfw folder structure not found)")


if __name__ == "__main__":
    main()
