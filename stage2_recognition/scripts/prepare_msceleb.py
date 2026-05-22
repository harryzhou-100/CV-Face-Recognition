#!/usr/bin/env python3
"""Download MS-Celeb-1M (MS1MV3) subset from HuggingFace WebDataset shards."""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

STAGE2_ROOT = Path(__file__).resolve().parents[1]
REPO = "gaunernst/ms1mv3-wds"
OUT_ROOT = STAGE2_ROOT / "data" / "ms1mv3_subset"


def extract_shard(tar_path: Path, out_root: Path) -> int:
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    label_path = out_root / "label.txt"
    groups: dict[str, dict[str, tarfile.TarInfo]] = defaultdict(dict)
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            key, ext = member.name.rsplit(".", 1)
            groups[key][ext] = member

    count = 0
    with label_path.open("a", encoding="utf-8") as lf:
        with tarfile.open(tar_path, "r") as tf:
            for key, parts in groups.items():
                if "jpg" not in parts or "cls" not in parts:
                    continue
                cls_raw = tf.extractfile(parts["cls"]).read().decode().strip()
                identity = int(cls_raw)
                person_dir = images_dir / f"{identity:06d}"
                person_dir.mkdir(exist_ok=True)
                out_rel = f"images/{identity:06d}/{key}.jpg"
                out_abs = out_root / out_rel
                with tf.extractfile(parts["jpg"]) as src, open(out_abs, "wb") as dst:
                    dst.write(src.read())
                lf.write(f"{out_rel}\t{identity}\n")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shards", type=int, default=6)
    parser.add_argument("--max-images", type=int, default=80000)
    args = parser.parse_args()

    cache = STAGE2_ROOT / "data" / "downloads" / "ms1mv3-wds"
    cache.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "label.txt").write_text("", encoding="utf-8")

    total = 0
    shard_meta = []
    for i in range(args.max_shards):
        if total >= args.max_images:
            break
        fname = f"ms1mv3-{i:04d}.tar"
        print(f"[download] {fname}")
        tar_path = Path(
            hf_hub_download(
                repo_id=REPO,
                filename=fname,
                repo_type="dataset",
                local_dir=str(cache),
            )
        )
        n = extract_shard(tar_path, OUT_ROOT)
        total += n
        shard_meta.append({"file": fname, "images": n})
        print(f"[extract] {fname}: {n} images (total {total})")
        if total >= args.max_images:
            break

    identities = set()
    with (OUT_ROOT / "label.txt").open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                identities.add(int(parts[1]))
    meta = {
        "shards": shard_meta,
        "total_images": total,
        "num_identities": len(identities),
        "source": "MS-Celeb-1M (MS1MV3 subset via gaunernst/ms1mv3-wds)",
    }
    (OUT_ROOT / "dataset_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"[done] {total} images, {len(identities)} identities -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
