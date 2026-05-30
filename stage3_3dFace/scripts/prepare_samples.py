#!/usr/bin/env python3
"""Copy demo / project face images into data/samples."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples"
OUT.mkdir(parents=True, exist_ok=True)

sources = [
    ROOT / "third_party" / "3DDFA_V2" / "examples" / "inputs" / "emma.jpg",
    ROOT / "third_party" / "3DDFA_V2" / "examples" / "inputs" / "JianzhuGuo.jpg",
    Path("/root/workspace/CV-Face-Recognition/stage3_stargan/outputs/edited/Blond_Hair/sample_00.jpg"),
]

for i, src in enumerate(sources, 1):
    if src.exists():
        dst = OUT / f"face_{i:02d}{src.suffix}"
        shutil.copy2(src, dst)
        print(f"Copied {src.name} -> {dst}")
