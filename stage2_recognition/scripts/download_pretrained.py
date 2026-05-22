#!/usr/bin/env python3
"""Download standard MS1MV3 ArcFace R50 backbone weights (InsightFace ArcFace-Torch)."""

from __future__ import annotations

import subprocess
from pathlib import Path

STAGE2_ROOT = Path(__file__).resolve().parents[1]
MODELS = STAGE2_ROOT / "models"
URL = "https://huggingface.co/camenduru/show/resolve/main/models/arcface/ms1mv3_arcface_r50_fp16.pth"
OUT = MODELS / "ms1mv3_arcface_r50_backbone.pth"
LINK = MODELS / "resnet50_arcface_pretrained.pth"


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        subprocess.run(["wget", "-c", URL, "-O", str(OUT)], check=True)
    if not LINK.exists():
        LINK.symlink_to(OUT.name)
    print(f"Saved: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
