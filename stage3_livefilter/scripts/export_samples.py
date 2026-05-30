#!/usr/bin/env python3
"""Export static effect comparison images for the report."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.effects.beauty import BeautyEffect
from src.effects.expression import ExpressionEffect
from src.effects.stickers import StickerEffect
from src.pipeline import FilterPipeline

SAMPLE = ROOT / "assets" / "samples" / "face_00.jpg"
OUT = ROOT / "outputs" / "demo"


def main() -> None:
    if not SAMPLE.is_file():
        print(f"No sample: {SAMPLE}")
        return

    frame = cv2.imread(str(SAMPLE))
    OUT.mkdir(parents=True, exist_ok=True)

    configs = [
        ("01_original", []),
        ("02_beauty", [BeautyEffect()]),
        ("03_glasses", [StickerEffect("glasses")]),
        ("04_hat", [StickerEffect("hat")]),
        ("05_full", [BeautyEffect(), StickerEffect("glasses"), ExpressionEffect()]),
    ]

    for name, effects in configs:
        pipeline = FilterPipeline(detect_scale=0.75, effects=effects)
        out = pipeline.process(frame)
        cv2.imwrite(str(OUT / f"{name}.jpg"), out)
        pipeline.close()
        print(f"  saved {name}.jpg")


if __name__ == "__main__":
    main()
