#!/usr/bin/env python3
"""Live webcam / video face filter demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import FilterPipeline, build_preset
from src.effects.beauty import BeautyEffect
from src.effects.expression import ExpressionEffect
from src.effects.stickers import StickerEffect


def parse_effects(args) -> list:
    effects = []
    if args.beauty:
        effects.append(BeautyEffect(
            smooth_strength=args.smooth,
            whiten_strength=args.whiten,
            lipstick_strength=args.lipstick,
        ))
    if args.glasses:
        effects.append(StickerEffect("glasses"))
    if args.hat:
        effects.append(StickerEffect("hat"))
    if args.cat:
        effects.append(StickerEffect("cat_ears"))
    if args.expression:
        effects.append(ExpressionEffect())
    return effects


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time face filter demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, default=None, help="Use video file instead of camera")
    parser.add_argument("--preset", choices=["beauty", "glasses", "hat", "cat", "expression", "full"], default=None)
    parser.add_argument("--beauty", action="store_true")
    parser.add_argument("--glasses", action="store_true")
    parser.add_argument("--hat", action="store_true")
    parser.add_argument("--cat", action="store_true")
    parser.add_argument("--expression", action="store_true")
    parser.add_argument("--smooth", type=float, default=0.65)
    parser.add_argument("--whiten", type=float, default=0.3)
    parser.add_argument("--lipstick", type=float, default=0.6)
    parser.add_argument("--detect-scale", type=float, default=0.5, help="Downscale for detection (mobile opt)")
    parser.add_argument("--detect-every", type=int, default=1, help="Run detector every N frames")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--landmarks", action="store_true")
    parser.add_argument("--no-hud", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Save output video")
    args = parser.parse_args()

    if args.preset:
        pipeline = build_preset(args.preset, detect_scale=args.detect_scale)
    else:
        effects = parse_effects(args)
        if not effects:
            effects = parse_effects(argparse.Namespace(
                beauty=True, glasses=True, expression=True,
                hat=False, cat=False,
                smooth=args.smooth, whiten=args.whiten, lipstick=args.lipstick,
            ))
        pipeline = FilterPipeline(
            detect_scale=args.detect_scale,
            draw_landmarks=args.landmarks,
            effects=effects,
        )

    if args.video:
        cap = cv2.VideoCapture(str(args.video))
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise SystemExit("Cannot open video source")

    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
        writer = cv2.VideoWriter(str(args.output), fourcc, 25, (w, h))

    print("Press 'q' to quit. Keys: 1=beauty 2=glasses 3=hat 4=cat 5=expression")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            out = pipeline.process(frame, detect_every=args.detect_every)
            if not args.no_hud:
                out = pipeline.draw_hud(out)

            if writer:
                writer.write(out)

            cv2.imshow("Live Filter", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            for i, eff in enumerate(pipeline.effects):
                if key == ord(str(i + 1)):
                    eff.enabled = not eff.enabled
    finally:
        cap.release()
        if writer:
            writer.release()
        pipeline.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
