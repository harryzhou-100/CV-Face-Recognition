#!/usr/bin/env python3
"""Generate demo video showcasing stickers, beauty, and expression effects."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.effects.beauty import BeautyEffect
from src.effects.expression import ExpressionEffect
from src.effects.stickers import StickerEffect
from src.pipeline import FilterPipeline

SAMPLES_DIR = ROOT / "assets" / "samples"
OUTPUT_DIR = ROOT / "outputs" / "demo"

# Public-domain / sample face images for demo
SAMPLE_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Face_of_a_man_in_China%2C_2012.jpg/440px-Face_of_a_man_in_China%2C_2012.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Default_pfp.svg/440px-Default_pfp.svg.png",
]


def ensure_samples() -> list[Path]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, url in enumerate(SAMPLE_URLS):
        dst = SAMPLES_DIR / f"face_{i:02d}.jpg"
        if not dst.exists() or dst.stat().st_size < 5000:
            try:
                print(f"  downloading {url[:60]}...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    dst.write_bytes(resp.read())
            except Exception as e:
                print(f"  skip {url}: {e}")
                continue
        if dst.exists() and dst.stat().st_size > 5000:
            paths.append(dst)
    return paths


def synthetic_face_sequence(n_frames: int = 90, size: int = 480) -> list[np.ndarray]:
    """Fallback: animated synthetic portrait-like frames when no samples available."""
    frames = []
    for i in range(n_frames):
        img = np.full((size, size, 3), (220, 200, 180), dtype=np.uint8)
        cx, cy = size // 2, size // 2
        # face oval
        cv2.ellipse(img, (cx, cy), (120, 150), 0, 0, 360, (240, 210, 190), -1)
        # eyes
        eye_y = cy - 30
        for ex in (cx - 45, cx + 45):
            cv2.circle(img, (ex, eye_y), 18, (255, 255, 255), -1)
            cv2.circle(img, (ex + int(3 * np.sin(i * 0.1)), eye_y), 8, (50, 50, 50), -1)
        # mouth animation
        smile = 0.3 + 0.4 * abs(np.sin(i * 0.08))
        mouth_w = int(40 + 20 * smile)
        cv2.ellipse(img, (cx, cy + 50), (mouth_w, int(15 * smile + 5)), 0, 0, 180, (180, 100, 100), 2)
        # slight motion
        M = np.float32([[1, 0, 5 * np.sin(i * 0.05)], [0, 1, 3 * np.cos(i * 0.07)]])
        img = cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
        frames.append(img)
    return frames


def load_or_create_frames(sample_paths: list[Path], n_frames: int) -> list[np.ndarray]:
    if sample_paths:
        img = cv2.imread(str(sample_paths[0]))
        if img is not None:
            h, w = img.shape[:2]
            target = 480
            scale = target / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            frames = []
            for i in range(n_frames):
                angle = 3 * np.sin(i * 0.05)
                M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1.0)
                f = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
                # simulate expression by slight warp
                frames.append(f)
            return frames
    return synthetic_face_sequence(n_frames)


def add_title(frame: np.ndarray, title: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(out, title, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def render_segment(
    base_frames: list[np.ndarray],
    pipeline: FilterPipeline,
    title: str,
    detect_every: int = 1,
) -> list[np.ndarray]:
    segment = []
    pipeline.setup()
    for frame in base_frames:
        out = pipeline.process(frame, detect_every=detect_every)
        out = pipeline.draw_hud(out)
        segment.append(add_title(out, title))
    return segment


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate filter demo video")
    parser.add_argument("--frames", type=int, default=60, help="Frames per segment")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--detect-scale", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "demo_video.mp4")
    args = parser.parse_args()

    print("Preparing sample images...")
    samples = ensure_samples()
    base_frames = load_or_create_frames(samples, args.frames)
    fh, fw = base_frames[0].shape[:2]

    segments = [
        ("Original", FilterPipeline(detect_scale=args.detect_scale, effects=[])),
        ("Beauty: Smooth + Whiten + Lipstick", FilterPipeline(
            detect_scale=args.detect_scale,
            effects=[BeautyEffect(smooth_strength=0.7, whiten_strength=0.35, lipstick_strength=0.65)],
        )),
        ("Sticker: Glasses", FilterPipeline(
            detect_scale=args.detect_scale,
            effects=[StickerEffect("glasses")],
        )),
        ("Sticker: Hat", FilterPipeline(
            detect_scale=args.detect_scale,
            effects=[StickerEffect("hat")],
        )),
        ("Sticker: Cat Ears", FilterPipeline(
            detect_scale=args.detect_scale,
            effects=[StickerEffect("cat_ears")],
        )),
        ("Full: Beauty + Glasses + Expression", FilterPipeline(
            detect_scale=args.detect_scale,
            effects=[BeautyEffect(), StickerEffect("glasses"), ExpressionEffect()],
        )),
    ]

    all_frames: list[np.ndarray] = []
    for title, pipeline in segments:
        print(f"Rendering: {title}")
        all_frames.extend(render_segment(base_frames[: args.frames // 2 + 15], pipeline, title))

    for p in segments:
        p[1].close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (fw, fh + 0))
    for f in all_frames:
        writer.write(f)
    writer.release()

    print(f"Demo video saved: {args.output}  ({len(all_frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
