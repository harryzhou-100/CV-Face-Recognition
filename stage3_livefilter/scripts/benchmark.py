#!/usr/bin/env python3
"""Benchmark face filter pipeline on CPU with various optimization settings."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.effects.beauty import BeautyEffect
from src.effects.expression import ExpressionEffect
from src.effects.stickers import StickerEffect
from src.pipeline import FilterPipeline

OUTPUT_DIR = ROOT / "outputs" / "benchmark"


def make_test_frame(w: int = 640, h: int = 480) -> np.ndarray:
    img = np.full((h, w, 3), (210, 190, 170), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.ellipse(img, (cx, cy), (100, 130), 0, 0, 360, (235, 210, 195), -1)
    cv2.circle(img, (cx - 40, cy - 25), 15, (255, 255, 255), -1)
    cv2.circle(img, (cx + 40, cy - 25), 15, (255, 255, 255), -1)
    cv2.circle(img, (cx - 38, cy - 25), 6, (40, 40, 40), -1)
    cv2.circle(img, (cx + 42, cy - 25), 6, (40, 40, 40), -1)
    cv2.ellipse(img, (cx, cy + 45), (35, 12), 0, 0, 180, (160, 90, 90), 2)
    return img


def benchmark_config(
    name: str,
    pipeline: FilterPipeline,
    frame: np.ndarray,
    warmup: int,
    iterations: int,
    detect_every: int = 1,
) -> dict:
    pipeline.setup()
    for _ in range(warmup):
        pipeline.process(frame, detect_every=detect_every)

    detect_times, effect_times, total_times = [], [], []
    for _ in range(iterations):
        t0 = time.perf_counter()
        pipeline.process(frame, detect_every=detect_every)
        total_times.append((time.perf_counter() - t0) * 1000)
        detect_times.append(pipeline.stats.detect_ms)
        effect_times.append(pipeline.stats.effects_ms)

    pipeline.close()

    def stats(arr):
        arr = np.array(arr)
        return {
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "fps": float(1000.0 / arr.mean()),
        }

    return {
        "name": name,
        "detect_every": detect_every,
        "detect_scale": pipeline.detect_scale,
        "total": stats(total_times),
        "detect": stats(detect_times),
        "effects": stats(effect_times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark live filter pipeline")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "benchmark_results.json")
    args = parser.parse_args()

    frame = make_test_frame(args.width, args.height)

    configs = [
        ("detect_only_scale1.0", FilterPipeline(detect_scale=1.0, effects=[]), 1),
        ("detect_only_scale0.5", FilterPipeline(detect_scale=0.5, effects=[]), 1),
        ("detect_only_scale0.25", FilterPipeline(detect_scale=0.25, effects=[]), 1),
        ("detect_every2_scale0.5", FilterPipeline(detect_scale=0.5, effects=[]), 2),
        ("beauty_scale0.5", FilterPipeline(detect_scale=0.5, effects=[BeautyEffect()]), 1),
        ("stickers_scale0.5", FilterPipeline(detect_scale=0.5, effects=[StickerEffect("glasses")]), 1),
        ("expression_scale0.5", FilterPipeline(detect_scale=0.5, effects=[ExpressionEffect()]), 1),
        ("full_scale0.5", FilterPipeline(
            detect_scale=0.5,
            effects=[BeautyEffect(), StickerEffect("glasses"), ExpressionEffect()],
        ), 1),
        ("full_scale0.5_every2", FilterPipeline(
            detect_scale=0.5,
            effects=[BeautyEffect(), StickerEffect("glasses"), ExpressionEffect()],
        ), 2),
    ]

    results = {
        "platform": "CPU (MediaPipe XNNPACK)",
        "resolution": f"{args.width}x{args.height}",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "configs": [],
    }

    print(f"Benchmark: {args.width}x{args.height}, {args.iterations} iterations\n")
    for name, pipeline, detect_every in configs:
        print(f"  {name}...", end=" ", flush=True)
        r = benchmark_config(name, pipeline, frame, args.warmup, args.iterations, detect_every)
        results["configs"].append(r)
        print(f"{r['total']['fps']:.1f} FPS  (det={r['detect']['mean_ms']:.1f}ms fx={r['effects']['mean_ms']:.1f}ms)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults -> {args.output}")


if __name__ == "__main__":
    main()
