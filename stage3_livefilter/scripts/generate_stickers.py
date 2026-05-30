"""Generate PNG sticker assets for glasses, hat, and cat ears."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "stickers"


def _save_rgba(name: str, img: np.ndarray) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / f"{name}.png"), img)
    print(f"  saved {name}.png  shape={img.shape}")


def make_glasses(w: int = 400, h: int = 120) -> np.ndarray:
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    cx1, cx2 = w // 4, 3 * w // 4
    cy = h // 2
    r = h // 3
    for cx in (cx1, cx2):
        cv2.ellipse(canvas, (cx, cy), (r, int(r * 0.85)), 0, 0, 360, (40, 40, 40, 220), 4, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy), (r - 6, int(r * 0.85) - 6), 0, 0, 360, (180, 200, 255, 80), -1, cv2.LINE_AA)
    cv2.line(canvas, (cx1 + r, cy), (cx2 - r, cy), (40, 40, 40, 255), 4, cv2.LINE_AA)
    # temple arms
    cv2.line(canvas, (cx1 - r, cy), (10, cy - 10), (40, 40, 40, 255), 4, cv2.LINE_AA)
    cv2.line(canvas, (cx2 + r, cy), (w - 10, cy - 10), (40, 40, 40, 255), 4, cv2.LINE_AA)
    return canvas


def make_hat(w: int = 300, h: int = 200) -> np.ndarray:
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    brim_y = int(h * 0.65)
    cv2.ellipse(canvas, (w // 2, brim_y), (w // 2 - 10, 18), 0, 0, 360, (30, 30, 150, 255), -1, cv2.LINE_AA)
    pts = np.array([[w // 2 - 80, brim_y], [w // 2 + 80, brim_y], [w // 2 + 55, 20], [w // 2 - 55, 20]], np.int32)
    cv2.fillPoly(canvas, [pts], (50, 50, 200, 240))
    cv2.rectangle(canvas, (w // 2 - 85, brim_y - 8), (w // 2 + 85, brim_y + 4), (200, 200, 200, 255), -1)
    return canvas


def make_cat_ears(w: int = 300, h: int = 180) -> np.ndarray:
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    left = np.array([[20, h - 10], [80, h - 10], [60, 20]], np.int32)
    right = np.array([[w - 20, h - 10], [w - 80, h - 10], [w - 60, 20]], np.int32)
    cv2.fillPoly(canvas, [left], (180, 140, 255, 230))
    cv2.fillPoly(canvas, [right], (180, 140, 255, 230))
    inner_l = np.array([[35, h - 15], [75, h - 15], [62, 45]], np.int32)
    inner_r = np.array([[w - 35, h - 15], [w - 75, h - 15], [w - 62, 45]], np.int32)
    cv2.fillPoly(canvas, [inner_l], (255, 180, 220, 200))
    cv2.fillPoly(canvas, [inner_r], (255, 180, 220, 200))
    return canvas


def main() -> None:
    print(f"Generating stickers -> {OUT_DIR}")
    _save_rgba("glasses", make_glasses())
    _save_rgba("hat", make_hat())
    _save_rgba("cat_ears", make_cat_ears())


if __name__ == "__main__":
    main()
