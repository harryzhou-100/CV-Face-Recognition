"""Dynamic sticker overlay: glasses, hat, etc."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..face_detector import FaceResult
from ..landmarks import FOREHEAD, LEFT_EYE_OUTER, NOSE_TIP, RIGHT_EYE_OUTER
from .base import Effect

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "stickers"


def _load_rgba(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack([img, alpha])
    return img


def _overlay_rgba(frame: np.ndarray, sticker: np.ndarray, cx: float, cy: float, scale: float, angle_deg: float) -> np.ndarray:
    h, w = sticker.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(sticker, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    center = (new_w // 2, new_h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        resized, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    fh, fw = frame.shape[:2]
    x1 = int(cx - new_w // 2)
    y1 = int(cy - new_h // 2)
    x2 = x1 + new_w
    y2 = y1 + new_h

    sx1, sy1 = max(0, -x1), max(0, -y1)
    sx2 = new_w - max(0, x2 - fw)
    sy2 = new_h - max(0, y2 - fh)
    dx1, dy1 = max(0, x1), max(0, y1)
    dx2 = min(fw, x2)
    dy2 = min(fh, y2)

    if sx2 <= sx1 or sy2 <= sy1 or dx2 <= dx1 or dy2 <= dy1:
        return frame

    sticker_crop = rotated[sy1:sy2, sx1:sx2]
    frame_crop = frame[dy1:dy2, dx1:dx2]
    alpha = sticker_crop[:, :, 3:4].astype(np.float32) / 255.0
    rgb = sticker_crop[:, :, :3].astype(np.float32)
    blended = frame_crop.astype(np.float32) * (1 - alpha) + rgb * alpha
    out = frame.copy()
    out[dy1:dy2, dx1:dx2] = blended.astype(np.uint8)
    return out


class StickerEffect(Effect):
    name = "stickers"

    def __init__(self, sticker_name: str = "glasses"):
        self.sticker_name = sticker_name
        self._cache: dict[str, np.ndarray] = {}

    def _get_sticker(self, name: str) -> np.ndarray | None:
        if name not in self._cache:
            self._cache[name] = _load_rgba(ASSETS_DIR / f"{name}.png")
        return self._cache[name]

    def apply(self, frame: np.ndarray, faces: list[FaceResult]) -> np.ndarray:
        if not faces or not self.enabled:
            return frame

        sticker = self._get_sticker(self.sticker_name)
        if sticker is None:
            return frame

        out = frame
        for face in faces:
            geo = face.geometry
            angle = geo.face_angle_deg()

            if self.sticker_name == "glasses":
                left, right = geo.eye_centers()
                cx = (left[0] + right[0]) / 2
                cy = (left[1] + right[1]) / 2 - geo.inter_eye_distance() * 0.05
                scale = geo.inter_eye_distance() / sticker.shape[1] * 2.2
            elif self.sticker_name == "hat":
                forehead = geo.point(FOREHEAD)
                nose = geo.point(NOSE_TIP)
                face_h = abs(nose[1] - forehead[1]) * 2.5
                cx = forehead[0]
                cy = forehead[1] - face_h * 0.35
                scale = face_h / sticker.shape[0] * 1.1
            elif self.sticker_name == "cat_ears":
                left_eye = geo.point(LEFT_EYE_OUTER)
                right_eye = geo.point(RIGHT_EYE_OUTER)
                cx = (left_eye[0] + right_eye[0]) / 2
                cy = min(left_eye[1], right_eye[1]) - geo.inter_eye_distance() * 0.6
                scale = geo.inter_eye_distance() / sticker.shape[1] * 2.5
            else:
                cx, cy = geo.point(NOSE_TIP)
                scale = geo.inter_eye_distance() / sticker.shape[1] * 2.0

            out = _overlay_rgba(out, sticker, cx, cy, scale, angle)

        return out
