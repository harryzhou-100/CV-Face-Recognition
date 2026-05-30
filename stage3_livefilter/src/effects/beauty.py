"""Beauty and makeup effects: skin smoothing, whitening, lipstick."""

from __future__ import annotations

import cv2
import numpy as np

from ..face_detector import FaceResult
from .base import Effect


class BeautyEffect(Effect):
    """ROI-based beauty pipeline optimized for real-time mobile use."""

    name = "beauty"

    def __init__(
        self,
        smooth_strength: float = 0.6,
        whiten_strength: float = 0.25,
        lipstick_color: tuple[int, int, int] = (50, 30, 180),
        lipstick_strength: float = 0.55,
    ):
        self.smooth_strength = smooth_strength
        self.whiten_strength = whiten_strength
        self.lipstick_color = lipstick_color
        self.lipstick_strength = lipstick_strength
        self.enable_smooth = True
        self.enable_whiten = True
        self.enable_lipstick = True

    def apply(self, frame: np.ndarray, faces: list[FaceResult]) -> np.ndarray:
        if not faces or not self.enabled:
            return frame

        out = frame.copy()
        for face in faces:
            geo = face.geometry
            x1, y1, x2, y2 = geo.roi_bbox(padding=0.1)
            roi = out[y1:y2, x1:x2].copy()
            rh, rw = roi.shape[:2]

            face_mask = geo.face_oval_mask(feather=11)
            roi_mask = face_mask[y1:y2, x1:x2]
            mask_f = roi_mask.astype(np.float32) / 255.0

            processed = roi.astype(np.float32)

            if self.enable_smooth and self.smooth_strength > 0:
                d = max(5, int(min(rh, rw) * 0.04)) | 1
                sigma = max(20, int(min(rh, rw) * 0.08))
                smooth = cv2.bilateralFilter(roi, d, sigma, sigma).astype(np.float32)
                alpha = self.smooth_strength * mask_f[..., None]
                processed = processed * (1 - alpha) + smooth * alpha

            if self.enable_whiten and self.whiten_strength > 0:
                lab = cv2.cvtColor(processed.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
                lab[:, :, 0] = np.clip(
                    lab[:, :, 0] + self.whiten_strength * 40 * mask_f, 0, 255
                )
                whitened = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
                alpha = self.whiten_strength * mask_f[..., None]
                processed = processed * (1 - alpha) + whitened * alpha

            if self.enable_lipstick and self.lipstick_strength > 0:
                lip_mask = geo.lip_mask(feather=7)
                roi_lip = lip_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
                lip_color = np.array(self.lipstick_color, dtype=np.float32)
                alpha = self.lipstick_strength * roi_lip[..., None]
                processed = processed * (1 - alpha) + lip_color * alpha

            out[y1:y2, x1:x2] = np.clip(processed, 0, 255).astype(np.uint8)

        return out
