"""Expression-driven interactive animations using blendshapes."""

from __future__ import annotations

import cv2
import numpy as np

from ..face_detector import FaceResult
from ..landmarks import LEFT_EYE_OUTER, RIGHT_EYE_OUTER, UPPER_LIP
from .base import Effect


class ExpressionEffect(Effect):
    """Trigger visual effects based on facial expressions (blendshapes)."""

    name = "expression"

    def __init__(self):
        self._anim_phase = 0.0
        self.show_hearts = True
        self.show_sparkles = True
        self.show_mouth_open = True

    def apply(self, frame: np.ndarray, faces: list[FaceResult]) -> np.ndarray:
        if not faces or not self.enabled:
            return frame

        out = frame
        self._anim_phase += 0.15

        for face in faces:
            bs = face.blendshapes
            geo = face.geometry
            smile = bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)
            jaw_open = bs.get("jawOpen", 0)
            blink_l = bs.get("eyeBlinkLeft", 0)
            blink_r = bs.get("eyeBlinkRight", 0)

            if self.show_hearts and smile > 0.45:
                out = self._draw_hearts(out, geo, intensity=min(1.0, smile))

            if self.show_sparkles and (blink_l > 0.5 or blink_r > 0.5):
                eye = geo.point(LEFT_EYE_OUTER if blink_l > blink_r else RIGHT_EYE_OUTER)
                out = self._draw_sparkles(out, eye, self._anim_phase)

            if self.show_mouth_open and jaw_open > 0.35:
                lip = geo.point(UPPER_LIP)
                out = self._draw_note(out, lip, self._anim_phase, jaw_open)

        return out

    def _draw_hearts(self, frame: np.ndarray, geo, intensity: float) -> np.ndarray:
        out = frame.copy()
        left, right = geo.eye_centers()
        d = geo.inter_eye_distance() * 0.25
        color = (80, 80, 255)
        for cx, cy in [(left[0] - d, left[1] - d), (right[0] + d, right[1] - d)]:
            offset = int(8 * np.sin(self._anim_phase) * intensity)
            self._heart(out, int(cx), int(cy + offset), int(d * intensity), color)
        return out

    @staticmethod
    def _heart(img: np.ndarray, cx: int, cy: int, size: int, color: tuple) -> None:
        if size < 3:
            return
        for t in np.linspace(0, 2 * np.pi, 30):
            x = int(size * 16 * np.sin(t) ** 3 / 16)
            y = int(-size * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) / 16)
            cv2.circle(img, (cx + x, cy + y), max(1, size // 8), color, -1, cv2.LINE_AA)

    def _draw_sparkles(self, frame: np.ndarray, center: np.ndarray, phase: float) -> np.ndarray:
        out = frame.copy()
        cx, cy = int(center[0]), int(center[1])
        for i in range(5):
            angle = phase + i * 1.2
            r = 15 + 8 * np.sin(phase + i)
            px = int(cx + r * np.cos(angle))
            py = int(cy + r * np.sin(angle))
            cv2.line(out, (px - 4, py), (px + 4, py), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(out, (px, py - 4), (px, py + 4), (0, 255, 255), 1, cv2.LINE_AA)
        return out

    def _draw_note(self, frame: np.ndarray, pos: np.ndarray, phase: float, intensity: float) -> np.ndarray:
        out = frame.copy()
        cx = int(pos[0] + 30 * np.sin(phase))
        cy = int(pos[1] - 40 - 10 * abs(np.sin(phase)))
        r = max(3, int(8 * intensity))
        cv2.circle(out, (cx, cy), r, (200, 200, 0), -1, cv2.LINE_AA)
        cv2.line(out, (cx + r, cy), (cx + r, cy - r * 3), (200, 200, 0), 2, cv2.LINE_AA)
        return out
