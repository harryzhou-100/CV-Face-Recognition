"""MediaPipe Face Landmarker index constants and geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Key landmark indices (478-point face mesh)
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
LEFT_EYEBROW = 70
RIGHT_EYEBROW = 300
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

# Face oval contour for skin/beauty masks
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Lip region for lipstick
LIP_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
    270, 269, 267, 0, 37, 39, 40, 185, 61,
]

LEFT_EYE_REGION = [33, 160, 158, 133, 153, 144, 33]
RIGHT_EYE_REGION = [362, 385, 387, 263, 373, 380, 362]


@dataclass
class FaceGeometry:
    """Pixel-space face geometry derived from normalized landmarks."""

    landmarks_px: np.ndarray  # (N, 2) float32
    landmarks_norm: list  # raw MediaPipe landmarks
    image_shape: tuple[int, int]  # (h, w)

    @property
    def h(self) -> int:
        return self.image_shape[0]

    @property
    def w(self) -> int:
        return self.image_shape[1]

    def point(self, idx: int) -> np.ndarray:
        return self.landmarks_px[idx]

    def midpoint(self, *indices: int) -> np.ndarray:
        pts = self.landmarks_px[list(indices)]
        return pts.mean(axis=0)

    def eye_centers(self) -> tuple[np.ndarray, np.ndarray]:
        left = self.midpoint(LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM)
        right = self.midpoint(RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)
        return left, right

    def inter_eye_distance(self) -> float:
        left, right = self.eye_centers()
        return float(np.linalg.norm(right - left))

    def face_angle_deg(self) -> float:
        left, right = self.eye_centers()
        dy = right[1] - left[1]
        dx = right[0] - left[0]
        return float(np.degrees(np.arctan2(dy, dx)))

    def polygon(self, indices: list[int]) -> np.ndarray:
        return self.landmarks_px[indices].astype(np.int32)

    def face_oval_mask(self, feather: int = 15) -> np.ndarray:
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, self.polygon(FACE_OVAL), 255)
        if feather > 0:
            k = feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return mask

    def lip_mask(self, feather: int = 5) -> np.ndarray:
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.fillPoly(mask, [self.polygon(LIP_OUTER)], 255)
        if feather > 0:
            k = feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return mask

    def roi_bbox(self, padding: float = 0.15) -> tuple[int, int, int, int]:
        xs = self.landmarks_px[:, 0]
        ys = self.landmarks_px[:, 1]
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        pw = (x2 - x1) * padding
        ph = (y2 - y1) * padding
        return (
            max(0, int(x1 - pw)),
            max(0, int(y1 - ph)),
            min(self.w, int(x2 + pw)),
            min(self.h, int(y2 + ph)),
        )


def landmarks_to_pixels(landmarks, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)
    return pts
