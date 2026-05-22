"""Face alignment utilities (68 landmarks -> affine warp)."""

from __future__ import annotations

import cv2
import numpy as np

# 68-point indices used for similarity alignment (left eye, right eye, nose, mouth corners)
IDX_LEFT_EYE = slice(36, 42)
IDX_RIGHT_EYE = slice(42, 48)
IDX_NOSE = 30
IDX_MOUTH_L = 48
IDX_MOUTH_R = 54

# Reference template on 112x112 (common face recognition layout)
REF_TEMPLATE_112 = np.array(
    [
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ],
    dtype=np.float32,
)


def landmarks_68_to_5(pts68: np.ndarray) -> np.ndarray:
    """Convert 68x2 landmarks to 5 reference points."""
    pts68 = np.asarray(pts68, dtype=np.float32).reshape(68, 2)
    left_eye = pts68[IDX_LEFT_EYE].mean(axis=0)
    right_eye = pts68[IDX_RIGHT_EYE].mean(axis=0)
    nose = pts68[IDX_NOSE]
    mouth_l = pts68[IDX_MOUTH_L]
    mouth_r = pts68[IDX_MOUTH_R]
    return np.stack([left_eye, right_eye, nose, mouth_l, mouth_r], axis=0)


def get_reference_points(output_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    w, h = output_size
    scale = np.array([w / 112.0, h / 112.0], dtype=np.float32)
    return REF_TEMPLATE_112 * scale


def align_face(
    image: np.ndarray,
    landmarks_68: np.ndarray,
    output_size: tuple[int, int] = (256, 256),
) -> tuple[np.ndarray, np.ndarray]:
    """Warp face to canonical pose via partial affine (similarity) transform."""
    src = landmarks_68_to_5(landmarks_68)
    dst = get_reference_points(output_size)
    mat, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if mat is None:
        mat = cv2.getAffineTransform(src[:3], dst[:3])
    aligned = cv2.warpAffine(
        image,
        mat,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned, mat


def draw_landmarks_68(
    image: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    radius: int = 2,
) -> np.ndarray:
    vis = image.copy()
    pts = np.asarray(pts).reshape(-1, 2)
    for x, y in pts:
        cv2.circle(vis, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return vis
