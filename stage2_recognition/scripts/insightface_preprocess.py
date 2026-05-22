"""InsightFace ArcFace input preprocessing (aligned with arcface_onnx.ArcFaceONNX)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def insightface_blob_from_image(
    img_bgr: np.ndarray,
    image_size: int = 112,
    input_mean: float = 127.5,
    input_std: float = 127.5,
) -> np.ndarray:
    """Same as ArcFaceONNX.get_feat: BGR in, NCHW float32 out."""
    blob = cv2.dnn.blobFromImage(
        img_bgr,
        1.0 / input_std,
        (image_size, image_size),
        (input_mean, input_mean, input_mean),
        swapRB=True,
    )
    return blob.astype(np.float32)


def load_bgr(path: Path, image_size: int) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    if img.shape[0] != image_size or img.shape[1] != image_size:
        img = cv2.resize(img, (image_size, image_size))
    return img


def image_to_torch_tensor(
    path: Path,
    image_size: int = 112,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    img = load_bgr(path, image_size)
    if img is None:
        return None
    blob = insightface_blob_from_image(img, image_size)
    t = torch.from_numpy(blob)
    if device is not None:
        t = t.to(device)
    return t
