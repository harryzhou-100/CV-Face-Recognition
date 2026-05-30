"""Real-time face landmark detection using MediaPipe Face Landmarker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions, vision

from .landmarks import FaceGeometry, landmarks_to_pixels

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


@dataclass
class FaceResult:
    geometry: FaceGeometry
    blendshapes: dict[str, float] = field(default_factory=dict)


class FaceLandmarkDetector:
    """Wrapper around MediaPipe Face Landmarker with mobile-friendly options."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        num_faces: int = 1,
        detect_scale: float = 1.0,
        output_blendshapes: bool = True,
    ):
        self.detect_scale = detect_scale
        self.output_blendshapes = output_blendshapes
        model_path = str(model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_faces=num_faces,
            output_face_blendshapes=output_blendshapes,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        h, w = frame_bgr.shape[:2]
        if self.detect_scale != 1.0:
            sw = max(1, int(w * self.detect_scale))
            sh = max(1, int(h * self.detect_scale))
            small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            det_shape = (sh, sw)
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            det_shape = (h, w)

        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        faces: list[FaceResult] = []
        if not result.face_landmarks:
            return faces

        scale_x = w / det_shape[1]
        scale_y = h / det_shape[0]

        for i, lms in enumerate(result.face_landmarks):
            pts = landmarks_to_pixels(lms, det_shape)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y

            blendshapes: dict[str, float] = {}
            if self.output_blendshapes and result.face_blendshapes:
                for bs in result.face_blendshapes[i]:
                    blendshapes[bs.category_name] = bs.score

            geo = FaceGeometry(
                landmarks_px=pts,
                landmarks_norm=lms,
                image_shape=(h, w),
            )
            faces.append(FaceResult(geometry=geo, blendshapes=blendshapes))

        return faces

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
