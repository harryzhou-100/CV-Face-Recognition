"""Base class for face filter effects."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..face_detector import FaceResult


class Effect(ABC):
    name: str = "base"
    enabled: bool = True

    @abstractmethod
    def apply(self, frame: np.ndarray, faces: list[FaceResult]) -> np.ndarray:
        ...
