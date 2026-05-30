"""Real-time face filter processing pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .effects.base import Effect
from .effects.beauty import BeautyEffect
from .effects.expression import ExpressionEffect
from .effects.stickers import StickerEffect
from .face_detector import FaceLandmarkDetector, FaceResult


@dataclass
class PipelineStats:
    fps: float = 0.0
    detect_ms: float = 0.0
    effects_ms: float = 0.0
    total_ms: float = 0.0
    frame_count: int = 0


@dataclass
class FilterPipeline:
    """Composable real-time filter pipeline with performance knobs."""

    detect_scale: float = 0.5
    draw_landmarks: bool = False
    effects: list[Effect] = field(default_factory=list)
    stats: PipelineStats = field(default_factory=PipelineStats)

    _detector: FaceLandmarkDetector | None = field(default=None, repr=False)
    _last_faces: list[FaceResult] = field(default_factory=list, repr=False)
    _frame_idx: int = field(default=0, repr=False)
    _fps_ema: float = field(default=0.0, repr=False)

    def setup(self) -> None:
        if self._detector is None:
            need_bs = any(isinstance(e, ExpressionEffect) and e.enabled for e in self.effects)
            self._detector = FaceLandmarkDetector(
                detect_scale=self.detect_scale,
                output_blendshapes=need_bs,
            )

    def close(self) -> None:
        if self._detector:
            self._detector.close()
            self._detector = None

    def process(self, frame_bgr: np.ndarray, detect_every: int = 1) -> np.ndarray:
        t0 = time.perf_counter()
        self.setup()
        assert self._detector is not None

        run_detect = (self._frame_idx % detect_every) == 0
        if run_detect:
            t_d0 = time.perf_counter()
            self._last_faces = self._detector.detect(frame_bgr)
            detect_ms = (time.perf_counter() - t_d0) * 1000
        else:
            detect_ms = 0.0

        out = frame_bgr
        t_e0 = time.perf_counter()
        for effect in self.effects:
            if effect.enabled:
                out = effect.apply(out, self._last_faces)
        effects_ms = (time.perf_counter() - t_e0) * 1000

        if self.draw_landmarks:
            out = self._draw_landmarks(out, self._last_faces)

        total_ms = (time.perf_counter() - t0) * 1000
        self._frame_idx += 1
        self.stats.frame_count += 1
        self.stats.detect_ms = detect_ms
        self.stats.effects_ms = effects_ms
        self.stats.total_ms = total_ms

        inst_fps = 1000.0 / max(total_ms, 0.001)
        self._fps_ema = inst_fps if self._fps_ema == 0 else 0.9 * self._fps_ema + 0.1 * inst_fps
        self.stats.fps = self._fps_ema

        return out

    @staticmethod
    def _draw_landmarks(frame: np.ndarray, faces: list[FaceResult]) -> np.ndarray:
        out = frame.copy()
        for face in faces:
            for x, y in face.geometry.landmarks_px.astype(int):
                cv2.circle(out, (x, y), 1, (0, 255, 0), -1)
        return out

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        text = (
            f"FPS: {self.stats.fps:.1f}  "
            f"det: {self.stats.detect_ms:.1f}ms  "
            f"fx: {self.stats.effects_ms:.1f}ms  "
            f"scale: {self.detect_scale}"
        )
        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        return out


def build_preset(name: str, detect_scale: float = 0.5) -> FilterPipeline:
    """Factory for common filter presets."""
    presets = {
        "beauty": [
            BeautyEffect(smooth_strength=0.65, whiten_strength=0.3, lipstick_strength=0.6),
        ],
        "glasses": [StickerEffect("glasses")],
        "hat": [StickerEffect("hat")],
        "cat": [StickerEffect("cat_ears")],
        "expression": [ExpressionEffect()],
        "full": [
            BeautyEffect(),
            StickerEffect("glasses"),
            ExpressionEffect(),
        ],
    }
    effects = presets.get(name, presets["full"])
    return FilterPipeline(detect_scale=detect_scale, effects=effects)
