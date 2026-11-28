"""
OpenCV overlay rendering for the live display window.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

# Colour palette (BGR)
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,   0)
_GREEN  = (80,  220, 80)
_YELLOW = (40,  220, 220)
_RED    = (60,  60,  220)
_BLUE   = (220, 160, 40)
_GREY   = (160, 160, 160)
_TEAL   = (200, 180, 60)

_SCORE_COLOUR = {
    range(1,  3): _RED,
    range(3,  5): (40, 120, 220),
    range(5,  7): _GREY,
    range(7,  9): _YELLOW,
    range(9, 11): _GREEN,
}


def _score_color(score: float):
    s = int(score)
    for rng, col in _SCORE_COLOUR.items():
        if s in rng:
            return col
    return _WHITE


def _draw_text_with_shadow(
    img,
    text: str,
    pos,
    font_scale: float,
    color,
    thickness: int = 2,
    font=cv2.FONT_HERSHEY_DUPLEX,
):
    x, y = pos
    cv2.putText(img, text, (x + 2, y + 2), font, font_scale, _BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),          font, font_scale, color,  thickness,     cv2.LINE_AA)


def _semi_rect(img, x1, y1, x2, y2, alpha=0.45):
    """Draw a semi-transparent dark rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def render_overlay(
    frame: np.ndarray,
    *,
    score: float,
    session_high: float,
    state: str,
    is_active: bool,
    recent_scores: List[float],
    fps: float,
) -> np.ndarray:
    """
    Draw the HUD on top of the camera frame (in-place).
    Returns the modified frame.
    """
    h, w = frame.shape[:2]

    # ------------------------------------------------------------------ #
    # Top-left panel: live session info
    # ------------------------------------------------------------------ #
    panel_w = 320
    panel_h = 155
    _semi_rect(frame, 10, 10, 10 + panel_w, 10 + panel_h)

    if is_active:
        status_text = "SESSION ACTIVE"
        status_col  = _GREEN
    else:
        status_text = "WAITING FOR PERSON..."
        status_col  = _GREY

    _draw_text_with_shadow(frame, status_text, (20, 40), 0.55, status_col, 1)

    state_display = state.upper().replace("_", " ").replace("+", " + ")
    _draw_text_with_shadow(frame, f"State: {state_display}", (20, 70), 0.55, _WHITE, 1)

    # Score (large)
    score_col = _score_color(score)
    _draw_text_with_shadow(
        frame, f"Score: {score:.1f}", (20, 110), 0.9, score_col, 2
    )
    _draw_text_with_shadow(
        frame, f"Session High: {session_high:.1f}", (20, 148), 0.55, _YELLOW, 1
    )

    # ------------------------------------------------------------------ #
    # Bottom-left panel: recent finalised scores
    # ------------------------------------------------------------------ #
    if recent_scores:
        log_x = 10
        log_y_start = h - 20 - len(recent_scores) * 24 - 15
        log_w = 240
        log_h = len(recent_scores) * 24 + 30
        _semi_rect(frame, log_x, log_y_start - 5, log_x + log_w, log_y_start + log_h)

        _draw_text_with_shadow(
            frame, "Recent sessions:", (log_x + 8, log_y_start + 15), 0.45, _GREY, 1
        )
        for i, s in enumerate(recent_scores):
            col = _score_color(s)
            _draw_text_with_shadow(
                frame,
                f"  #{len(recent_scores)-i:2d}  {s:.1f}",
                (log_x + 8, log_y_start + 15 + (i + 1) * 24),
                0.5, col, 1,
            )

    # ------------------------------------------------------------------ #
    # Top-right: FPS
    # ------------------------------------------------------------------ #
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
    _draw_text_with_shadow(frame, fps_text, (w - tw - 15, 30), 0.45, _GREY, 1)

    # ------------------------------------------------------------------ #
    # Centre-bottom: big score badge when session is active
    # ------------------------------------------------------------------ #
    if is_active:
        badge_text = f"{score:.0f}"
        font_scale = 3.5
        thickness  = 5
        (tw, th), baseline = cv2.getTextSize(
            badge_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
        )
        bx = (w - tw) // 2
        by = h - 60
        _semi_rect(frame, bx - 20, by - th - 20, bx + tw + 20, by + baseline + 10, alpha=0.5)
        _draw_text_with_shadow(
            frame, badge_text, (bx, by), font_scale, _score_color(score), thickness
        )

    return frame
