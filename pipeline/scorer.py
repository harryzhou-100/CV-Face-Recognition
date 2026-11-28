"""
Kindness score mapping and running-max logic.

Score table (1–10):
  hug            → 10
  happy          → 9   (big smile)
  surprised+smile → 8  (handled by tracker combining expression + gesture)
  pointing       → 7
  surprised      → 6
  neutral        → 5
  sad            → 3
  fearful        → 3
  angry          → 2
  disgusted      → 2
"""

from __future__ import annotations

from typing import Optional

# Base scores per detected state
_BASE_SCORES: dict[str, float] = {
    "hug":              10.0,
    "happy":             9.0,
    "surprised+smile":   8.0,
    "pointing":          7.0,
    "surprised":         6.0,
    "neutral":           5.0,
    "sad":               3.0,
    "fearful":           3.0,
    "angry":             2.0,
    "disgusted":         2.0,
}

DEFAULT_SCORE = 5.0


def state_label(expression: str, gesture: Optional[str]) -> str:
    """Combine expression + gesture into a single state label."""
    if gesture == "hug":
        return "hug"
    if gesture == "pointing":
        return "pointing"
    if expression == "happy" and gesture is not None:
        # Any positive gesture + smile → surprised+smile level
        return "surprised+smile"
    if expression == "surprised" and gesture is None:
        # Could distinguish surprised+smile if we had it
        return "surprised"
    return expression


def score_for_state(state: str) -> float:
    """Return the kindness score for a composite state label."""
    return _BASE_SCORES.get(state, DEFAULT_SCORE)


class RunningMaxScorer:
    """
    Tracks the running-maximum kindness score for a single session.

    The score only ever goes up during a session; it never drops.
    """

    def __init__(self):
        self._current_max: float = DEFAULT_SCORE
        self._peak_state: str = "neutral"

    @property
    def score(self) -> float:
        return self._current_max

    @property
    def peak_state(self) -> str:
        return self._peak_state

    def update(self, expression: str, gesture: Optional[str]) -> float:
        """
        Feed a new reading; returns the (possibly unchanged) running max.
        """
        label = state_label(expression, gesture)
        s = score_for_state(label)
        if s > self._current_max:
            self._current_max = s
            self._peak_state = label
        return self._current_max

    def reset(self):
        self._current_max = DEFAULT_SCORE
        self._peak_state = "neutral"
