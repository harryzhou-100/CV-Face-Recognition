"""
Person lifecycle / session management.

Session start  : face detected for ≥ SESSION_START_SECS consecutive seconds
Session end    : no face detected for ≥ SESSION_END_SECS consecutive seconds
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from scorer import RunningMaxScorer, state_label, score_for_state

SESSION_START_SECS = 2.0   # face must be present this long to begin a session
SESSION_END_SECS   = 3.0   # face must be absent this long to end a session


@dataclass
class ScoreReading:
    t: float          # epoch seconds
    score: float
    state: str


@dataclass
class FinalizedSession:
    start_time: float
    end_time: float
    duration_seconds: float
    final_score: float
    peak_expression: str
    score_history: List[ScoreReading]


class SessionTracker:
    """
    State machine that tracks a single person's interaction session.

    Call `update(face_detected, expression, gesture)` every frame.
    When a session finalises, `on_session_end` is called with FinalizedSession.
    """

    # Internal states
    _IDLE     = "idle"       # no face, waiting
    _WARMING  = "warming"    # face seen, counting towards SESSION_START_SECS
    _ACTIVE   = "active"     # session in progress
    _COOLING  = "cooling"    # face gone, counting towards SESSION_END_SECS

    def __init__(self, on_session_end: Callable[[FinalizedSession], None]):
        self._on_session_end = on_session_end
        self._state = self._IDLE

        self._face_start: Optional[float] = None   # when face first appeared
        self._no_face_start: Optional[float] = None

        self._session_start: Optional[float] = None
        self._scorer = RunningMaxScorer()
        self._history: List[ScoreReading] = []

        # Public read-outs for the display layer
        self.current_score: float = 5.0
        self.session_high: float = 5.0
        self.current_state: str = "neutral"
        self.is_active: bool = False

    # ------------------------------------------------------------------
    def update(
        self,
        face_detected: bool,
        expression: str,
        gesture: Optional[str],
        now: Optional[float] = None,
    ) -> None:
        if now is None:
            now = time.monotonic()

        if self._state == self._IDLE:
            self._handle_idle(face_detected, expression, gesture, now)
        elif self._state == self._WARMING:
            self._handle_warming(face_detected, expression, gesture, now)
        elif self._state == self._ACTIVE:
            self._handle_active(face_detected, expression, gesture, now)
        elif self._state == self._COOLING:
            self._handle_cooling(face_detected, expression, gesture, now)

    # ------------------------------------------------------------------
    def _handle_idle(self, face, expr, gesture, now):
        if face:
            self._face_start = now
            self._state = self._WARMING
        # Remains idle

    def _handle_warming(self, face, expr, gesture, now):
        if not face:
            self._face_start = None
            self._state = self._IDLE
            return
        elapsed = now - self._face_start
        if elapsed >= SESSION_START_SECS:
            self._start_session(now, expr, gesture)

    def _handle_active(self, face, expr, gesture, now):
        if face:
            self._no_face_start = None
            self._record_reading(expr, gesture, now)
        else:
            if self._no_face_start is None:
                self._no_face_start = now
            elif now - self._no_face_start >= SESSION_END_SECS:
                self._end_session(now)
                return
            # Still cooling down inside active state → switch to cooling
            if self._no_face_start is not None:
                self._state = self._COOLING

    def _handle_cooling(self, face, expr, gesture, now):
        if face:
            # Person returned; continue the session
            self._no_face_start = None
            self._state = self._ACTIVE
            self._record_reading(expr, gesture, now)
        else:
            elapsed = now - self._no_face_start
            if elapsed >= SESSION_END_SECS:
                self._end_session(now)

    # ------------------------------------------------------------------
    def _start_session(self, now: float, expr: str, gesture):
        self._session_start = now
        self._scorer.reset()
        self._history = []
        self._state = self._ACTIVE
        self.is_active = True
        self._no_face_start = None
        self._record_reading(expr, gesture, now)

    def _record_reading(self, expr: str, gesture, now: float):
        label = state_label(expr, gesture)
        score = self._scorer.update(expr, gesture)
        reading = ScoreReading(t=now, score=score, state=label)
        self._history.append(reading)
        self.current_score = score
        self.session_high = self._scorer.score
        self.current_state = label

    def _end_session(self, now: float):
        duration = now - (self._session_start or now)
        session = FinalizedSession(
            start_time=self._session_start or now,
            end_time=now,
            duration_seconds=duration,
            final_score=self._scorer.score,
            peak_expression=self._scorer.peak_state,
            score_history=list(self._history),
        )
        self._on_session_end(session)

        # Reset state machine
        self._state = self._IDLE
        self._face_start = None
        self._no_face_start = None
        self._session_start = None
        self._scorer.reset()
        self._history = []
        self.current_score = 5.0
        self.session_high = 5.0
        self.current_state = "neutral"
        self.is_active = False
