"""
Quantifying Kindness — real-time CV inference pipeline.

Usage:
    python pipeline/main.py
    python pipeline/main.py --camera 0 --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import List

import cv2

# Allow running from the repo root or from inside pipeline/
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from database import InteractionsDB
from detector import KindnessDetector
from display import render_overlay
from scorer import score_for_state, state_label
from tracker import FinalizedSession, SessionTracker


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantifying Kindness — real-time kindness inference pipeline"
    )
    p.add_argument("--camera",  type=int,   default=0,    help="Camera device index (default 0)")
    p.add_argument("--width",   type=int,   default=1280, help="Capture width")
    p.add_argument("--height",  type=int,   default=720,  help="Capture height")
    p.add_argument("--db",      type=str,   default=str(_HERE / "data" / "interactions.db"),
                   help="Path to SQLite database file")
    p.add_argument("--no-landmarks", action="store_true",
                   help="Skip drawing face/pose landmarks (faster)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    args = _parse_args()
    db   = InteractionsDB(Path(args.db))
    det  = KindnessDetector()
    recent_scores: List[float] = db.recent_scores(5)

    # --- Session finalisation callback ---
    def on_session_end(session: FinalizedSession):
        row_id = db.save_session(session)
        recent_scores.clear()
        recent_scores.extend(db.recent_scores(5))
        dur = session.duration_seconds
        print(
            f"[session #{row_id}] "
            f"score={session.final_score:.1f}  "
            f"peak={session.peak_expression}  "
            f"duration={dur:.1f}s"
        )

    tracker = SessionTracker(on_session_end=on_session_end)

    # --- Graceful shutdown ---
    _shutdown = [False]

    def _sigint_handler(*_):
        _shutdown[0] = True

    signal.signal(signal.SIGINT,  _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    # --- Open camera ---
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Quantifying Kindness pipeline started. Press Ctrl+C or Q to quit.")

    # FPS tracking
    fps_history: List[float] = []
    t_prev = time.monotonic()

    window_name = "Quantifying Kindness"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while not _shutdown[0]:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: dropped frame", file=sys.stderr)
                continue

            # --- Detection ---
            det_result = det.process(frame)

            # --- Session tracking ---
            tracker.update(
                face_detected=det_result.face_detected,
                expression=det_result.expression,
                gesture=det_result.gesture,
            )

            # --- Draw landmarks ---
            if not args.no_landmarks:
                det.draw_landmarks(frame, det_result)

            # --- FPS ---
            now = time.monotonic()
            dt = now - t_prev
            t_prev = now
            fps_history.append(1.0 / dt if dt > 0 else 0.0)
            if len(fps_history) > 30:
                fps_history.pop(0)
            fps = sum(fps_history) / len(fps_history)

            # --- Overlay ---
            frame = render_overlay(
                frame,
                score=tracker.current_score,
                session_high=tracker.session_high,
                state=tracker.current_state,
                is_active=tracker.is_active,
                recent_scores=recent_scores,
                fps=fps,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # q or Esc
                break

    finally:
        print("\nShutting down…")
        # Finalise any open session
        # Force a "no-face" reading long enough to trigger end immediately
        for _ in range(1):
            tracker.update(face_detected=False, expression="neutral", gesture=None,
                           now=time.monotonic() + 10.0)

        cap.release()
        cv2.destroyAllWindows()
        det.close()
        db.close()
        print("Done.")


if __name__ == "__main__":
    main()
