"""
SQLite persistence layer for completed sessions.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from tracker import FinalizedSession, ScoreReading

DB_PATH = Path(__file__).parent / "data" / "interactions.db"

_DDL = """
CREATE TABLE IF NOT EXISTS interactions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    duration_seconds REAL,
    final_score      REAL,
    peak_expression  TEXT,
    score_history    TEXT
);
"""


def _reading_to_dict(r: ScoreReading) -> dict:
    return {"t": round(r.t, 3), "score": r.score, "state": r.state}


class InteractionsDB:
    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(_DDL)
        self._conn.commit()

    def save_session(self, session: FinalizedSession) -> int:
        timestamp = datetime.fromtimestamp(
            session.end_time, tz=timezone.utc
        ).isoformat()
        history_json = json.dumps([_reading_to_dict(r) for r in session.score_history])
        cur = self._conn.execute(
            """
            INSERT INTO interactions
                (timestamp, duration_seconds, final_score, peak_expression, score_history)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                round(session.duration_seconds, 2),
                round(session.final_score, 2),
                session.peak_expression,
                history_json,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def recent_scores(self, n: int = 5) -> List[float]:
        """Return the n most-recent final_scores, oldest first."""
        cur = self._conn.execute(
            "SELECT final_score FROM interactions ORDER BY id DESC LIMIT ?", (n,)
        )
        rows = [r[0] for r in cur.fetchall()]
        return list(reversed(rows))

    def close(self):
        self._conn.close()
