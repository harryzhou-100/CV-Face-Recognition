# Quantifying Kindness — CV Pipeline

Real-time computer vision pipeline that infers a "kindness score" (1–10) from a webcam feed using MediaPipe face mesh + pose estimation. Built for the *Quantifying Kindness* performance art installation.

---

## Requirements

- Python 3.9+
- A webcam (or USB camera)

## Install

```bash
cd pipeline
pip install -r requirements.txt
```

> **Tip:** Use a virtual environment:
> ```bash
> python -m venv .venv && source .venv/bin/activate
> pip install -r requirements.txt
> ```

## Run

```bash
# From the repo root:
python pipeline/main.py

# Or from inside the pipeline/ directory:
python main.py
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--camera N` | `0` | Camera device index |
| `--width W` | `1280` | Capture width in pixels |
| `--height H` | `720` | Capture height in pixels |
| `--db PATH` | `pipeline/data/interactions.db` | SQLite database path |
| `--no-landmarks` | off | Skip drawing face/pose overlays (slightly faster) |

### Exit

Press **Q**, **Esc**, or **Ctrl+C**. Any open session is finalised and saved before exit.

---

## Architecture

```
pipeline/
  main.py       Entry point; camera loop, graceful shutdown
  detector.py   MediaPipe face mesh + pose wrapper → expression + gesture
  scorer.py     Score mapping table + running-max logic
  tracker.py    Person lifecycle state machine (session start/end)
  database.py   SQLite persistence
  display.py    OpenCV HUD overlay
  data/         Created at runtime; holds interactions.db
```

### Session lifecycle

1. **Warmup** — a face must be visible for ≥ 2 s before a session starts (avoids brief walk-bys).
2. **Active** — scores are sampled every frame; the displayed score is the running maximum (never drops).
3. **End** — no face detected for ≥ 3 s finalises the session, saves it to SQLite, and resets.

### Score table

| State | Score |
|---|---|
| Hug (arms-across-chest) | 10 |
| Happy / big smile | 9 |
| Surprised + smile | 8 |
| Pointing (arm extended) | 7 |
| Surprised | 6 |
| Neutral | 5 |
| Sad | 3 |
| Fearful | 3 |
| Angry | 2 |
| Disgusted | 2 |

### Database schema

```sql
CREATE TABLE interactions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,   -- ISO 8601 UTC
    duration_seconds REAL,
    final_score      REAL,
    peak_expression  TEXT,
    score_history    TEXT                -- JSON array of {t, score, state}
);
```

---

## Expression classification

Expressions are classified from MediaPipe Face Mesh 468-point landmarks using geometric heuristics:

- **Smile**: mouth-corner lift ratio relative to face width
- **Surprised**: brow-raise index + wide mouth opening + wide-open eyes
- **Angry/Sad**: brow furrow + corner direction
- **Fearful**: wide eyes with closed mouth
- **Disgusted**: open mouth without brow raise

No external model weights are needed; runs fully on-CPU.

## Gesture detection

- **Hug**: both wrists close together and crossed near the torso midline
- **Pointing**: one arm extended above shoulder height with elbow near-straight

## Performance

Targets ≥ 15 FPS on CPU. MediaPipe Pose is set to `model_complexity=0` (fastest). FPS is displayed in the top-right corner of the window. For higher throughput, pass `--no-landmarks` to skip landmark drawing.

## Known limitations

- Expression classification is heuristic (no trained classifier): edge cases like subtle expressions or extreme head angles may misfire.
- Gesture detection works best when the full torso is visible in frame; partial body views reduce accuracy.
- Single-person mode only (one face tracked at a time).
- Hug detection works best when the arms are visibly crossed; a loose hug facing the camera may not trigger.
