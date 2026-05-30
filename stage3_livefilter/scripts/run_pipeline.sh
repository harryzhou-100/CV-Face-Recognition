#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Stage 3 Live Filter Pipeline ==="

pip install -r requirements.txt -q

echo "[1/5] Generate sticker assets..."
python scripts/generate_stickers.py

echo "[2/5] Ensure face landmarker model..."
MODEL="$ROOT/models/face_landmarker.task"
if [[ ! -f "$MODEL" ]]; then
  mkdir -p "$ROOT/models"
  curl -sL -o "$MODEL" \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
fi

echo "[3/5] Generate demo video..."
python scripts/run_demo_video.py
python scripts/export_samples.py

echo "[4/5] Run benchmark..."
python scripts/benchmark.py --iterations 80

echo "[5/5] Generate report..."
python scripts/generate_report.py

echo "Done. See reports/experiment_report.md and outputs/demo/demo_video.mp4"
