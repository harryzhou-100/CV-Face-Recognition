#!/usr/bin/env bash
# End-to-end: prepare data -> train -> evaluate -> visualize
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "=== [1/4] Prepare WIDER FACE (COCO format) ==="
python3 stage2_detection/scripts/prepare_wider_face.py

echo "=== [2/4] Train RetinaNet on WIDER FACE ==="
python3 stage2_detection/scripts/train.py

echo "=== [3/4] Evaluate on validation set ==="
python3 stage2_detection/scripts/evaluate.py

echo "=== [4/4] Visualize detections (categorized) ==="
python3 stage2_detection/scripts/visualize_results.py

echo "=== Done. See stage2_detection/reports/ and stage2_detection/outputs/ ==="
