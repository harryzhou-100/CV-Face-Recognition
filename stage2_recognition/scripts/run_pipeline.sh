#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p stage2_recognition/models

echo "=== [1/4] Prepare MS1MV3 subset + LFW ==="
python3 stage2_recognition/scripts/prepare_msceleb.py --max-shards 6 --max-images 80000
python3 stage2_recognition/scripts/prepare_lfw.py

echo "=== [2/4] Train ResNet50 + ArcFace ==="
python3 stage2_recognition/scripts/train.py

echo "=== [3/4] LFW verification ==="
python3 stage2_recognition/scripts/verify_lfw.py

echo "=== Done ==="
echo "Report: stage2_recognition/reports/lfw_verification_report.md"
echo "Curves: stage2_recognition/outputs/curves/loss_accuracy_curves.png"
