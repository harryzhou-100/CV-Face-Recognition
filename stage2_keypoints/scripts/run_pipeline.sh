#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "=== [1/4] Prepare 300W ==="
python3 stage2_keypoints/scripts/prepare_300w.py

echo "=== [2/4] Train HRNet-W18 (300W, 68 pts) ==="
python3 stage2_keypoints/scripts/train.py --epochs 20

echo "=== [3/4] Evaluate NME (test) ==="
CKPT="stage2_keypoints/work_dirs/hrnetv2_w18_300w/best_NME_epoch_20.pth"
python3 stage2_keypoints/scripts/evaluate.py --split test --checkpoint "$CKPT"
cp "$CKPT" stage2_keypoints/models/hrnetv2_w18_300w_best.pth

echo "=== [4/4] Face alignment & visualization ==="
python3 stage2_keypoints/scripts/align_faces.py --checkpoint "$CKPT"

echo "Done. See stage2_keypoints/reports/ and stage2_keypoints/outputs/"
