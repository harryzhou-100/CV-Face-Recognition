#!/usr/bin/env bash
# StarGAN CelebA full pipeline
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1

echo "=== [1/5] Install dependencies ==="
pip install -q -r stage3_stargan/requirements.txt

echo "=== [2/5] Prepare CelebA ==="
python3 stage3_stargan/scripts/prepare_celeba.py

echo "=== [3/5] Train StarGAN ==="
python3 stage3_stargan/scripts/train.py

echo "=== [4/5] Generate attribute edits ==="
python3 stage3_stargan/scripts/generate_edits.py

echo "=== [5/5] Quality metrics (FID, IS) ==="
python3 stage3_stargan/scripts/evaluate_quality.py
python3 stage3_stargan/scripts/plot_training.py
python3 stage3_stargan/scripts/write_experiment_report.py

echo "Done. See stage3_stargan/reports/ and stage3_stargan/outputs/"
