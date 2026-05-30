#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

echo "=== Stage3 3D Face: setup ==="
bash scripts/setup_third_party.sh

echo "=== Prepare sample images ==="
python3 scripts/prepare_samples.py

echo "=== Reconstruct & render ==="
python3 scripts/reconstruct.py --config configs/pipeline.json

echo "=== Analysis & report ==="
python3 scripts/analyze_results.py

echo "Done. See reports/experiment_report.md and outputs/"
