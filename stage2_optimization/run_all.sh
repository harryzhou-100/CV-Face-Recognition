#!/usr/bin/env bash
# ResNet50 optimization pipeline: quantize → compare → export ONNX
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== 1/3 Dynamic INT8 quantization ==="
python3 stage2_optimization/scripts/quantize.py

echo "=== 2/3 FP32 vs INT8 comparison ==="
python3 -u stage2_optimization/scripts/compare_quantization.py "$@"

echo "=== 3/3 Export ONNX ==="
python3 stage2_optimization/scripts/export_onnx.py

echo "Done. Report: stage2_optimization/reports/quantization_comparison.md"
