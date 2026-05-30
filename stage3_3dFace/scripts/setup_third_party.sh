#!/usr/bin/env bash
# Build 3DDFA_V2 native extensions (FaceBoxes NMS, Sim3DR rasterizer)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TDDFA="${ROOT}/third_party/3DDFA_V2"

if [[ ! -d "${TDDFA}" ]]; then
  echo "Cloning 3DDFA_V2..."
  git clone --depth 1 https://github.com/cleardusk/3DDFA_V2.git "${TDDFA}"
fi

cd "${TDDFA}"
pip install -q cython onnx onnxruntime pyyaml
sh build.sh
echo "3DDFA_V2 extensions built."
