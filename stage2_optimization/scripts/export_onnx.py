#!/usr/bin/env python3
"""Export ResNet50-ArcFace embedding backbone to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

STAGE2_OPT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGE2_OPT / "scripts"))

from model_utils import load_fp32_backbone, resolve_checkpoint  # noqa: E402

DEFAULT_ONNX = STAGE2_OPT / "models" / "resnet50_arcface_embedding.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export IResNet50 embedding model to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export dynamic batch dimension",
    )
    args = parser.parse_args()

    ckpt = resolve_checkpoint(args.checkpoint)
    model = load_fp32_backbone(ckpt, device=torch.device("cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 112, 112)
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "embedding": {0: "batch"}}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes=dynamic_axes,
    )
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"[export] Source: {ckpt}")
    print(f"[export] ONNX saved: {args.output} ({size_mb:.2f} MB, opset={args.opset})")

    try:
        import onnx

        onnx_model = onnx.load(str(args.output))
        onnx.checker.check_model(onnx_model)
        print("[export] onnx.checker: OK")
    except ImportError:
        print("[export] onnx package not installed; skipped checker")


if __name__ == "__main__":
    main()
