#!/usr/bin/env python3
"""Dynamic INT8 quantization for ResNet50 (IResNet50) face embedding backbone."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from model_utils import STAGE2_OPT, load_fp32_backbone, resolve_checkpoint

DEFAULT_OUT = STAGE2_OPT / "models" / "resnet50_int8_dynamic.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch dynamic INT8 quantize ResNet50 backbone")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="FP32 checkpoint (default: frozen best.pth or resnet50_arcface_best.pth)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    ckpt = resolve_checkpoint(args.checkpoint)
    model = load_fp32_backbone(ckpt)
    print(f"[quantize] Source checkpoint: {ckpt}")

    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    quantized.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "torch_dynamic_int8",
        "source_checkpoint": str(ckpt.resolve()),
        "quantized_layers": "Linear",
        "dtype": "qint8",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save({"meta": payload, "model": quantized}, args.output)
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"[quantize] Saved: {args.output} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
