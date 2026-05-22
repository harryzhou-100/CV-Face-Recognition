"""Load IResNet50 embedding backbone from stage2_recognition checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECOGNITION_MODELS = PROJECT_ROOT / "stage2_recognition" / "models"
sys.path.insert(0, str(RECOGNITION_MODELS))

from iresnet import iresnet50  # noqa: E402

STAGE2_OPT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = PROJECT_ROOT / "stage2_recognition" / "work_dirs" / "resnet50_arcface_frozen" / "best.pth"
FALLBACK_CKPT = PROJECT_ROOT / "stage2_recognition" / "models" / "resnet50_arcface_best.pth"


class EmbeddingBackbone(nn.Module):
    """ResNet50 (IResNet50) face embedding extractor for inference / export."""

    def __init__(self, embedding_size: int = 512) -> None:
        super().__init__()
        self.backbone = iresnet50(num_features=embedding_size, dropout=0.0, fp16=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return F.normalize(feat, dim=1)


def resolve_checkpoint(path: Path | None) -> Path:
    if path and path.exists():
        return path
    if DEFAULT_CKPT.exists():
        return DEFAULT_CKPT
    if FALLBACK_CKPT.exists():
        return FALLBACK_CKPT
    raise FileNotFoundError("No ResNet50-ArcFace checkpoint found under stage2_recognition")


def load_fp32_backbone(
    checkpoint: Path | None = None,
    device: torch.device | None = None,
) -> EmbeddingBackbone:
    ckpt_path = resolve_checkpoint(checkpoint)
    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    prefix = "backbone."
    backbone_state = {
        k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)
    }
    if not backbone_state:
        raise KeyError(f"No backbone.* keys in checkpoint: {ckpt_path}")

    model = EmbeddingBackbone()
    model.backbone.load_state_dict(backbone_state, strict=True)
    model.eval()
    if device is not None:
        model.to(device)
    return model


def load_quantized_backbone(
    quant_path: Path,
    checkpoint: Path | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """Load dynamic-INT8 model saved by quantize.py."""
    del checkpoint  # weights are embedded in quant bundle
    payload = torch.load(quant_path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
    else:
        fp32 = load_fp32_backbone(None, device=torch.device("cpu"))
        model = torch.quantization.quantize_dynamic(fp32, {nn.Linear}, dtype=torch.qint8)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state)
    model.eval()
    if device is not None:
        model.to(device)
    return model
