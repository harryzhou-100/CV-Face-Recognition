"""ResNet50-ArcFace (InsightFace IResNet50 backbone) + ArcFace margin head."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from iresnet import iresnet50

STAGE2_MODELS = Path(__file__).resolve().parent
DEFAULT_PRETRAINED = STAGE2_MODELS / "ms1mv3_arcface_r50_backbone.pth"
ALT_PRETRAINED = STAGE2_MODELS / "resnet50_arcface_pretrained.pth"


class ArcFace(nn.Module):
    """Additive Angular Margin Loss (ArcFace)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.5,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.s


def load_iresnet_pretrained(backbone: nn.Module, ckpt_path: Path) -> tuple[int, int]:
    """Load ArcFace-Torch backbone weights; returns (loaded, skipped)."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    elif isinstance(raw, dict):
        state = raw
    else:
        state = raw

    cleaned = {}
    for k, v in state.items():
        key = k.replace("module.", "")
        if key.startswith("arcface.") or "weight.softmax" in key:
            continue
        cleaned[key] = v.float() if v.dtype == torch.float16 else v

    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    loaded = len(cleaned) - len(unexpected)
    return loaded, len(missing)


class ResNet50ArcFace(nn.Module):
    """
    Standard ResNet50-ArcFace (IResNet50 + 512-d embedding).
    Backbone initialized from MS1MV3 ArcFace pretrained weights when available.
    ArcFace classification head is re-initialized for the local subset class count.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 512,
        s: float = 64.0,
        m: float = 0.5,
        pretrained: bool = True,
        pretrained_path: str | Path | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = iresnet50(num_features=embedding_size, dropout=0.0, fp16=False)
        self.arcface = ArcFace(embedding_size, num_classes, s=s, m=m)
        self.embedding_size = embedding_size
        self.pretrained_loaded = False
        self.backbone_frozen = False

        if pretrained:
            path = Path(pretrained_path) if pretrained_path else DEFAULT_PRETRAINED
            if not path.exists():
                path = ALT_PRETRAINED
            if path.exists():
                n_loaded, n_miss = load_iresnet_pretrained(self.backbone, path)
                self.pretrained_loaded = True
                print(
                    f"[pretrained] Loaded IResNet50-ArcFace from {path.name} "
                    f"(tensors={n_loaded}, missing_keys={n_miss})"
                )
            else:
                print(f"[warn] ArcFace pretrained not found at {path}; training from scratch.")

        if freeze_backbone:
            self.freeze_feature_extractor()

    def freeze_feature_extractor(self) -> int:
        """Freeze all IResNet50 feature-extraction parameters (requires_grad=False)."""
        n_frozen = 0
        for param in self.backbone.parameters():
            param.requires_grad = False
            n_frozen += param.numel()
        self.backbone_frozen = True
        print(f"[freeze] IResNet50 backbone frozen ({n_frozen:,} params, requires_grad=False)")
        return n_frozen

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x))

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.forward_features(x)
        if labels is None:
            return emb
        return self.arcface(emb, labels)
