#!/usr/bin/env python3
"""Evaluate generated images: FID (vs CelebA val) and Inception Score."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage3_stargan"))

from src.data_loader import get_loader
from src.networks import Generator
from src.solver import label2onehot


class ImageFolderPaths(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)


def get_inception(device):
    model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.aux_logits = False
    model.eval().to(device)
    return model


def to_inception_input(x):
    """[-1,1] or [0,1] tensors -> ImageNet-normalized 299x299."""
    if x.min() < 0:
        x = (x + 1) / 2
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def extract_features(model, loader, device, max_batches=None):
    feats = []
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc="Inception features")):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = to_inception_input(batch.to(device))
            f = model(batch)
            feats.append(f.cpu().numpy())
            if max_batches and bi + 1 >= max_batches:
                break
    return np.concatenate(feats, axis=0)


def compute_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def inception_score(probs, splits=10):
    probs = np.array(probs)
    n = probs.shape[0]
    scores = []
    for i in range(splits):
        part = probs[i * (n // splits) : (i + 1) * (n // splits)]
        py = np.mean(part, axis=0)
        kl = part * (np.log(part + 1e-8) - np.log(py + 1e-8))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return float(np.mean(scores)), float(np.std(scores))


@torch.no_grad()
def generate_fake_dataset(G, loader, device, c_dim, out_dir, max_images=2000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    to_pil = transforms.ToPILImage()
    for x, c in tqdm(loader, desc="Generate fakes"):
        x = x.to(device)
        c = c.to(device)
        c_trg = label2onehot(torch.rand(c.size(0), c_dim, device=device), c_dim)
        for j in range(c.size(0)):
            c_trg[j] = c[j].clone()
            flip = torch.randint(0, c_dim, (1,)).item()
            c_trg[j, flip] = 1 - c[j, flip]
        fake = (G(x, c_trg) + 1) / 2
        for j in range(fake.size(0)):
            if count >= max_images:
                return count
            img = to_pil(fake[j].cpu().clamp(0, 1))
            img.save(out_dir / f"{count:06d}.png")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_stargan/configs/train_celeba_stargan.json")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = ROOT / cfg["checkpoint_dir"]
    if args.checkpoint:
        ckpt = ckpt_dir / f"{args.checkpoint}-G.pth"
    else:
        ckpt = sorted(ckpt_dir.glob("*-G.pth"), key=lambda p: int(p.stem.split("-")[0]))[-1]

    G = Generator(cfg["g_conv_dim"], cfg["c_dim"], cfg["g_repeat_num"]).to(device)
    G.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    G.eval()

    inc_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_loader = get_loader(
        str(ROOT / cfg["data_root"]),
        str(ROOT / cfg["data_root"] / "list_attr_celeba.txt"),
        cfg["selected_attrs"],
        cfg["image_size"],
        args.batch_size,
        "val",
        2,
    )

    fake_dir = ROOT / "stage3_stargan/outputs/fid_fakes"
    n_fake = generate_fake_dataset(G, val_loader, device, cfg["c_dim"], fake_dir, args.max_images)

    model = get_inception(device)
    # Real features from val set
    real_feats = extract_features(model, val_loader, device, max_batches=args.max_images // args.batch_size + 1)
    real_feats = real_feats[: args.max_images]

    fake_paths = sorted(fake_dir.glob("*.png"))[: args.max_images]
    fake_loader = DataLoader(
        ImageFolderPaths(fake_paths, inc_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    fake_feats = extract_features(model, fake_loader, device)

    mu_r, sigma_r = np.mean(real_feats, 0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = np.mean(fake_feats, 0), np.cov(fake_feats, rowvar=False)
    fid = compute_fid(mu_r, sigma_r, mu_f, sigma_f)

    # IS on fakes — use full Inception with classifier head
    inc_cls = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    inc_cls.eval().to(device)
    all_probs = []
    with torch.no_grad():
        for batch in fake_loader:
            batch = batch.to(device)  # already ImageNet-normalized
            out = inc_cls(batch)
            if isinstance(out, tuple):
                out = out[0]
            probs = F.softmax(out, dim=1).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)
    is_mean, is_std = inception_score(all_probs)

    metrics = {
        "fid": fid,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "num_real": int(len(real_feats)),
        "num_fake": int(len(fake_feats)),
        "checkpoint": str(ckpt),
    }
    report_dir = ROOT / "stage3_stargan/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "quality_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    md = f"""# StarGAN 生成图像质量评估

## 指标

| 指标 | 数值 | 说明 |
|------|------|------|
| FID ↓ | **{fid:.2f}** | 相对 CelebA 验证集真实分布（Inception 特征，越低越好） |
| IS ↑ | **{is_mean:.2f} ± {is_std:.2f}** | 生成图 Inception Score（越高越好） |

- 评估样本数：真实 {metrics['num_real']} / 生成 {metrics['num_fake']}
- 权重：`{ckpt.name}`

## 参考

- 原始 StarGAN 论文在 CelebA 上 FID 约 15–25（视训练迭代与实现而定）
- IS 对人脸域通常低于 ImageNet 自然图像（论文 IS 约 2–3）

## 生成假样本目录

`stage3_stargan/outputs/fid_fakes/`
"""
    (report_dir / "quality_evaluation_report.md").write_text(md, encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Report: {report_dir / 'quality_evaluation_report.md'}")


if __name__ == "__main__":
    main()
