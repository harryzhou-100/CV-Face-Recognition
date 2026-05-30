#!/usr/bin/env python3
"""Generate face attribute editing results with trained StarGAN."""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage3_stargan"))

from src.data_loader import get_loader
from src.networks import Generator
from src.solver import denorm, label2onehot


ATTR_CN = {
    "Black_Hair": "黑发",
    "Blond_Hair": "金发",
    "Brown_Hair": "棕发",
    "Male": "性别(男)",
    "Young": "年轻",
}


@torch.no_grad()
def edit_single_attr(G, x, c_org, attr_idx, c_dim, device):
    c_trg = c_org.clone()
    c_trg[:, attr_idx] = 1 - c_org[:, attr_idx]
    return G(x, c_trg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_stargan/configs/train_celeba_stargan.json")
    parser.add_argument("--checkpoint", type=int, default=None, help="Iteration tag, e.g. 80000")
    parser.add_argument("--num_samples", type=int, default=16)
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = ROOT / cfg["checkpoint_dir"]
    best_g = ckpt_dir / "G-best.pth"
    if args.checkpoint is None and best_g.exists():
        ckpt_path = best_g
        iters = "best"
    elif args.checkpoint is None:
        ckpts = sorted(ckpt_dir.glob("*-G.pth"), key=lambda p: int(p.stem.split("-")[0]))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
        ckpt_path = ckpts[-1]
        iters = int(ckpt_path.stem.split("-")[0])
    else:
        iters = args.checkpoint
        ckpt_path = ckpt_dir / f"{iters}-G.pth"

    G = Generator(cfg["g_conv_dim"], cfg["c_dim"], cfg["g_repeat_num"]).to(device)
    G.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    G.eval()

    data_root = ROOT / cfg["data_root"]
    loader = get_loader(
        str(data_root),
        str(data_root / "list_attr_celeba.txt"),
        cfg["selected_attrs"],
        cfg["image_size"],
        batch_size=args.num_samples,
        mode="val",
        num_workers=2,
    )
    x, c = next(iter(loader))
    x = x.to(device)
    c = c.to(device)

    out_root = ROOT / cfg["result_dir"]
    out_root.mkdir(parents=True, exist_ok=True)
    attrs = cfg["selected_attrs"]

    # Per-attribute folders
    for i, name in enumerate(attrs):
        sub = out_root / name
        sub.mkdir(parents=True, exist_ok=True)
        x_fake = edit_single_attr(G, x, c, i, cfg["c_dim"], device)
        grid = torch.cat([x, x_fake], dim=3)
        save_image(denorm(grid), sub / f"edit_{name}_iter{iters}.jpg", nrow=4)
        for j in range(min(8, x.size(0))):
            pair = torch.cat([x[j : j + 1], x_fake[j : j + 1]], dim=3)
            save_image(denorm(pair), sub / f"sample_{j:02d}.jpg")

    # Combined comparison grid: original + 5 edits
    rows = []
    for j in range(min(8, x.size(0))):
        row = [x[j : j + 1]]
        for i in range(cfg["c_dim"]):
            row.append(edit_single_attr(G, x[j : j + 1], c[j : j + 1], i, cfg["c_dim"], device))
        rows.append(torch.cat(row, dim=3))
    full = torch.cat(rows, dim=2)
    save_image(denorm(full), out_root / f"all_attrs_grid_iter{iters}.jpg")

    # Attribute matrix visualization (one identity, toggle each attr)
    idx = 0
    x0, c0 = x[idx : idx + 1], c[idx : idx + 1]
    panels = [x0]
    labels = ["原图"]
    for i, name in enumerate(attrs):
        panels.append(edit_single_attr(G, x0, c0, i, cfg["c_dim"], device))
        state = "开启" if c0[0, i].item() < 0.5 else "关闭"
        labels.append(f"{ATTR_CN.get(name, name)}→{state}")
    strip = torch.cat(panels, dim=3)
    save_image(denorm(strip), out_root / "single_identity_attr_strip.jpg")

    meta = {
        "checkpoint": str(ckpt_path),
        "iterations": iters,
        "attributes": attrs,
        "num_samples": args.num_samples,
        "output_dir": str(out_root),
    }
    with open(out_root / "generation_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved edits to {out_root}")


if __name__ == "__main__":
    main()
