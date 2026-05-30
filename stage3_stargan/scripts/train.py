#!/usr/bin/env python3
"""Train StarGAN on CelebA."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage3_stargan"))

from src.data_loader import get_loader
from src.solver import Solver


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="stage3_stargan/configs/train_celeba_stargan.json",
    )
    parser.add_argument("--resume", type=int, default=0, help="Resume from iteration tag")
    parser.add_argument("--auto-resume", action="store_true", default=True)
    parser.add_argument("--resume-g-only", action="store_true", help="Only load G weights (e.g. after D arch fix)")
    parser.add_argument("--no-auto-resume", dest="auto_resume", action="store_false")
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    cfg = load_config(cfg_path)
    ckpt_dir = ROOT / cfg["checkpoint_dir"]
    if args.resume_g_only:
        cfg["resume_g_only"] = True
    if args.resume:
        cfg["resume_iters"] = args.resume
    elif args.auto_resume:
        latest = ckpt_dir / "latest.txt"
        if latest.exists():
            cfg["resume_iters"] = int(latest.read_text().strip())
            print(f"Auto-resume from iter {cfg['resume_iters']}")
        else:
            ckpts = sorted(ckpt_dir.glob("*-G.pth"), key=lambda p: int(p.stem.split("-")[0]))
            if ckpts:
                cfg["resume_iters"] = int(ckpts[-1].stem.split("-")[0])
                print(f"Auto-resume from iter {cfg['resume_iters']}")

    set_seed(cfg.get("seed", 42))
    data_root = ROOT / cfg["data_root"]
    attr_path = data_root / "list_attr_celeba.txt"

    loader = get_loader(
        root=str(data_root),
        attr_path=str(attr_path),
        selected_attrs=cfg["selected_attrs"],
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        mode="train",
        num_workers=cfg["num_workers"],
    )
    steps = len(loader)
    max_steps = cfg.get("max_steps_per_epoch") or steps
    epochs = cfg.get("num_epochs", 3)
    print(
        f"Train loader: {len(loader.dataset)} images, batch={cfg['batch_size']}, "
        f"{epochs} epoch(s) × {max_steps} steps/epoch",
        flush=True,
    )

    for k in ("checkpoint_dir", "sample_dir"):
        cfg[k] = str(ROOT / cfg[k])

    solver = Solver(cfg)
    history = solver.train(loader)

    report_dir = ROOT / cfg.get("report_dir", "stage3_stargan/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    hist_path = report_dir / "train_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training done. History saved to {hist_path}")


if __name__ == "__main__":
    main()
