#!/usr/bin/env python3
"""Train HRNet-W18 face landmark model on 300W (MMPose)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE2_ROOT = Path(__file__).resolve().parents[1]
CONFIG = STAGE2_ROOT / "configs" / "hrnetv2_w18_300w_256x256.py"
WORK_DIR = STAGE2_ROOT / "work_dirs" / "hrnetv2_w18_300w"
DATA_ROOT = STAGE2_ROOT / "data" / "300w"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmpose.utils import register_all_modules

    register_all_modules(init_default_scope=True)

    data_root = DATA_ROOT.as_posix() + "/"
    cfg = Config.fromfile(str(args.config))
    cfg.data_root = data_root
    for key in ("train_dataloader", "val_dataloader", "test_dataloader"):
        cfg[key].dataset.data_root = data_root
    cfg.work_dir = str(args.work_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.epochs is not None:
        cfg.train_cfg.max_epochs = args.epochs
        cfg.param_scheduler[1]["end"] = args.epochs
    if args.resume:
        cfg.resume = True

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
