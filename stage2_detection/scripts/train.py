#!/usr/bin/env python3
"""Train RetinaNet face detector on WIDER FACE with MMDetection."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE2_ROOT = Path(__file__).resolve().parents[1]
CONFIG = STAGE2_ROOT / "configs" / "retinanet_r50_fpn_wider_face.py"
WORK_DIR = STAGE2_ROOT / "work_dirs" / "retinanet_r50_wider_face"


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
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(str(args.config))
    data_root = (STAGE2_ROOT / "data" / "wider_face").as_posix() + "/"
    cfg.data_root = data_root
    for key in ("train_dataloader", "val_dataloader", "test_dataloader"):
        if hasattr(cfg, key):
            cfg[key].dataset.data_root = data_root
    for key in ("val_evaluator", "test_evaluator"):
        if hasattr(cfg, key):
            cfg[key].ann_file = data_root + "annotations/instances_val.json"
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
