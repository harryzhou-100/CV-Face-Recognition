#!/usr/bin/env python3
"""Evaluate face landmark model on 300W test set (NME)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE2_ROOT = Path(__file__).resolve().parents[1]
CONFIG = STAGE2_ROOT / "configs" / "hrnetv2_w18_300w_256x256.py"
REPORT_DIR = STAGE2_ROOT / "reports"
DATA_ROOT = STAGE2_ROOT / "data" / "300w"
WORK_DIR = STAGE2_ROOT / "work_dirs" / "hrnetv2_w18_300w"


def find_checkpoint(work_dir: Path) -> Path:
    for pattern in ("best_NME_*.pth", "epoch_*.pth"):
        files = sorted(work_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        if files:
            if pattern.startswith("best"):
                return files[0]
    last = work_dir / "last_checkpoint"
    if last.exists():
        name = last.read_text().strip()
        p = work_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No checkpoint in {work_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR)
    parser.add_argument("--split", choices=("test", "valid"), default="test")
    args = parser.parse_args()

    import os

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmpose.utils import register_all_modules

    register_all_modules(init_default_scope=True)

    data_root = DATA_ROOT.as_posix() + "/"
    cfg = Config.fromfile(str(args.config))
    ann = (
        "annotations/face_landmarks_300w_test.json"
        if args.split == "test"
        else "annotations/face_landmarks_300w_valid.json"
    )
    cfg.test_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.ann_file = ann
    cfg.test_evaluator = dict(type="NME", norm_mode="keypoint_distance")

    ckpt = args.checkpoint or find_checkpoint(args.work_dir)
    cfg.load_from = str(ckpt)
    cfg.work_dir = str(args.work_dir / f"eval_{args.split}")

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    nme = None
    if metrics:
        nme = metrics.get("NME", metrics.get("coco/NME"))
        if nme is None:
            for k, v in metrics.items():
                if "NME" in str(k).upper():
                    nme = v
                    break

    report = {
        "checkpoint": str(ckpt),
        "split": args.split,
        "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in (metrics or {}).items()},
        "NME": float(nme) if nme is not None else None,
    }
    out_json = REPORT_DIR / f"nme_{args.split}.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = REPORT_DIR / "evaluation_report.md"
    lines = [
        "# 300W 人脸关键点评估报告",
        "",
        f"- 模型: HRNet-W18 (68 点)",
        f"- 权重: `{ckpt}`",
        f"- 评估集: 300W **{args.split}**",
        "",
        "## 指标",
        "",
        "| 指标 | 数值 | 说明 |",
        "|------|------|------|",
        f"| **NME** | **{nme:.4f}** | 归一化平均误差（眼间距归一化，越小越好） |" if nme is not None else "| NME | - | - |",
        "",
        "## 详细",
        "",
        "```json",
        json.dumps(metrics or {}, indent=2),
        "```",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"NME ({args.split}): {nme}")
    print(f"Report: {md}")


if __name__ == "__main__":
    main()
