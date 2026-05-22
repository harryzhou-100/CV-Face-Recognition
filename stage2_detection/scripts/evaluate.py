#!/usr/bin/env python3
"""Evaluate face detector on WIDER FACE val set (mAP, precision, recall)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE2_ROOT = Path(__file__).resolve().parents[1]
CONFIG = STAGE2_ROOT / "configs" / "retinanet_r50_fpn_wider_face.py"
REPORT_DIR = STAGE2_ROOT / "reports"
DATA_ROOT = STAGE2_ROOT / "data" / "wider_face"


def find_checkpoint(work_dir: Path) -> Path:
    candidates = sorted(work_dir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    best = work_dir / "best_coco_bbox_mAP_epoch_*.pth"
    for p in sorted(work_dir.glob("best_*.pth")):
        return p
    last = work_dir / "last_checkpoint"
    if last.exists():
        name = last.read_text().strip()
        ckpt = work_dir / name
        if ckpt.exists():
            return ckpt
    raise FileNotFoundError(f"No checkpoint in {work_dir}")


def compute_pr_at_threshold(
    results: list,
    dataset_items: list,
    iou_thr: float = 0.5,
    score_thr: float = 0.3,
) -> dict[str, float]:
    """Image-level precision/recall at fixed score and IoU thresholds."""
    tp = fp = fn = 0
    for idx in range(len(dataset_items)):
        data = dataset_items[idx]
        gt = data["data_samples"].gt_instances
        gt_boxes = gt.bboxes.cpu().numpy() if len(gt.bboxes) else np.zeros((0, 4))
        pred = results[idx].pred_instances
        keep = pred.scores.cpu().numpy() >= score_thr
        pred_boxes = pred.bboxes.cpu().numpy()[keep] if keep.any() else np.zeros((0, 4))

        matched_gt = set()
        for pb in pred_boxes:
            if len(gt_boxes) == 0:
                fp += 1
                continue
            ious = box_iou(pb, gt_boxes)
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and j not in matched_gt:
                tp += 1
                matched_gt.add(j)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "score_thr": score_thr,
        "iou_thr": iou_thr,
    }


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-8
    return inter / union


def wider_subset_metrics(
    results: list, dataset_items: list, ann_file: Path, split_name: str
) -> dict:
    """Approximate Easy/Medium/Hard using GT face height (WIDER convention)."""
    with ann_file.open() as f:
        coco = json.load(f)
    id_to_h = {}
    for img in coco["images"]:
        id_to_h[img["id"]] = img["height"]
    img_anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    subsets = {"easy": [], "medium": [], "hard": []}
    for res_idx, data in enumerate(dataset_items):
        img_id = int(data["data_samples"].img_id)
        h_img = id_to_h.get(img_id, 0)
        for ann in img_anns.get(img_id, []):
            _, _, _, bh = ann["bbox"]
            if h_img <= 0:
                continue
            scale = bh / h_img
            if scale >= 0.08:
                subsets["easy"].append((res_idx, ann))
            elif scale >= 0.03:
                subsets["medium"].append((res_idx, ann))
            else:
                subsets["hard"].append((res_idx, ann))

    out = {}
    for name, items in subsets.items():
        if not items:
            out[name] = {"count": 0}
            continue
        tp = fp = fn = 0
        by_img: dict[int, list] = {}
        for idx, ann in items:
            by_img.setdefault(idx, []).append(ann)
        for idx, anns in by_img.items():
            data = dataset_items[idx]
            gt_boxes = np.array(
                [
                    [
                        a["bbox"][0],
                        a["bbox"][1],
                        a["bbox"][0] + a["bbox"][2],
                        a["bbox"][1] + a["bbox"][3],
                    ]
                    for a in anns
                ]
            )
            pred = results[idx].pred_instances
            keep = pred.scores.cpu().numpy() >= 0.3
            pred_boxes = pred.bboxes.cpu().numpy()[keep] if keep.any() else np.zeros((0, 4))
            matched = set()
            for pb in pred_boxes:
                if len(gt_boxes) == 0:
                    fp += 1
                    continue
                ious = box_iou(pb, gt_boxes)
                j = int(np.argmax(ious))
                if ious[j] >= 0.5 and j not in matched:
                    tp += 1
                    matched.add(j)
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        out[name] = {
            "count": len(items),
            "precision": float(prec),
            "recall": float(rec),
        }
    out["split"] = split_name
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to .pth; default: latest in work_dir",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=STAGE2_ROOT / "work_dirs" / "retinanet_r50_wider_face",
    )
    parser.add_argument(
        "--compute-pr",
        action="store_true",
        help="额外抽样推理计算 Precision/Recall（较慢）",
    )
    args = parser.parse_args()

    import os

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)

    data_root = (DATA_ROOT).as_posix() + "/"
    cfg = Config.fromfile(str(args.config))
    cfg.data_root = data_root
    for key in ("train_dataloader", "val_dataloader", "test_dataloader"):
        if hasattr(cfg, key):
            cfg[key].dataset.data_root = data_root
    cfg.val_evaluator.ann_file = data_root + "annotations/instances_val.json"
    cfg.test_evaluator.ann_file = data_root + "annotations/instances_val.json"

    default_ckpt = args.work_dir / "epoch_6.pth"
    ckpt = args.checkpoint or (
        default_ckpt if default_ckpt.exists() else find_checkpoint(args.work_dir)
    )
    cfg.load_from = str(ckpt)
    cfg.work_dir = str(args.work_dir / "eval")

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    val_loader = runner.val_dataloader
    dataset = val_loader.dataset
    pr = None
    subsets = {}
    if args.compute_pr:
        from mmdet.apis import init_detector, inference_detector

        model = init_detector(str(args.config), str(ckpt), device="cuda:0")
        results = []
        sample_n = min(400, len(dataset))
        indices = list(range(0, len(dataset), max(1, len(dataset) // sample_n)))[
            :sample_n
        ]
        for i in indices:
            data = dataset[i]
            img_path = data["data_samples"].img_path
            results.append(inference_detector(model, img_path))
        pr = compute_pr_at_threshold(
            results, [dataset[i] for i in indices], score_thr=0.3
        )
        ann_val = DATA_ROOT / "annotations" / "instances_val.json"
        subsets = wider_subset_metrics(
            results, [dataset[i] for i in indices], ann_val, "val"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(ckpt),
        "coco_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in (metrics or {}).items()},
        "precision_recall_at_0.3": pr,
        "wider_subsets_approx": subsets,
    }
    # Flatten coco bbox mAP keys from metrics dict
    if metrics:
        for k, v in metrics.items():
            if "mAP" in k or "bbox" in k:
                report.setdefault("summary", {})[k] = float(v) if isinstance(v, (int, float, np.floating)) else v

    out_json = REPORT_DIR / "evaluation_metrics.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md = REPORT_DIR / "evaluation_report.md"
    mAP = None
    if metrics:
        mAP = metrics.get("coco/bbox_mAP", metrics.get("bbox_mAP"))
        if mAP is None:
            for k, v in metrics.items():
                if "mAP" in str(k):
                    mAP = v
                    break

    lines = [
        "# WIDER FACE 人脸检测评估报告",
        "",
        f"- 模型: RetinaNet R50-FPN",
        f"- 权重: `{ckpt}`",
        f"- 验证集: WIDER FACE val ({len(dataset)} 张图)",
        "",
        "## 核心指标",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| mAP (COCO bbox, IoU 0.5:0.95) | {mAP if mAP is not None else '见 coco_metrics'} |",
        f"| mAP@0.5 (COCO) | {metrics.get('coco/bbox_mAP_50', 'N/A') if metrics else 'N/A'} |",
    ]
    if pr:
        lines.extend(
            [
                f"| Precision (@score≥0.3, IoU≥0.5, 抽样) | {pr['precision']:.4f} |",
                f"| Recall (@score≥0.3, IoU≥0.5, 抽样) | {pr['recall']:.4f} |",
                f"| F1 | {pr['f1']:.4f} |",
            ]
        )
    else:
        lines.append(
            "| Precision / Recall (固定阈值) | 见 COCO mAP / AR（全量验证集） |"
        )
    if subsets:
        lines.extend(
            [
                "## WIDER 难度子集（按 GT 人脸尺度近似，抽样）",
                "",
                "| 子集 | GT 框数 | Precision | Recall |",
                "|------|---------|-----------|--------|",
            ]
        )
        for name in ("easy", "medium", "hard"):
            s = subsets.get(name, {})
            lines.append(
                f"| {name} | {s.get('count', 0)} | "
                f"{s.get('precision', 0):.4f} | {s.get('recall', 0):.4f} |"
            )
    lines.extend(
        [
            "",
            "## 按尺度 mAP（COCO 标准）",
            "",
            f"| 尺度 | mAP |",
            f"|------|-----|",
            f"| small | {metrics.get('coco/bbox_mAP_s', 'N/A') if metrics else 'N/A'} |",
            f"| medium | {metrics.get('coco/bbox_mAP_m', 'N/A') if metrics else 'N/A'} |",
            f"| large | {metrics.get('coco/bbox_mAP_l', 'N/A') if metrics else 'N/A'} |",
        ]
    )
    lines.extend(["", "## COCO 详细指标", "", "```json", json.dumps(metrics or {}, indent=2), "```"])
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {md}")
    print(f"Metrics JSON: {out_json}")


if __name__ == "__main__":
    main()
