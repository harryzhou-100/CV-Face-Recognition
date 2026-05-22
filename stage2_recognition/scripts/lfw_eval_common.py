"""Shared LFW pair construction, metrics, and report writing."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

STAGE2_ROOT = Path(__file__).resolve().parents[1]
NAME_PATTERN = re.compile(r"^(.+)_(\d+)$")


def parse_person(stem: str) -> str:
    m = NAME_PATTERN.match(stem)
    return m.group(1) if m else stem


def list_lfw_images(root: Path) -> list[Path]:
    if list(root.glob("*.jpg")):
        return sorted(root.glob("*.jpg"))
    return sorted(root.rglob("*.jpg"))


def group_by_person(lfw_root: Path, images: list[Path]) -> dict[str, list[Path]]:
    by_person: dict[str, list[Path]] = defaultdict(list)
    for p in images:
        if lfw_root.name == "lfw_flat" or "_" in p.stem:
            by_person[parse_person(p.stem)].append(p)
        else:
            by_person[p.parent.name].append(p)
    return by_person


def lfw_image_path(
    lfw_root: Path,
    person: str,
    index: int,
) -> Path:
    """Map official LFW 1-based index to file path (flat or nested layout)."""
    stem = f"{person}_{person}_{index:04d}"
    flat = lfw_root / f"{stem}.jpg"
    if flat.exists():
        return flat
    nested = lfw_root / person / f"{person}_{index:04d}.jpg"
    if nested.exists():
        return nested
    return flat


def load_official_lfw_pairs(pairs_file: Path, lfw_root: Path) -> list[tuple[Path, Path, int]]:
    """Parse InsightFace/LFW official pairs.txt (6000 pairs, 3000 same + 3000 diff)."""
    lines = pairs_file.read_text(encoding="utf-8").strip().splitlines()
    pairs: list[tuple[Path, Path, int]] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) == 3:
            name, i, j = parts
            p1 = lfw_image_path(lfw_root, name, int(i))
            p2 = lfw_image_path(lfw_root, name, int(j))
            pairs.append((p1, p2, 1))
        elif len(parts) == 4:
            n1, i, n2, j = parts
            p1 = lfw_image_path(lfw_root, n1, int(i))
            p2 = lfw_image_path(lfw_root, n2, int(j))
            pairs.append((p1, p2, 0))
    return pairs


def build_pairs(by_person: dict, n_each: int, seed: int) -> list[tuple[Path, Path, int]]:
    rng = random.Random(seed)
    same, diff = [], []
    multi = [p for p, imgs in by_person.items() if len(imgs) >= 2]
    rng.shuffle(multi)
    for i in range(n_each):
        if multi:
            p = multi[i % len(multi)]
            a, b = rng.sample(by_person[p], 2)
            same.append((a, b, 1))
    persons = list(by_person.keys())
    while len(diff) < n_each and len(persons) >= 2:
        p1, p2 = rng.sample(persons, 2)
        diff.append((rng.choice(by_person[p1]), rng.choice(by_person[p2]), 0))
    pairs = same + diff
    rng.shuffle(pairs)
    return pairs


def pairwise_sq_dist(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """InsightFace verification metric: sum of squared differences (lower = same)."""
    diff = np.subtract(emb1, emb2)
    return np.sum(np.square(diff), axis=1)


def insightface_lfw_10fold(
    emb1: np.ndarray,
    emb2: np.ndarray,
    issame: np.ndarray,
    n_folds: int = 10,
) -> tuple[float, float, list[float]]:
    """
    Official InsightFace/MXNet LFW protocol (verification.py):
    L2-squared distance, thresholds in [0, 4), sklearn KFold shuffle=False.
    """
    from sklearn.model_selection import KFold

    issame = np.asarray(issame, dtype=bool)
    dist = pairwise_sq_dist(emb1, emb2)
    thresholds = np.arange(0, 4, 0.01)
    k_fold = KFold(n_splits=n_folds, shuffle=False)
    indices = np.arange(len(issame))
    fold_accs: list[float] = []

    for train_idx, test_idx in k_fold.split(indices):
        dist_train = dist[train_idx]
        issame_train = issame[train_idx]
        acc_train = np.zeros(len(thresholds))
        for i, thr in enumerate(thresholds):
            pred = np.less(dist_train, thr)
            tp = np.sum(np.logical_and(pred, issame_train))
            tn = np.sum(np.logical_and(np.logical_not(pred), np.logical_not(issame_train)))
            acc_train[i] = (tp + tn) / len(issame_train)
        best_thr = thresholds[np.argmax(acc_train)]
        dist_test = dist[test_idx]
        issame_test = issame[test_idx]
        pred = np.less(dist_test, best_thr)
        tp = np.sum(np.logical_and(pred, issame_test))
        tn = np.sum(np.logical_and(np.logical_not(pred), np.logical_not(issame_test)))
        fold_accs.append(float((tp + tn) / len(issame_test)))

    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


def lfw_10fold_accuracy(
    labels: np.ndarray,
    scores: np.ndarray,
    n_folds: int = 10,
    pairs_per_fold: int = 600,
) -> tuple[float, list[float]]:
    """Legacy cosine-based 10-fold (kept for reference). Prefer insightface_lfw_10fold."""
    assert len(labels) == n_folds * pairs_per_fold
    fold_accs: list[float] = []
    for fold in range(n_folds):
        test_start = fold * pairs_per_fold
        test_end = test_start + pairs_per_fold
        test_idx = slice(test_start, test_end)
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[test_start:test_end] = False
        thr, _ = best_threshold(labels[train_mask], scores[train_mask])
        y_test = labels[test_idx]
        s_test = scores[test_idx]
        pred = (s_test >= thr).astype(int)
        fold_accs.append(float(accuracy_score(y_test, pred)))
    return float(np.mean(fold_accs)), fold_accs


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_thr, best_acc = 0.5, 0.0
    for thr in np.linspace(-1.0, 1.0, 401):
        pred = (scores >= thr).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr, best_acc


def evaluate_pairs(
    pairs: list[tuple[Path, Path, int]],
    embed_fn,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, np.ndarray | None]]:
    cache: dict[str, np.ndarray | None] = {}
    labels, scores = [], []
    skipped = 0
    for p1, p2, y in pairs:
        k1, k2 = str(p1), str(p2)
        if k1 not in cache:
            cache[k1] = embed_fn(p1)
        if k2 not in cache:
            cache[k2] = embed_fn(p2)
        e1, e2 = cache[k1], cache[k2]
        if e1 is None or e2 is None:
            skipped += 1
            continue
        labels.append(y)
        scores.append(float(np.dot(e1, e2)))
    return np.array(labels), np.array(scores), skipped, cache


def compute_stats(
    y_true: np.ndarray,
    scores: np.ndarray,
    thr: float,
    *,
    lfw_dir: str,
    model_desc: str,
    n_images: int,
    n_persons: int,
    skipped: int,
    target: float = 0.985,
    accuracy_10fold: float | None = None,
    accuracy_10fold_std: float | None = None,
    fold_accs: list[float] | None = None,
    pair_source: str = "",
    metric: str = "cosine",
) -> dict:
    y_pred = (scores >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    acc_report = accuracy_10fold if accuracy_10fold is not None else acc
    return {
        "lfw_dir": lfw_dir,
        "model": model_desc,
        "n_images": n_images,
        "n_persons": n_persons,
        "n_pairs": len(y_true),
        "skipped": skipped,
        "threshold": thr,
        "accuracy": float(acc_report),
        "accuracy_unconstrained": float(acc),
        "accuracy_10fold": accuracy_10fold,
        "accuracy_10fold_std": accuracy_10fold_std,
        "fold_accuracies": fold_accs,
        "pair_source": pair_source,
        "metric": metric,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores))
        if len(np.unique(y_true)) > 1
        else 0.0,
        "target_met": bool(acc_report >= target),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def write_report(stats: dict, report_path: Path, *, backend: str, train_note: str) -> None:
    acc = stats["accuracy"]
    thr = stats["threshold"]
    cm = stats["confusion_matrix"]
    metric_note = ""
    if stats.get("accuracy_10fold") is not None:
        metric_note = "（LFW **10-fold** 交叉验证均值）"
    lines = [
        "# LFW 人脸识别验证报告（ResNet50 + ArcFace）",
        "",
        f"- 推理后端: **{backend}**",
        f"- 训练数据: {train_note}",
        f"- 模型: `{stats['model']}`",
        f"- LFW 目录: `{stats['lfw_dir']}`",
        f"- 配对协议: `{stats.get('pair_source', 'N/A')}`",
        f"- 预处理: InsightFace 官方 `(pixel - 127.5) / 127.5`，BGR→RGB（`blobFromImages`）",
        "",
        "## 验证结果",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| **准确率 (Accuracy)**{metric_note} | **{acc * 100:.2f}%** |",
        f"| 最优阈值 | {thr:.4f} |",
        f"| 精确率 | {stats['precision'] * 100:.2f}% |",
        f"| 召回率 | {stats['recall'] * 100:.2f}% |",
        f"| F1 | {stats['f1'] * 100:.2f}% |",
        f"| ROC-AUC | {stats['roc_auc']:.4f} |",
        "",
    ]
    if stats.get("accuracy_unconstrained") is not None and stats.get("accuracy_10fold") is not None:
        lines.append(
            f"| 无约束全局阈值准确率 | {stats['accuracy_unconstrained'] * 100:.2f}% |"
        )
        folds = stats.get("fold_accuracies") or []
        if folds:
            std = stats.get("accuracy_10fold_std")
            std_s = f" ± {std*100:.2f}%" if std is not None else ""
            lines.append(
                f"| 10-fold 各折准确率 | {', '.join(f'{x*100:.2f}%' for x in folds)}{std_s} |"
            )
    lines.extend([
        "",
        f"| 图像数 | {stats['n_images']} |",
        f"| 身份数 | {stats['n_persons']} |",
        f"| 有效配对数 | {stats['n_pairs']} |",
        f"| 跳过（读图失败） | {stats['skipped']} |",
        "",
        "### 混淆矩阵",
        "",
        "| | 预测: 不同人 | 预测: 同一人 |",
        "|---|-------------|-------------|",
        f"| 真实: 不同人 | {cm[0][0]} | {cm[0][1]} |",
        f"| 真实: 同一人 | {cm[1][0]} | {cm[1][1]} |",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    metrics_path = report_path.parent / "lfw_metrics.json"
    metrics_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
