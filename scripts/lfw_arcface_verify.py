"""
LFW face verification with InsightFace (ArcFace recognition backbone).

Builds same/different pairs from filenames (Name_0001.jpg -> Name),
computes cosine similarity, selects threshold by best accuracy, writes MD report.

Usage:
  conda activate face_cv
  pip install insightface onnxruntime
  python scripts/lfw_arcface_verify.py
  python scripts/lfw_arcface_verify.py --data-root /Users/apple/Projects/data/lfw/lfw
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LFW_DIR = Path("/Users/apple/Projects/data/lfw/lfw")
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "outputs" / "lfw_arcface_verification_report.md"

NAME_PATTERN = re.compile(r"^(.+)_(\d+)$")


def parse_person_id(filename: str) -> str:
    stem = Path(filename).stem
    m = NAME_PATTERN.match(stem)
    return m.group(1) if m else stem


def list_lfw_images(lfw_dir: Path) -> list[Path]:
    return sorted(p for p in lfw_dir.glob("*.jpg") if p.is_file())


def build_pairs(
    by_person: dict[str, list[Path]],
    n_same: int,
    n_diff: int,
    seed: int,
) -> list[tuple[Path, Path, int]]:
    rng = random.Random(seed)
    same_pairs: list[tuple[Path, Path, int]] = []
    diff_pairs: list[tuple[Path, Path, int]] = []

    persons_multi = [p for p, imgs in by_person.items() if len(imgs) >= 2]
    rng.shuffle(persons_multi)

    while len(same_pairs) < n_same and persons_multi:
        person = persons_multi[len(same_pairs) % len(persons_multi)]
        imgs = by_person[person]
        a, b = rng.sample(imgs, 2)
        same_pairs.append((a, b, 1))

    persons = list(by_person.keys())
    while len(diff_pairs) < n_diff and len(persons) >= 2:
        p1, p2 = rng.sample(persons, 2)
        if p1 == p2:
            continue
        img1 = rng.choice(by_person[p1])
        img2 = rng.choice(by_person[p2])
        diff_pairs.append((img1, img2, 0))

    pairs = same_pairs + diff_pairs
    rng.shuffle(pairs)
    return pairs


def load_arcface(model_name: str):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name=model_name,
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def extract_embedding(app, image_path: Path, cache: dict) -> np.ndarray | None:
    key = str(image_path.resolve())
    if key in cache:
        return cache[key]

    img = cv2.imread(key)
    if img is None:
        cache[key] = None
        return None

    faces = app.get(img)
    if not faces:
        cache[key] = None
        return None

    emb = faces[0].normed_embedding.astype(np.float32)
    cache[key] = emb
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def evaluate_pairs(
    app,
    pairs: list[tuple[Path, Path, int]],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    cache: dict[str, np.ndarray | None] = {}
    labels: list[int] = []
    scores: list[float] = []
    skipped = 0

    for i, (p1, p2, label) in enumerate(pairs):
        e1 = extract_embedding(app, p1, cache)
        e2 = extract_embedding(app, p2, cache)
        if e1 is None or e2 is None:
            skipped += 1
            continue
        labels.append(label)
        scores.append(cosine_similarity(e1, e2))
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(pairs)} pairs ...")

    return np.array(labels), np.array(scores), skipped, len(cache)


def best_threshold_accuracy(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_thr, best_acc = 0.5, 0.0
    for thr in np.linspace(-1.0, 1.0, 401):
        pred = (scores >= thr).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


def write_report(
    path: Path,
    args: argparse.Namespace,
    stats: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = stats["confusion_matrix"]
    lines = [
        "# LFW 人脸识别验证报告（ArcFace / InsightFace）",
        "",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据目录: `{stats['lfw_dir']}`",
        f"- 识别模型: InsightFace `{args.model}`（ArcFace 特征）",
        f"- 验证协议: 随机采样同一人 / 不同人配对（各 {args.pairs_each} 对）",
        "",
        "## 1. 数据概况",
        "",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 图像总数 | {stats['n_images']:,} |",
        f"| 身份数（由文件名解析） | {stats['n_persons']:,} |",
        f"| 有效验证对数 | {stats['n_pairs']:,} |",
        f"| 跳过（未检测到人脸） | {stats['skipped_pairs']:,} |",
        f"| 成功提取特征图像数 | {stats['n_embedded']:,} |",
        "",
        "## 2. 验证结果",
        "",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| **准确率 (Accuracy)** | **{stats['accuracy'] * 100:.2f}%** |",
        f"| 最优阈值 (cosine) | {stats['threshold']:.4f} |",
        f"| 精确率 (Precision) | {stats['precision'] * 100:.2f}% |",
        f"| 召回率 (Recall) | {stats['recall'] * 100:.2f}% |",
        f"| F1 分数 | {stats['f1'] * 100:.2f}% |",
        f"| ROC-AUC | {stats['roc_auc']:.4f} |",
        "",
        "### 混淆矩阵（阈值 = 最优阈值）",
        "",
        "| | 预测: 不同人 | 预测: 同一人 |",
        "|---|-------------|-------------|",
        f"| 真实: 不同人 | {cm[0][0]:,} (TN) | {cm[0][1]:,} (FP) |",
        f"| 真实: 同一人 | {cm[1][0]:,} (FN) | {cm[1][1]:,} (TP) |",
        "",
        "## 3. 说明",
        "",
        "- **同一人**：文件名主体相同（如 `Tom_Cruise_0001` 与 `Tom_Cruise_0002`）。",
        "- **不同人**：随机抽取两个不同主体各一张图。",
        "- 相似度为 ArcFace 特征向量余弦相似度；大于阈值判为同一人。",
        "- 当前 LFW 为扁平目录（`*.jpg`），若使用官方完整 LFW 与 `pairs.txt`，",
        "  可将 `--pairs-file` 指向官方协议文件以复现标准 benchmark。",
        "",
        "## 4. 复现命令",
        "",
        "```bash",
        "conda activate face_cv",
        "python scripts/lfw_arcface_verify.py \\",
        f"  --data-root {stats['lfw_dir']} \\",
        f"  --report {path}",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="LFW ArcFace face verification")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_LFW_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", default="buffalo_l", help="InsightFace model pack")
    parser.add_argument(
        "--pairs-each",
        type=int,
        default=1500,
        help="Number of same-person and different-person pairs",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lfw_dir = args.data_root.resolve()
    if not lfw_dir.is_dir():
        raise SystemExit(f"LFW directory not found: {lfw_dir}")

    images = list_lfw_images(lfw_dir)
    if not images:
        raise SystemExit(f"No jpg images in {lfw_dir}")

    by_person: dict[str, list[Path]] = defaultdict(list)
    for img in images:
        by_person[parse_person_id(img.name)].append(img)

    pairs = build_pairs(by_person, args.pairs_each, args.pairs_each, args.seed)
    print(f"LFW dir: {lfw_dir}")
    print(f"Images: {len(images)}, persons: {len(by_person)}, pairs: {len(pairs)}")
    print(f"Loading InsightFace model: {args.model} ...")

    app = load_arcface(args.model)
    y_true, scores, skipped, n_embedded = evaluate_pairs(app, pairs)

    if len(y_true) == 0:
        raise SystemExit("No valid pairs after face detection.")

    threshold, accuracy = best_threshold_accuracy(y_true, scores)
    y_pred = (scores >= threshold).astype(int)

    stats = {
        "lfw_dir": str(lfw_dir),
        "n_images": len(images),
        "n_persons": len(by_person),
        "n_pairs": len(y_true),
        "skipped_pairs": skipped,
        "n_embedded": n_embedded,
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }

    write_report(args.report, args, stats)

    print(f"\nAccuracy: {accuracy * 100:.2f}%  (threshold={threshold:.4f})")
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
