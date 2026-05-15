"""Explore CelebA and LFW: statistics, label distributions, visualization, report."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
HAIR_ATTRS = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]
GENDER_ATTR = "Male"


@dataclass
class DatasetStats:
    name: str
    image_count: int = 0
    identity_count: int | None = None
    extra: dict = field(default_factory=dict)
    label_stats: dict[str, dict[str, int | float]] = field(default_factory=dict)


def find_celeba_paths(data_root: Path) -> tuple[Path | None, Path | None]:
    img_candidates = [
        data_root / "celeba" / "img_align_celeba",
        data_root / "celeba" / "_zip" / "Dataset" / "CelebA_train" / "img_align_celeba",
        data_root / "celeba" / "Img" / "img_align_celeba",
    ]
    attr_candidates = [
        data_root / "celeba" / "list_attr_celeba.txt",
        data_root / "celeba" / "Anno" / "list_attr_celeba.txt",
    ]
    img_dir = next((p for p in img_candidates if p.is_dir()), None)
    attr_file = next((p for p in attr_candidates if p.is_file()), None)
    return img_dir, attr_file


def find_lfw_paths(data_root: Path) -> Path | None:
    candidates = [
        data_root / "lfw" / "lfw_hf_extract" / "lfw_multifaces-ingestion",
        data_root / "lfw" / "lfw",
        data_root / "lfw",
    ]
    best: Path | None = None
    best_count = 0
    for path in candidates:
        if not path.is_dir():
            continue
        count = len(list_images(path))
        if count > best_count:
            best, best_count = path, count
    return best if best_count > 0 else None


def lfw_identity_from_path(path: Path) -> str:
    """Aaron_Eckhart_0001.jpg -> Aaron_Eckhart (flat LFW layout)."""
    stem = path.stem
    if "_" in stem:
        prefix, suffix = stem.rsplit("_", 1)
        if suffix.isdigit():
            return prefix
    return path.parent.name


def list_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_celeba_attributes(attr_path: Path) -> dict[str, dict[str, int]]:
    lines = attr_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return {}
    attr_names = lines[1].split()
    records: dict[str, dict[str, int]] = {}
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < len(attr_names) + 1:
            continue
        fname = parts[0]
        values = [int(x) for x in parts[1 : 1 + len(attr_names)]]
        records[fname] = dict(zip(attr_names, values))
    return records


def celeba_label_distributions(
    image_paths: list[Path], attr_records: dict[str, dict[str, int]]
) -> dict[str, dict[str, int | float]]:
    matched = 0
    male = female = unknown_gender = 0
    hair_counts = Counter()
    no_hair_label = 0

    for path in image_paths:
        attrs = attr_records.get(path.name)
        if attrs is None:
            continue
        matched += 1

        g = attrs.get(GENDER_ATTR)
        if g == 1:
            male += 1
        elif g == -1:
            female += 1
        else:
            unknown_gender += 1

        hair_hit = False
        for name in HAIR_ATTRS:
            if attrs.get(name) == 1:
                hair_counts[name] += 1
                hair_hit = True
        if not hair_hit:
            no_hair_label += 1

    total = max(matched, 1)
    gender = {
        "male": male,
        "female": female,
        "unknown": unknown_gender,
        "male_pct": round(100 * male / total, 2),
        "female_pct": round(100 * female / total, 2),
    }
    hair = {k: hair_counts.get(k, 0) for k in HAIR_ATTRS}
    hair["none_labeled"] = no_hair_label
    for k in HAIR_ATTRS:
        hair[f"{k}_pct"] = round(100 * hair[k] / total, 2)
    return {"gender": gender, "hair_color": hair, "matched_attributes": matched}


def lfw_identity_stats(image_paths: list[Path]) -> dict[str, int | float]:
    identities = {lfw_identity_from_path(p) for p in image_paths}
    per_person = Counter(lfw_identity_from_path(p) for p in image_paths)
    counts = list(per_person.values()) if per_person else [0]
    return {
        "identities": len(identities),
        "min_images_per_person": min(counts),
        "max_images_per_person": max(counts),
        "avg_images_per_person": round(sum(counts) / len(counts), 2),
    }


def analyze_celeba(data_root: Path) -> tuple[DatasetStats | None, list[Path]]:
    img_dir, attr_path = find_celeba_paths(data_root)
    if img_dir is None:
        return None, []

    images = list_images(img_dir)
    stats = DatasetStats(name="CelebA", image_count=len(images))

    if attr_path:
        attrs = parse_celeba_attributes(attr_path)
        stats.label_stats = celeba_label_distributions(images, attrs)
        stats.extra["attribute_file"] = str(attr_path)
    else:
        stats.extra["warning"] = "list_attr_celeba.txt not found; skip label stats"

    return stats, images


def analyze_lfw(data_root: Path) -> tuple[DatasetStats | None, list[Path]]:
    lfw_dir = find_lfw_paths(data_root)
    if lfw_dir is None:
        return None, []

    images = list_images(lfw_dir)
    stats = DatasetStats(
        name="LFW",
        image_count=len(images),
        identity_count=lfw_identity_stats(images)["identities"],
    )
    stats.extra = lfw_identity_stats(images)
    stats.extra["note"] = (
        "LFW 官方包不含发色/性别标签；标签分布仅对 CelebA 统计。"
    )
    return stats, images


def plot_sample_grid(
    image_paths: list[Path],
    title: str,
    out_path: Path,
    grid_size: int = 4,
    seed: int = 42,
) -> None:
    if not image_paths:
        return

    rng = random.Random(seed)
    n_show = min(grid_size * grid_size, len(image_paths))
    picks = rng.sample(image_paths, n_show)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle(title, fontsize=14)
    axes_flat = axes.flatten()

    for ax, img_path in zip(axes_flat, picks):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            ax.axis("off")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(img_path.name[:18], fontsize=8)
        ax.axis("off")

    for ax in axes_flat[n_show:]:
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_label_bars(stats: DatasetStats, out_path: Path) -> None:
    if not stats.label_stats:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{stats.name} label distribution")

    gender = stats.label_stats.get("gender", {})
    g_labels = ["male", "female"]
    g_vals = [gender.get("male", 0), gender.get("female", 0)]
    axes[0].bar(g_labels, g_vals, color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Gender (CelebA Male attr)")
    axes[0].set_ylabel("count")

    hair = stats.label_stats.get("hair_color", {})
    h_labels = HAIR_ATTRS
    h_vals = [hair.get(k, 0) for k in h_labels]
    axes[1].bar(h_labels, h_vals, color="#55A868")
    axes[1].set_title("Hair color (positive labels)")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def write_report(
    report_dir: Path,
    data_root: Path,
    celeba: DatasetStats | None,
    lfw: DatasetStats | None,
    figures: list[str],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = report_dir / "analysis_report.md"

    lines = [
        "# 数据集分析报告",
        "",
        f"- 生成时间: {ts}",
        f"- 数据根目录: `{data_root.resolve()}`",
        "",
        "## 1. 数据概览",
        "",
    ]

    for ds in (celeba, lfw):
        if ds is None:
            continue
        lines.append(f"### {ds.name}")
        lines.append("")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 图片数量 | {ds.image_count:,} |")
        if ds.identity_count is not None:
            lines.append(f"| 身份/人数 | {ds.identity_count:,} |")
        for k, v in ds.extra.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    if celeba and celeba.label_stats:
        lines.extend(["## 2. CelebA 标签分布", ""])
        gender = celeba.label_stats["gender"]
        lines.append("### 性别比例")
        lines.append("")
        lines.append(
            f"- 男性 (Male=1): {gender['male']:,} ({gender['male_pct']}%)"
        )
        lines.append(
            f"- 女性 (Male=-1): {gender['female']:,} ({gender['female_pct']}%)"
        )
        lines.append("")

        hair = celeba.label_stats["hair_color"]
        lines.append("### 发色标签（多标签，可重叠）")
        lines.append("")
        for attr in HAIR_ATTRS:
            lines.append(
                f"- {attr}: {hair[attr]:,} ({hair[f'{attr}_pct']}%)"
            )
        lines.append(
            f"- 无发色正标签: {hair['none_labeled']:,}"
        )
        lines.append(
            f"- 有属性标注的图片数: {celeba.label_stats['matched_attributes']:,}"
        )
        lines.append("")

    if lfw:
        lines.extend(
            [
                "## 3. LFW 说明",
                "",
                "LFW 按人物文件夹组织。官方发布包不包含发色/性别属性文件，",
                "本报告对 LFW 仅统计图片数与每人图片数分布。",
                "",
            ]
        )

    lines.extend(["## 4. 可视化图表", ""])
    for fig in figures:
        lines.append(f"![{Path(fig).name}]({Path(fig).name})")
        lines.append("")

    lines.extend(
        [
            "## 5. 目录结构建议",
            "",
            "```text",
            "../data/",
            "  celeba/",
            "    img_align_celeba/",
            "    list_attr_celeba.txt",
            "  lfw/",
            "    lfw/<person_name>/*.jpg",
            "```",
            "",
            "项目内 `samples/` 存放 10–20 张抽样图，完整数据通过 Docker volume 挂载 `../data`。",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def resolve_data_root(args: argparse.Namespace, project_root: Path) -> Path:
    if args.data_root:
        return args.data_root.expanduser().resolve()
    return (project_root.parent / "data").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="CelebA & LFW data exploration")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Host dataset directory (default: ../data beside project)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("数据集分析报告"),
        help="Output report folder",
    )
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="Use ./samples only (quick test without full dataset)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=4)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_root = (
        (project_root / "samples").resolve()
        if args.use_samples
        else resolve_data_root(args, project_root)
    )

    if args.use_samples:
        celeba_root = data_root / "celeba"
        lfw_root = data_root / "lfw"
        celeba_stats = (
            DatasetStats("CelebA (samples)", len(list_images(celeba_root)))
            if celeba_root.is_dir()
            else None
        )
        lfw_stats = (
            DatasetStats("LFW (samples)", len(list_images(lfw_root)))
            if lfw_root.is_dir()
            else None
        )
        celeba_images = list_images(celeba_root) if celeba_root.is_dir() else []
        lfw_images = list_images(lfw_root) if lfw_root.is_dir() else []
        attr_path = find_celeba_paths(resolve_data_root(args, project_root))[1]
        if celeba_stats and attr_path and attr_path.is_file():
            attrs = parse_celeba_attributes(attr_path)
            celeba_stats.label_stats = celeba_label_distributions(
                celeba_images, attrs
            )
    else:
        celeba_stats, celeba_images = analyze_celeba(data_root)
        lfw_stats, lfw_images = analyze_lfw(data_root)

    if not celeba_images and not lfw_images:
        raise SystemExit(
            f"No images found under {data_root}. "
            "Download datasets (see docs/DATA_SETUP.md) or run prepare_dataset_samples.py."
        )

    report_dir = (project_root / args.report_dir).resolve()
    figures: list[str] = []

    if celeba_stats and celeba_images:
        fig = report_dir / "celeba_sample_grid.png"
        plot_sample_grid(
            celeba_images,
            "CelebA random samples",
            fig,
            args.grid_size,
            args.seed,
        )
        figures.append(fig.name)
        if celeba_stats.label_stats:
            bar = report_dir / "celeba_label_distribution.png"
            plot_label_bars(celeba_stats, bar)
            figures.append(bar.name)

    if lfw_stats and lfw_images:
        fig = report_dir / "lfw_sample_grid.png"
        plot_sample_grid(
            lfw_images,
            "LFW random samples",
            fig,
            args.grid_size,
            args.seed + 7,
        )
        figures.append(fig.name)

    report_path = write_report(
        report_dir, data_root, celeba_stats, lfw_stats, figures
    )

    print(f"Data root: {data_root}")
    if celeba_stats:
        print(f"CelebA images: {celeba_stats.image_count}")
    if lfw_stats:
        print(f"LFW images: {lfw_stats.image_count}")
    print(f"Report: {report_path}")
    for fig in figures:
        print(f"Figure: {report_dir / fig}")


if __name__ == "__main__":
    main()
