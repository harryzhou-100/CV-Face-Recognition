#!/usr/bin/env python3
"""Generate analysis charts and markdown report from reconstruction outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_summary(out_root: Path) -> list[dict]:
    fp = out_root / "reconstruction_summary.json"
    if not fp.exists():
        raise FileNotFoundError(f"Run reconstruct.py first: {fp}")
    with open(fp) as f:
        return json.load(f)


def plot_pose_bars(summary: list[dict], fig_dir: Path) -> Path:
    names = [s["stem"] for s in summary]
    pitch = [s["pose_deg"]["pitch"] for s in summary]
    yaw = [s["pose_deg"]["yaw"] for s in summary]
    roll = [s["pose_deg"]["roll"] for s in summary]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    ax.bar(x - w, pitch, w, label="pitch")
    ax.bar(x, yaw, w, label="yaw")
    ax.bar(x + w, roll, w, label="roll")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("degrees")
    ax.set_title("Estimated head pose (3DMM camera)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out = fig_dir / "pose_estimation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_mesh_stats(summary: list[dict], fig_dir: Path) -> Path:
    stems = [s["stem"] for s in summary]
    nv = [s["n_vertices"] for s in summary]
    nf = [s["n_faces"] for s in summary]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].bar(stems, nv, color="#4C72B0")
    axes[0].set_title("Vertex count (dense BFM)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(stems, nf, color="#55A868")
    axes[1].set_title("Triangle count")
    axes[1].tick_params(axis="x", rotation=20)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    out = fig_dir / "mesh_statistics.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_comparison_panel(summary: list[dict], out_root: Path, fig_dir: Path) -> Path | None:
    if not summary:
        return None
    n = min(len(summary), 4)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.array([axes])

    for i, s in enumerate(summary[:n]):
        img = cv2.imread(s["image"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        ov = out_root / "renders" / s["stem"] / "overlay_3ddfa.jpg"
        if ov.exists():
            axes[i, 1].imshow(cv2.cvtColor(cv2.imread(str(ov)), cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title("3D overlay")
        axes[i, 1].axis("off")

        mv = Path(s["multiview"])
        if mv.exists():
            axes[i, 2].imshow(cv2.cvtColor(cv2.imread(str(mv)), cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title("Multi-view")
        axes[i, 2].axis("off")

    out = fig_dir / "reconstruction_comparison.png"
    fig.suptitle("Single-image 3D face reconstruction pipeline", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def write_report(summary: list[dict], fig_paths: dict, report_fp: Path, backend: str) -> None:
  lines = [
    "# Stage 3：单张图像 3D 人脸重建实验报告",
    "",
    "## 1. 任务与目标",
    "",
    "从单张 RGB 人脸图像估计 **3D Morphable Model (3DMM)** 参数，重建稠密 3D 人脸网格，",
    "并使用 **PyTorch3D / OpenGL (pyrender) / Matplotlib** 进行多角度渲染与可视化分析。",
    "",
    "## 2. 算法选型",
    "",
    "| 模块 | 方案 | 说明 |",
    "|------|------|------|",
    "| 3D 重建 | **3DDFA_V2** (ECCV 2020) | 回归 62 维 3DMM 参数，BFM 稠密顶点 (~38k) |",
    "| 人脸检测 | FaceBoxes + ONNX | 与 3DDFA 官方实现一致 |",
    "| 渲染 | 自动选择后端 | 优先 PyTorch3D，其次 OpenGL，回退 Matplotlib |",
    "",
    f"本次运行渲染后端：**{backend}**",
    "",
    "## 3. 重建结果摘要",
    "",
    "| 样本 | 顶点数 | 三角面数 | Pitch° | Yaw° | Roll° |",
    "|------|--------|----------|--------|------|-------|",
  ]
  for s in summary:
    p = s["pose_deg"]
    lines.append(
      f"| {s['stem']} | {s['n_vertices']} | {s['n_faces']} | {p['pitch']:.1f} | {p['yaw']:.1f} | {p['roll']:.1f} |"
    )
  lines += [
    "",
    "## 4. 可视化",
    "",
    "### 4.1 输入 / 叠加 / 多角度",
    "",
    f"![comparison](figures/reconstruction_comparison.png)",
    "",
    "### 4.2 姿态估计",
    "",
    f"![pose](figures/pose_estimation.png)",
    "",
    "### 4.3 网格规模",
    "",
    f"![mesh stats](figures/mesh_statistics.png)",
    "",
    "## 5. 输出目录",
    "",
    "| 路径 | 内容 |",
    "|------|------|",
    "| `outputs/meshes/*.obj` | 带纹理顶点色的 OBJ 模型 |",
    "| `outputs/meshes/*.ply` | PLY 点云/网格 |",
    "| `outputs/renders/<stem>/` | 单视角渲染与 3DDFA 叠加图 |",
    "| `outputs/multiview/` | 12 视角合成图 |",
    "",
    "## 6. 结论",
    "",
    "3DDFA_V2 可在单张图像上快速得到与 BFM 对齐的稠密 3D 人脸；",
    "结合参数化姿态可解释地控制多角度渲染。",
    "纹理细节受限于 3DMM 子空间，极端姿态或遮挡时误差增大。",
    "",
  ]
  report_fp.parent.mkdir(parents=True, exist_ok=True)
  report_fp.write_text("\n".join(lines), encoding="utf-8")


def main():
    out_root = ROOT / "outputs"
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(out_root)
    backend = summary[0].get("render_backend", "matplotlib") if summary else "matplotlib"

    plot_pose_bars(summary, fig_dir)
    plot_mesh_stats(summary, fig_dir)
    plot_comparison_panel(summary, out_root, fig_dir)

    write_report(summary, {}, ROOT / "reports" / "experiment_report.md", backend)
    print(f"Report -> {ROOT / 'reports' / 'experiment_report.md'}")


if __name__ == "__main__":
    main()
