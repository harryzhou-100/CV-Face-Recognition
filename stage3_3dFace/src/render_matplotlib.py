"""Matplotlib 3D mesh rendering (always available fallback)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def render_mesh_view(
    verts: np.ndarray,
    faces: np.ndarray,
    azim: float,
    elev: float,
    out_fp: Path,
    title: str | None = None,
) -> None:
    fig = plt.figure(figsize=(5, 5), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    tris = verts[faces]
    mesh = Poly3DCollection(tris, alpha=0.92, linewidths=0.05, edgecolor="#333333")
    z = verts[:, 2]
    mesh.set_facecolor(cm.coolwarm((z - z.min()) / (z.max() - z.min() + 1e-8)))
    ax.add_collection3d(mesh)
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)
    plt.tight_layout()
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render_multiview_grid(
    verts: np.ndarray,
    faces: np.ndarray,
    angles_deg: list[float],
    elev: float,
    out_fp: Path,
    cols: int = 4,
) -> None:
    n = len(angles_deg)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(3 * cols, 3 * rows), dpi=100)
    z = verts[:, 2]
    colors = cm.coolwarm((z - z.min()) / (z.max() - z.min() + 1e-8))

    for i, azim in enumerate(angles_deg):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        tris = verts[faces]
        mesh = Poly3DCollection(tris, alpha=0.9, linewidths=0.02, edgecolor="#222222")
        mesh.set_facecolor(colors)
        ax.add_collection3d(mesh)
        lim = np.abs(verts).max() * 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f"{int(azim)}°", fontsize=8)

    plt.suptitle("Multi-view 3D Face Reconstruction", fontsize=12)
    plt.tight_layout()
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, bbox_inches="tight")
    plt.close(fig)
