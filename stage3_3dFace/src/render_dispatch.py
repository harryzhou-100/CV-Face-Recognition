"""Select rendering backend: pytorch3d, opengl (pyrender), or matplotlib."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import render_matplotlib
from .render_opengl import pyrender_available
from .render_pytorch3d import pytorch3d_available


def pick_backend(requested: str = "auto") -> str:
    if requested in ("matplotlib", "mpl"):
        return "matplotlib"
    if requested == "pytorch3d" and pytorch3d_available():
        return "pytorch3d"
    if requested in ("opengl", "pyrender") and pyrender_available():
        return "opengl"
    if requested == "auto":
        if pytorch3d_available():
            return "pytorch3d"
        if pyrender_available():
            return "opengl"
        return "matplotlib"
    return "matplotlib"


def render_view(verts, faces, azim, elev, out_fp: Path, backend: str = "auto") -> str:
    b = pick_backend(backend)
    try:
        if b == "pytorch3d":
            from . import render_pytorch3d

            render_pytorch3d.render_mesh_view(verts, faces, azim, elev, out_fp)
        elif b == "opengl":
            from . import render_opengl

            render_opengl.render_mesh_view(verts, faces, azim, elev, out_fp)
        else:
            render_matplotlib.render_mesh_view(verts, faces, azim, elev, out_fp)
    except Exception as exc:
        print(f"  [warn] {b} render failed ({exc}), fallback to matplotlib")
        render_matplotlib.render_mesh_view(verts, faces, azim, elev, out_fp)
        b = "matplotlib"
    return b


def render_multiview(verts, faces, angles, elev, out_fp: Path, backend: str = "auto") -> str:
    b = pick_backend(backend)
    render_matplotlib.render_multiview_grid(verts, faces, angles, elev, out_fp)
    return b
