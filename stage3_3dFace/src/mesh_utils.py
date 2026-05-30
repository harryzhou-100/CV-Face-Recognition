"""Mesh normalization and camera helpers for rendering."""

from __future__ import annotations

import numpy as np


def vertices_to_render_coords(vertices: np.ndarray, img_h: int) -> np.ndarray:
    """Convert 3DDFA image-space vertices (3,N) to centered metric coords (N,3)."""
    v = vertices.T.astype(np.float64).copy()
    v[:, 1] = img_h - v[:, 1]
    center = v.mean(axis=0)
    v -= center
    scale = np.linalg.norm(v, axis=1).max()
    if scale > 1e-6:
        v /= scale
    return v.astype(np.float32)


def simplify_for_render(verts: np.ndarray, faces: np.ndarray, max_faces: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    """Subsample faces for fast multiview rendering (full mesh still exported)."""
    if faces.shape[0] <= max_faces:
        return verts, faces
    step = max(1, faces.shape[0] // max_faces)
    return verts, faces[::step]


def parse_pose_from_param(param: np.ndarray) -> tuple[float, float, float]:
    """Euler angles (pitch, yaw, roll) in degrees from 3DMM camera param."""
    import sys
    from pathlib import Path

    tddfa = Path(__file__).resolve().parents[1] / "third_party" / "3DDFA_V2"
    if str(tddfa) not in sys.path:
        sys.path.insert(0, str(tddfa))
    from utils.pose import calc_pose

    _, pose = calc_pose(param)
    return float(pose[0]), float(pose[1]), float(pose[2])
