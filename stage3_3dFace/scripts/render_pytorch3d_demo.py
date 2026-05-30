#!/usr/bin/env python3
"""Optional: render one sample with PyTorch3D (GPU) for comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mesh_utils import simplify_for_render, vertices_to_render_coords
from src.render_pytorch3d import pytorch3d_available, render_mesh_view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default=str(ROOT / "outputs/meshes/face_03.npz"))
    parser.add_argument("--out", type=str, default=str(ROOT / "outputs/renders/face_03/front_pytorch3d.png"))
    args = parser.parse_args()

    if not pytorch3d_available():
        raise SystemExit("PyTorch3D not installed")

    data = np.load(args.npz)
    verts, faces = simplify_for_render(data["vertices"], data["faces"], max_faces=3000)
    render_mesh_view(verts, faces, azim=45, elev=15, out_fp=Path(args.out))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
