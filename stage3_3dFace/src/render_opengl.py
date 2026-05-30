"""OpenGL offscreen rendering via trimesh + pyrender (optional)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def pyrender_available() -> bool:
    try:
        import pyrender  # noqa: F401

        return True
    except ImportError:
        return False


def render_mesh_view(
    verts: np.ndarray,
    faces: np.ndarray,
    azim: float,
    elev: float,
    out_fp: Path,
) -> None:
    import pyrender
    from pyrender.constants import RenderFlags

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    scene = pyrender.Scene(ambient_light=[0.35, 0.35, 0.35], bg_color=[255, 255, 255, 0])
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(pr_mesh)

    r = 2.2
    az = np.radians(azim)
    el = np.radians(elev)
    cam_pose = np.array(
        [
            [np.cos(az) * np.cos(el), -np.sin(az), np.cos(az) * np.sin(el), r * np.cos(az) * np.cos(el)],
            [np.sin(az) * np.cos(el), np.cos(az), np.sin(az) * np.sin(el), r * np.sin(az) * np.cos(el)],
            [-np.sin(el), 0, np.cos(el), r * np.sin(el)],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    scene.add(camera, pose=cam_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.5)
    scene.add(light, pose=cam_pose)

    r = pyrender.OffscreenRenderer(512, 512)
    color, _ = r.render(scene, flags=RenderFlags.RGBA)
    r.delete()
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    import imageio

    imageio.imwrite(str(out_fp), color[:, :, :3])
