"""PyTorch3D mesh rendering (optional, GPU-accelerated)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def pytorch3d_available() -> bool:
    try:
        import pytorch3d  # noqa: F401

        return True
    except ImportError:
        return False


def render_mesh_view(
    verts: np.ndarray,
    faces: np.ndarray,
    azim: float,
    elev: float,
    out_fp: Path,
    device: str | None = None,
) -> None:
    from pytorch3d.io import load_objs_as_meshes
    from pytorch3d.renderer import (
        AmbientLights,
        FoVPerspectiveCameras,
        MeshRenderer,
        MeshRasterizer,
        PointLights,
        RasterizationSettings,
        SoftPhongShader,
        look_at_view_transform,
    )
    from pytorch3d.structures import Meshes

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from pytorch3d.renderer import TexturesVertex

    v = torch.tensor(verts, dtype=torch.float32, device=device)
    f = torch.tensor(faces, dtype=torch.int64, device=device)
    verts_rgb = torch.ones_like(v)[None]
    textures = TexturesVertex(verts_features=verts_rgb)
    meshes = Meshes(verts=[v], faces=[f], textures=textures)

    dist = 2.5
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )
    images = renderer(meshes)
    img = (images[0, ..., :3].detach().cpu().numpy() * 255).astype(np.uint8)

    import imageio

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(out_fp), img)
