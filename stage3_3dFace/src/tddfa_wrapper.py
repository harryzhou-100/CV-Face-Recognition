"""Thin wrapper around 3DDFA_V2 for single-image 3D face reconstruction."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
TDDFA_ROOT = ROOT / "third_party" / "3DDFA_V2"


def _ensure_tddfa_path() -> None:
    p = str(TDDFA_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_tddfa(use_onnx: bool = True, gpu_mode: bool = True, config_rel: str = "configs/mb1_120x120.yml"):
    _ensure_tddfa_path()
    cfg_path = TDDFA_ROOT / config_rel
    cfg = yaml.load(open(cfg_path), Loader=yaml.SafeLoader)
    for key in ("checkpoint_fp", "bfm_fp", "param_mean_std_fp", "onnx_fp"):
        if key in cfg and cfg[key] and not Path(cfg[key]).is_absolute():
            cfg[key] = str((TDDFA_ROOT / cfg[key]).resolve())

    if use_onnx:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        from FaceBoxes import FaceBoxes
        from TDDFA import TDDFA

        face_boxes = FaceBoxes()
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    return face_boxes, tddfa, cfg


def reconstruct_image(
    img_bgr: np.ndarray,
    face_boxes,
    tddfa,
    dense: bool = True,
    face_index: int = 0,
) -> dict[str, Any]:
    """Detect face, regress 3DMM params, return vertices and metadata."""
    boxes = face_boxes(img_bgr)
    if not boxes:
        raise RuntimeError("No face detected in image")

    param_lst, roi_box_lst = tddfa(img_bgr, boxes)
    if face_index >= len(param_lst):
        face_index = 0

    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense)
    param = param_lst[face_index]
    roi_box = roi_box_lst[face_index]
    vertices = ver_lst[face_index]  # (3, N)

    return {
        "vertices": vertices,
        "triangles": tddfa.tri,
        "param": param,
        "roi_box": roi_box,
        "n_faces": len(boxes),
        "img_shape": img_bgr.shape,
    }


def save_mesh_assets(
    img_bgr: np.ndarray,
    recon: dict[str, Any],
    stem: str,
    out_mesh_dir: Path,
) -> dict[str, Path]:
    _ensure_tddfa_path()
    from utils.serialization import ser_to_obj_single, ser_to_ply_single

    out_mesh_dir.mkdir(parents=True, exist_ok=True)
    ver_lst = [recon["vertices"]]
    tri = recon["triangles"]
    h = img_bgr.shape[0]

    obj_fp = out_mesh_dir / f"{stem}.obj"
    ply_fp = out_mesh_dir / f"{stem}.ply"
    # ser_to_*_single appends _1 when one face; write directly for stable names
    from utils.serialization import get_colors, header_temp, get_suffix

    ver = ver_lst[0]
    colors = get_colors(img_bgr, ver)
    n_face = tri.shape[0]
    with open(obj_fp, "w") as f:
        for j in range(ver.shape[1]):
            x, y, z = ver[:, j]
            f.write(f"v {x:.4f} {h - y:.4f} {z:.4f} {colors[j, 0]:.4f} {colors[j, 1]:.4f} {colors[j, 2]:.4f}\n")
        for j in range(n_face):
            i1, i2, i3 = tri[j]
            f.write(f"f {i3 + 1}/{i3 + 1} {i2 + 1}/{i2 + 1} {i1 + 1}/{i1 + 1}\n")
    with open(ply_fp, "w") as f:
        hdr = header_temp.format(ver.shape[1], n_face)
        f.write(hdr + "\n")
        for j in range(ver.shape[1]):
            x, y, z = ver[:, j]
            f.write(f"{x:.2f} {h - y:.2f} {z:.2f}\n")
        for j in range(n_face):
            i1, i2, i3 = tri[j]
            f.write(f"3 {i3} {i2} {i1}\n")

    verts = recon["vertices"].T.astype(np.float32)
    tri_np = tri.astype(np.int64)
    np.savez(
        out_mesh_dir / f"{stem}.npz",
        vertices=verts,
        faces=tri_np,
        param=recon["param"],
    )
    return {"obj": obj_fp, "ply": ply_fp, "npz": out_mesh_dir / f"{stem}.npz"}
