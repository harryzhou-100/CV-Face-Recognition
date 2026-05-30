#!/usr/bin/env python3
"""Single/batch 3D face reconstruction with 3DDFA_V2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mesh_utils import parse_pose_from_param, simplify_for_render, vertices_to_render_coords
from src.render_dispatch import pick_backend, render_multiview, render_view
from src.tddfa_wrapper import load_tddfa, reconstruct_image, save_mesh_assets, TDDFA_ROOT


def load_config(cfg_fp: Path) -> dict:
    with open(cfg_fp) as f:
        return json.load(f)


def process_one(
    img_fp: Path,
    face_boxes,
    tddfa,
    cfg: dict,
    out_root: Path,
) -> dict:
    img_fp = img_fp.resolve()
    img = cv2.imread(str(img_fp))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_fp}")

    recon = reconstruct_image(img, face_boxes, tddfa, dense=cfg.get("dense_mesh", True))
    stem = img_fp.stem
    mesh_dir = out_root / "meshes"
    paths = save_mesh_assets(img, recon, stem, mesh_dir)

    verts_img = recon["vertices"]
    verts_full = vertices_to_render_coords(verts_img, img.shape[0])
    faces_full = recon["triangles"].astype(np.int64)
    verts, faces = simplify_for_render(verts_full, faces_full, max_faces=2000)
    pitch, yaw, roll = parse_pose_from_param(recon["param"])

    backend = pick_backend(cfg.get("render_backend", "auto"))
    angles = cfg.get("multiview_angles_deg", list(range(0, 360, 30)))
    elev = cfg.get("elevation_deg", 15)

    render_dir = out_root / "renders" / stem
    render_view(verts, faces, azim=yaw, elev=elev, out_fp=render_dir / "front.png", backend=backend)
    multiview_fp = out_root / "multiview" / f"{stem}_multiview.png"
    render_multiview(verts, faces, angles, elev, multiview_fp, backend=backend)

    if cfg.get("save_3ddfa_overlay", True):
        overlay_fp = out_root / "renders" / stem / "overlay_3ddfa.jpg"
        _save_3ddfa_overlay(img, recon, overlay_fp)

    meta = {
        "image": str(img_fp),
        "stem": stem,
        "n_vertices": int(verts_full.shape[0]),
        "n_faces": int(faces_full.shape[0]),
        "n_faces_render": int(faces.shape[0]),
        "n_detected_faces": recon["n_faces"],
        "pose_deg": {"pitch": pitch, "yaw": yaw, "roll": roll},
        "mesh_obj": str(paths["obj"]),
        "mesh_ply": str(paths["ply"]),
        "render_backend": backend,
        "multiview": str(multiview_fp),
    }
    return meta


def _save_3ddfa_overlay(img, recon, out_fp: Path) -> None:
    sys.path.insert(0, str(TDDFA_ROOT))
    from utils.render import render as render_overlay

    ver_lst = [recon["vertices"]]
    tri = recon["triangles"]
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    render_overlay(img.copy(), ver_lst, tri, alpha=0.55, show_flag=False, wfp=str(out_fp))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "pipeline.json"))
    parser.add_argument("--input", type=str, default=None, help="Image file or directory")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    input_dir = Path(args.input) if args.input else ROOT / cfg["input_dir"]
    out_root = ROOT / cfg["output_dir"]

    if input_dir.is_file():
        images = [input_dir]
    else:
        images = sorted(
            p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
    if not images:
        raise SystemExit(f"No images in {input_dir}")

    face_boxes, tddfa, _ = load_tddfa(
        use_onnx=cfg.get("use_onnx", True),
        gpu_mode=cfg.get("gpu_mode", True),
        config_rel=cfg.get("tddfa_config", "configs/mb1_120x120.yml"),
    )

    results = []
    for img_fp in images:
        print(f"Processing {img_fp.name} ...")
        meta = process_one(img_fp, face_boxes, tddfa, cfg, out_root)
        results.append(meta)
        print(f"  -> {meta['mesh_obj']}")

    summary_fp = out_root / "reconstruction_summary.json"
    with open(summary_fp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {summary_fp}")


if __name__ == "__main__":
    main()
