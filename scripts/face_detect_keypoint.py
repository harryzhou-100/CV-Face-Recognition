"""
Face detection (MMDetection YOLOX-Face) + face keypoints (MMPose RTMPose-Face).

Usage:
  conda activate face_cv
  python scripts/face_detect_keypoint.py
  python scripts/face_detect_keypoint.py --input docs/samples/celeba/036580.jpg
  python scripts/face_detect_keypoint.py --input docs/samples --save
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
from mmengine.config.utils import MODULE2PACKAGE
from mmengine.registry import init_default_scope
from mmengine.utils import get_installed_path
from mmdet.utils import register_all_modules as register_mmdet_modules
from mmpose.utils import register_all_modules as register_mmpose_modules

register_mmdet_modules()
register_mmpose_modules()

from mmdet.apis import inference_detector, init_detector
from mmdet.structures import DetDataSample
from mmpose.apis import inference_topdown, init_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "docs" / "samples"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "outputs" / "face_detection"
MODELS_DIR = PROJECT_ROOT / "models" / "face"

MMPOSE_ROOT = Path(get_installed_path(MODULE2PACKAGE["mmpose"]))
DET_CONFIG = MMPOSE_ROOT / ".mim/demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py"
POSE_CONFIG = (
    MMPOSE_ROOT
    / ".mim/configs/face_2d_keypoint/rtmpose/face6/rtmpose-m_8xb256-120e_face6-256x256.py"
)

DET_WEIGHT_URL = (
    "https://download.openmmlab.com/mmpose/mmdet_pretrained/"
    "yolo-x_8xb8-300e_coco-face_13274d7c.pth"
)
POSE_WEIGHT_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth"
)
DET_WEIGHT_FILE = MODELS_DIR / "yolox_face_det.pth"
POSE_WEIGHT_FILE = MODELS_DIR / "rtmpose_face_kpt.pth"

# PyTorch 默认缓存（旧版脚本会下到这里）
TORCH_CACHE = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )


def _copy_if_valid(src: Path, dst: Path, min_bytes: int) -> bool:
    if src.is_file() and src.stat().st_size >= min_bytes:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file() or dst.stat().st_size < min_bytes:
            shutil.copy2(src, dst)
        return True
    return False


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading -> {dst.name} ...")
    urllib.request.urlretrieve(url, dst)  # noqa: S310


def ensure_project_weights() -> tuple[str, str]:
    """Prefer weights under project models/face/; sync from torch cache or URL."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not DET_WEIGHT_FILE.is_file() or DET_WEIGHT_FILE.stat().st_size < 1_000_000:
        _copy_if_valid(
            TORCH_CACHE / "yolo-x_8xb8-300e_coco-face_13274d7c.pth",
            DET_WEIGHT_FILE,
            1_000_000,
        ) or _download(DET_WEIGHT_URL, DET_WEIGHT_FILE)

    if not POSE_WEIGHT_FILE.is_file() or POSE_WEIGHT_FILE.stat().st_size < 1_000_000:
        _copy_if_valid(
            TORCH_CACHE
            / "rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth",
            POSE_WEIGHT_FILE,
            1_000_000,
        ) or _download(POSE_WEIGHT_URL, POSE_WEIGHT_FILE)

    return str(DET_WEIGHT_FILE), str(POSE_WEIGHT_FILE)


def build_models(device: str):
    det_ckpt, pose_ckpt = ensure_project_weights()
    print(f"Det weights:  {det_ckpt}")
    print(f"Pose weights: {pose_ckpt}")

    init_default_scope("mmdet")
    det_model = init_detector(str(DET_CONFIG), det_ckpt, device=device)
    init_default_scope("mmpose")
    pose_model = init_model(str(POSE_CONFIG), pose_ckpt, device=device)
    return det_model, pose_model


def detect_faces(det_model, image: np.ndarray, score_thr: float) -> DetDataSample:
    init_default_scope("mmdet")
    result = inference_detector(det_model, image)
    if isinstance(result, list):
        result = result[0]
    keep = result.pred_instances.scores > score_thr
    result.pred_instances = result.pred_instances[keep]
    return result


def estimate_keypoints(pose_model, image: np.ndarray, det_result: DetDataSample):
    bboxes = det_result.pred_instances.bboxes.cpu().numpy()
    if len(bboxes) == 0:
        return []
    init_default_scope("mmpose")
    return inference_topdown(pose_model, image, bboxes)


def draw_result(
    image: np.ndarray,
    det_result: DetDataSample,
    pose_samples: list,
) -> np.ndarray:
    """Draw face boxes (MMDet) and keypoints (MMPose) with OpenCV."""
    canvas = image.copy()
    instances = det_result.pred_instances
    bboxes = instances.bboxes.cpu().numpy().astype(int)
    scores = instances.scores.cpu().numpy()

    for box, score in zip(bboxes, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"face {score:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    for sample in pose_samples:
        kpts = sample.pred_instances.keypoints
        if hasattr(kpts, "cpu"):
            kpts = kpts.cpu().numpy()
        kpts = np.asarray(kpts).reshape(-1, 2)
        for x, y in kpts:
            if x > 0 and y > 0:
                cv2.circle(canvas, (int(x), int(y)), 2, (0, 0, 255), -1)

    return canvas


def process_image(
    image_path: Path,
    det_model,
    pose_model,
    score_thr: float,
    output_dir: Path | None,
) -> dict:
    image = mmcv.imread(str(image_path))
    det_result = detect_faces(det_model, image, score_thr)
    n_face = len(det_result.pred_instances)
    pose_samples = estimate_keypoints(pose_model, image, det_result)
    vis = draw_result(image, det_result, pose_samples)

    summary = {
        "image": str(image_path),
        "faces": n_face,
        "keypoint_sets": len(pose_samples),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{image_path.stem}_face_result.jpg"
        cv2.imwrite(str(out_path), vis)
        summary["output"] = str(out_path)
        print(f"  saved: {out_path}  (faces={n_face})")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMDetection face det + MMPose face keypoints"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Image file or directory (default: docs/samples)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Save visualizations here (default: docs/outputs/face_detection)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="cpu | cuda:0 | mps | auto",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.5,
        help="Face detection score threshold",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write result images (default: on when running batch)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write result images",
    )
    args = parser.parse_args()

    images = list_images(args.input)
    if not images:
        raise SystemExit(f"No images under {args.input}")

    save = args.save or (not args.no_save and len(images) > 0)
    out_dir = args.output_dir if save else None
    device = pick_device(args.device)

    print(f"Device: {device}")
    print(f"MMDetection detector: YOLOX-S (WIDER Face / coco-face)")
    print(f"MMPose keypoints: RTMPose-m Face6 (106 landmarks)")
    print(f"Images: {len(images)}")

    det_model, pose_model = build_models(device)

    for path in images:
        print(f"\n[{path.name}]")
        process_image(path, det_model, pose_model, args.score_thr, out_dir)

    if out_dir:
        print(f"\nAll results -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
