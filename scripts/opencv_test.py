import argparse
from pathlib import Path

import cv2
import numpy as np

def load_image(path: Path) -> np.ndarray:
    if path.is_file():
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        return image

    print(f"Image not found ({path}), using generated test image.")
    height, width = 240, 320
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 3] = (255, 0, 0)
    image[:, width // 3 : 2 * width // 3] = (0, 255, 0)
    image[:, 2 * width // 3 :] = (0, 0, 255)
    return image


def save_outputs(color: np.ndarray, gray: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    color_path = output_dir / "output_color.jpg"
    gray_path = output_dir / "output_gray.jpg"
    cv2.imwrite(str(color_path), color)
    cv2.imwrite(str(gray_path), gray)
    print(f"Saved: {color_path}")
    print(f"Saved: {gray_path}")


def show_windows(color: np.ndarray, gray: np.ndarray) -> None:
    cv2.imshow("Original (BGR)", color)
    cv2.imshow("Grayscale", gray)
    print("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCV read / grayscale / display or save")
    parser.add_argument(
        "image",
        nargs="?",
        default="test.jpg",
        help="Input image path (default: test.jpg)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save result images instead of opening windows (use in Docker)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output",
        type=Path,
        help="Directory for saved images when using --save (default: output)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    color = load_image(image_path)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    print(f"Input: {image_path.resolve() if image_path.is_file() else image_path}")
    print(f"Shape: color {color.shape}, gray {gray.shape}")

    # Docker has no display; save writes files to disk instead of showing picture
    if args.save or not _has_display():
        save_outputs(color, gray, args.output_dir)
    else:
        show_windows(color, gray)


def _has_display() -> bool:
    import os

    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


if __name__ == "__main__":
    main()
