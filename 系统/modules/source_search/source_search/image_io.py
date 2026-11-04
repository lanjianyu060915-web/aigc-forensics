from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_image(path: Path) -> np.ndarray | None:
    """Decode an image while supporting non-ASCII Windows paths."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image if image is not None and image.size else None
    except (OSError, ValueError, cv2.error):
        return None


def write_image(path: Path, image: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if suffix in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise OSError(f"cannot encode image: {path}")
    encoded.tofile(path)


def scan_library(
    directory: Path,
    progress_callback: Callable[[int, int, Path, bool], None] | None = None,
) -> tuple[list[tuple[Path, np.ndarray]], list[str]]:
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"library directory does not exist: {directory}")
    images: list[tuple[Path, np.ndarray]] = []
    skipped: list[str] = []
    supported_paths = [
        path
        for path in sorted(directory.rglob("*"), key=lambda p: str(p).lower())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    total = len(supported_paths)
    for current, path in enumerate(supported_paths, 1):
        image = read_image(path)
        if image is None:
            skipped.append(str(path))
        else:
            images.append((path, image))
        if progress_callback is not None:
            progress_callback(current, total, path, image is not None)
    if not images:
        raise ValueError("library contains no decodable supported images")
    return images, skipped
