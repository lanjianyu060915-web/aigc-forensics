from __future__ import annotations

import cv2
import numpy as np


def _letterbox(image: np.ndarray, mask: np.ndarray, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    mask_canvas = np.zeros((size, size), dtype=np.uint8)
    x, y = (size - nw) // 2, (size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    mask_canvas[y:y + nh, x:x + nw] = resized_mask
    return canvas, mask_canvas


def extract_global(image: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    image, mask = _letterbox(image, mask)
    valid = mask > 0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [24, 16], [0, 180, 0, 256]).flatten()
    hist /= np.linalg.norm(hist) + 1e-12

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fill = int(np.median(gray[valid])) if valid.any() else 127
    filled = gray.copy()
    filled[~valid] = fill
    thumb = cv2.resize(filled, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-6)
    edges = cv2.Canny(filled, 60, 160)
    edge_thumb = cv2.resize(edges, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    # A face swap normally preserves the outer scene. This fixed central exclusion
    # supplies a detector-independent, deliberately non-face comparison channel.
    border_mask = mask.copy()
    h, w = border_mask.shape
    cv2.ellipse(border_mask, (w // 2, h // 2), (round(w * 0.30), round(h * 0.38)), 0, 0, 360, 0, -1)
    border_image = cv2.GaussianBlur(image, (5, 5), 0).astype(np.float32)
    border_small = cv2.resize(border_image, (64, 64), interpolation=cv2.INTER_AREA)
    border_valid = cv2.resize(border_mask, (64, 64), interpolation=cv2.INTER_NEAREST) > 0
    return {
        "hist": hist.astype(np.float32),
        "thumb": thumb.flatten(),
        "edge": edge_thumb.flatten(),
        "border": border_small,
        "border_valid": border_valid,
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip((float(np.dot(a, b)) / denom + 1.0) / 2.0, 0.0, 1.0))


def global_similarity_with_components(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray]
) -> tuple[float, dict[str, float | str | dict[str, float]]]:
    """Return the frozen global score together with its real sub-scores.

    The arithmetic and operation order below are deliberately identical to the
    original :func:`global_similarity` implementation.  The additional mapping
    is an observability side channel only; callers must continue to rank using
    the returned ``score`` value.
    """

    hist = float(np.clip(cv2.compareHist(a["hist"], b["hist"], cv2.HISTCMP_CORREL), -1, 1))
    hist = (hist + 1.0) / 2.0
    common = a["border_valid"] & b["border_valid"]
    if int(common.sum()) >= 64:
        mae = float(np.abs(a["border"][common] - b["border"][common]).mean())
        border = float(np.exp(-mae / 45.0))
    else:
        border = 0.0
    grayscale = _cosine(a["thumb"], b["thumb"])
    edge = _cosine(a["edge"], b["edge"])
    score = float(np.clip(
        0.30 * hist
        + 0.25 * grayscale
        + 0.15 * edge
        + 0.30 * border,
        0,
        1,
    ))
    return score, {
        "hsv_histogram_similarity": hist,
        "grayscale_thumbnail_similarity": grayscale,
        "edge_map_similarity": edge,
        "border_region_similarity": border,
        "border_common_pixel_count": int(common.sum()),
        "weights": {
            "hsv_histogram_similarity": 0.30,
            "grayscale_thumbnail_similarity": 0.25,
            "edge_map_similarity": 0.15,
            "border_region_similarity": 0.30,
        },
        "weighted_terms": {
            "hsv_histogram_similarity": 0.30 * hist,
            "grayscale_thumbnail_similarity": 0.25 * grayscale,
            "edge_map_similarity": 0.15 * edge,
            "border_region_similarity": 0.30 * border,
        },
        "formula": (
            "0.30 * hsv_histogram_similarity + "
            "0.25 * grayscale_thumbnail_similarity + "
            "0.15 * edge_map_similarity + "
            "0.30 * border_region_similarity"
        ),
        "score": score,
    }


def global_similarity(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    """Return the frozen global score (backward-compatible public API)."""

    score, _ = global_similarity_with_components(a, b)
    return score
