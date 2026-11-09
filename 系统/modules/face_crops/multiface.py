"""在保留旧最大脸文件的同时保存可展示的逐脸裁剪。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from cropper import MIN_FACE_PIXELS, square_crop


def write_frame_crops(
    frame,
    boxes: list[tuple[int, int, int, int]],
    *,
    source: str,
    prefix: str,
    out_dir: Path,
    min_face_side: int = 64,
    is_video: bool = False,
    timestamp_seconds: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """最大脸按旧编码落盘，其余脸使用新增文件名；调用方提供面积排序。"""
    legacy, per_face = [], []
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, box in enumerate(boxes, 1):
        short_side = min(box[2] - box[0], box[3] - box[1])
        if short_side <= 0:
            continue
        legacy_selected = index == 1 and short_side >= MIN_FACE_PIXELS
        crop = square_crop(frame, box, min_pixels=MIN_FACE_PIXELS if legacy_selected else 1)
        if crop is None:
            continue
        name = f"{prefix}_face.jpg" if legacy_selected else f"{prefix}_face_{index:03d}.jpg"
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 98])
        if not ok:
            continue
        encoded.tofile(str(out_dir / name))
        old = {"file": name, "source": source, "bbox": list(box)}
        if is_video:
            old["timestamp_seconds"] = timestamp_seconds
        if legacy_selected:
            legacy.append(old)
        per_face.append({**old, "face_id": index, "short_edge_px": short_side,
                         "too_small": short_side < min_face_side, "legacy_selected": legacy_selected,
                         "timestamp_seconds": timestamp_seconds,
                         "reliability_note": "人脸过小，结果不可靠" if short_side < min_face_side else "未触发小脸限制"})
    return legacy, per_face
