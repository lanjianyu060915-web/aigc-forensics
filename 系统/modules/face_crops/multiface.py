"""在保留旧最大脸文件的同时保存可展示的逐脸裁剪。"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import cv2

from cropper import CROP_MARGIN, MIN_FACE_PIXELS, crop_bounds, square_crop
from audit import base_check, frame_identifier, summarize


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
    audit: dict[str, Any] | None = None,
    frame_id: str | None = None,
    detection_records: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """最大脸按旧编码落盘，其余脸使用新增文件名；调用方提供面积排序。"""
    legacy, per_face = [], []
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_id = frame_id or frame_identifier(source)
    records = detection_records if detection_records is not None else [
        {"face_id": index, "bbox": list(box), "selected_for_crop": True}
        for index, box in enumerate(boxes, 1)]
    checks = []
    for record in records:
        check = base_check(source, frame_id, record["face_id"], record.get("bbox"),
                           timestamp_seconds=timestamp_seconds)
        check.update({key: value for key, value in record.items() if key not in {"bbox", "face_id"}})
        if not check.get("selected_for_crop"):
            reason = check.get("exclusion_reason") or "over_limit"
            check.update(crop_status=reason, exclusion_reason=reason)
        checks.append(check)
    if audit is not None:
        audit["face_checks"].extend(checks)
    by_id = {row["face_id"]: row for row in checks}
    for index, box in enumerate(boxes, 1):
        check = by_id[index]
        short_side = min(box[2] - box[0], box[3] - box[1])
        if short_side <= 0:
            check.update(crop_status="invalid_bbox", exclusion_reason="invalid_bbox")
            continue
        legacy_selected = index == 1 and short_side >= MIN_FACE_PIXELS
        minimum = MIN_FACE_PIXELS if legacy_selected else 1
        check.update(short_edge_px=short_side, too_small=short_side < min_face_side,
                     legacy_selected=legacy_selected,
                     exclusion_reason=None if legacy_selected else "not_selected_by_frozen_rule")
        try:
            crop = square_crop(frame, box, min_pixels=minimum)
        except Exception as exc:
            check.update(crop_status="crop_failed", exclusion_reason="crop_failed", error_type=type(exc).__name__)
            if audit is not None:
                summarize(audit)
            raise
        if crop is None:
            check.update(crop_status="crop_failed", exclusion_reason="crop_failed")
            continue
        name = f"{prefix}_face.jpg" if legacy_selected else f"{prefix}_face_{index:03d}.jpg"
        try:
            ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 98])
        except Exception as exc:
            check.update(crop_status="encode_failed", exclusion_reason="encode_failed", error_type=type(exc).__name__)
            if audit is not None:
                summarize(audit)
            raise
        if not ok:
            check.update(crop_status="encode_failed", exclusion_reason="encode_failed")
            continue
        try:
            encoded.tofile(str(out_dir / name))
        except Exception as exc:
            check.update(crop_status="write_failed", exclusion_reason="write_failed", error_type=type(exc).__name__)
            if audit is not None:
                summarize(audit)
            raise
        bounds = crop_bounds(frame.shape, box, min_pixels=minimum)
        x0, y0, x1, y1 = bounds
        check.update(crop_status="success", file=name,
                     transform={"kind": "crop_translation", "from": "crop_pixels", "to": "original_image",
                                "crop_box_original": [x0, y0, x1, y1],
                                "crop_to_original": [[1, 0, x0], [0, 1, y0], [0, 0, 1]],
                                "source_dimensions": [int(frame.shape[1]), int(frame.shape[0])],
                                "crop_dimensions": [int(crop.shape[1]), int(crop.shape[0])]},
                     model_input={"file": name, "sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
                                  "encoding": "jpeg", "jpeg_quality": 98, "crop_margin": CROP_MARGIN,
                                  "crop_recipe_version": "legacy-margin1.3-jpeg98-v1",
                                  "model_preprocessing": "unrecorded"})
        old = {"file": name, "source": source, "bbox": list(box)}
        if is_video:
            old["timestamp_seconds"] = timestamp_seconds
        if legacy_selected:
            legacy.append(old)
        per_face.append({**check, **old, "face_id": index, "short_edge_px": short_side,
                         "too_small": short_side < min_face_side, "legacy_selected": legacy_selected,
                         "timestamp_seconds": timestamp_seconds,
                         "reliability_note": "人脸过小，结果不可靠" if short_side < min_face_side else "未触发小脸限制"})
    if audit is not None:
        summarize(audit)
    return legacy, per_face
