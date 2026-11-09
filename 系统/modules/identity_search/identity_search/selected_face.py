"""Strict opt-in contract for a preselected query face; no model loading."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import re
from typing import Any

from .errors import InputValidationError

INTERFACE_VERSION = "selected-face-v1-experimental"


def require_selected_face_opt_in(enabled: bool, experimental: bool) -> None:
    if enabled is not True or experimental is not True:
        raise InputValidationError("显式选脸接口需要同时启用 selected_face_enabled 和 selected_face_experimental。")


def validate_selected_face(value: Mapping[str, Any], *, context: Mapping[str, Any],
                           input_sha256: str, image_size: tuple[int, int]) -> dict[str, Any]:
    """Bind a selection to caller-owned task scope and the actual decoded bytes.

    context must come from the task/input binding, not be copied blindly from the
    selection. This establishes bookkeeping consistency, not real identity or
    proof that supplied landmarks came from a trustworthy detector.
    """
    if not isinstance(value, Mapping) or not isinstance(context, Mapping):
        raise InputValidationError("显式选脸及其独立来源上下文必须为对象。")
    if value.get("interface_version") != INTERFACE_VERSION:
        raise InputValidationError("显式选脸接口版本不匹配。")
    for key in ("source_id", "frame_id"):
        expected = context.get(key)
        if not isinstance(expected, str) or not expected or value.get(key) != expected:
            raise InputValidationError(f"显式选脸的 {key} 与当前输入上下文不一致。")
    digest = value.get("input_sha256")
    if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
            or digest.lower() != input_sha256.lower()):
        raise InputValidationError("显式选脸输入摘要与实际图片不一致。")
    size = value.get("image_size")
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in size)
            or tuple(size) != tuple(image_size)):
        raise InputValidationError("显式选脸记录的原图尺寸与实际图片不一致。")
    if value.get("coordinate_space") != "original_image":
        raise InputValidationError("显式选脸只接受原图坐标，变换后的坐标须先由调用方可靠还原。")
    for key in ("detection_scope_id", "detection_id"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise InputValidationError("显式选脸缺少可追溯检测来源标识。")
    def finite_number(number):
        return isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number)
    bbox = value.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not all(finite_number(v) for v in bbox):
        raise InputValidationError("显式选脸的人脸框无效。")
    left, top, right, bottom = (float(v) for v in bbox)
    width, height = image_size
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise InputValidationError("显式选脸的人脸框越出原图或没有正面积。")
    points = value.get("keypoints_5")
    if (not isinstance(points, (list, tuple)) or len(points) != 5
            or any(not isinstance(p, (list, tuple)) or len(p) != 2 or not all(finite_number(v) for v in p) for p in points)):
        raise InputValidationError("显式选脸必须包含五个有限原图关键点。")
    points = [[float(v) for v in point] for point in points]
    if any(not (left <= x <= right and top <= y <= bottom) for x, y in points):
        raise InputValidationError("显式选脸关键点不在人脸框内，拒绝不确定的坐标对应。")
    if len({tuple(point) for point in points}) != 5:
        raise InputValidationError("显式选脸关键点重复。")
    score = value.get("det_score")
    if score is not None and (not finite_number(score) or not 0 <= score <= 1):
        raise InputValidationError("显式选脸检测分数无效。")
    result = {"interface_version": INTERFACE_VERSION, "source_id": context["source_id"], "frame_id": context["frame_id"],
              "input_sha256": input_sha256.lower(), "image_size": [width, height],
              "coordinate_space": "original_image", "bbox": [left, top, right, bottom], "keypoints_5": points,
              "detection_scope_id": value["detection_scope_id"], "detection_id": value["detection_id"],
              "det_score": float(score) if score is not None else None}
    result["selection_cache_key"] = hashlib.sha256(json.dumps(
        result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    result["query_cache_policy"] = "not_cached"
    return result
