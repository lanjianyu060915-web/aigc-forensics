# -*- coding: utf-8 -*-
"""身份检索的检测尺度回退策略（CLI worker 与常驻 worker 共用）。

沿用已存在的固定尺度链，不比较分数选择尺度。实际检测器按比例缩放并补边；
未检出的原因不能仅由图像尺寸推断。

本模块只做两件事：

1. 把检测输入尺度切换到指定档位（复用同一个 FaceAnalysis 实例，不重复加载模型）；
2. 按尺度链依次尝试查询，只对「一个尺度都没检出人脸」触发下一档。

检索算法、库索引、特征口径、单脸约束都不在这里改动。

可选的主脸选择（--multi-face-policy largest_face）：查询图有多张人脸时，把面积最大以外的
人脸涂掉，再走同一条单脸检索。主脸完成后复用同一模型追加其余脸的候选，规则、
原图坐标和处理上限写进结果 JSON；主脸候选和分数不变。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence
from copy import deepcopy
import hashlib
import math

from identity_search.errors import InputValidationError, MultipleFacesError, NoFaceError
from identity_search.reference_detection import (
    DEFAULT_REFERENCE_SCALES, apply_detection_scale, reference_policy,
)


DEFAULT_SCALE_CHAIN = DEFAULT_REFERENCE_SCALES
FALLBACK_POLICY = "scale_fallback_on_no_face"
MULTI_FACE_STRICT = "strict"
MULTI_FACE_LARGEST = "largest_face"
MULTI_FACE_POLICIES = (MULTI_FACE_STRICT, MULTI_FACE_LARGEST)
MASK_EXPANSION = 0.15


def _face_area(face: dict[str, Any]) -> float:
    bbox = face.get("bbox") or [0, 0, 0, 0]
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def select_main_face(faces: Sequence[dict[str, Any]]) -> int:
    """面积最大的人脸；面积相同时取检测分高的，保证结果确定。"""

    usable = [index for index, face in enumerate(faces) if len(face.get("bbox") or []) == 4]
    if not usable:
        raise InputValidationError("多人脸诊断信息里没有人脸框，无法选择主脸。")
    return max(
        usable,
        key=lambda index: (_face_area(faces[index]), float(faces[index].get("det_score") or 0.0)),
    )


def write_main_face_query(
    query: str | Path, faces: Sequence[dict[str, Any]], selected: int, output_dir: Path,
    *, trace: dict[str, Any] | None = None,
) -> Path:
    """涂掉主脸以外的人脸，写成任务目录里的一张新查询图；原始查询图不动。"""

    import cv2
    import numpy as np

    data = np.fromfile(str(query), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise InputValidationError(f"无法解码查询图：{query}")
    height, width = image.shape[:2]
    mask_regions = []
    fill = tuple(int(value) for value in image.reshape(-1, 3).mean(axis=0))
    for index, face in enumerate(faces):
        bbox = face.get("bbox") or []
        if index == selected or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in bbox)
        pad_x = (x2 - x1) * MASK_EXPANSION
        pad_y = (y2 - y1) * MASK_EXPANSION
        left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        right, bottom = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
        image[top:bottom, left:right] = fill
        selected_box = faces[selected].get("bbox") or []
        overlap = None
        if len(selected_box) == 4:
            sx1, sy1, sx2, sy2 = (float(value) for value in selected_box)
            overlap = max(left, sx1) < min(right, sx2) and max(top, sy1) < min(bottom, sy2)
        mask_regions.append({"detector_index": index, "bbox_original": [left, top, right, bottom],
                             "overlaps_selected_bbox": overlap})
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "query_main_face_only.png"
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise InputValidationError("主脸查询图编码失败。")
    encoded.tofile(str(path))
    if trace is not None:
        trace.update(schema_version="1.0", mode="legacy_mask_then_redetect",
                     original_input_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                     original_digest_basis="bytes_decoded_for_masking", first_detection_input_sha256=None,
                     masked_input_sha256=hashlib.sha256(encoded.tobytes()).hexdigest(),
                     image_size=[width, height], coordinate_space="original_image",
                     transform={"kind": "identity", "geometry_unchanged": True},
                     mask_expansion=MASK_EXPANSION, mask_regions=mask_regions,
                     selected_detector_index=selected)
    return path


def selection_trace(first_face: dict[str, Any], result: dict[str, Any] | None,
                    processing: dict[str, Any] | None = None, *, error_type: str | None = None) -> dict[str, Any]:
    """Record first selection and actual second detection without changing candidates."""
    trace = deepcopy(processing or {})
    trace.setdefault("schema_version", "1.0")
    trace.setdefault("mode", "legacy_mask_then_redetect")
    trace["first_detection"] = {key: deepcopy(first_face.get(key)) for key in ("bbox", "keypoints_5", "det_score")}
    diagnostics = (result or {}).get("diagnostics") or {}
    detected = diagnostics.get("detected_faces") if isinstance(diagnostics, dict) else None
    second = detected[0] if isinstance(detected, list) and len(detected) == 1 and isinstance(detected[0], dict) else None
    if second is None and result is not None:
        metrics = result.get("metrics") or {}
        if isinstance(metrics, dict) and metrics.get("selected_face_bbox") is not None:
            second = {"bbox": metrics.get("selected_face_bbox"),
                      "keypoints_5": metrics.get("selected_face_keypoints_5"),
                      "det_score": metrics.get("face_detection_score")}
    trace["second_detection"] = ({key: deepcopy(second.get(key)) for key in ("bbox", "keypoints_5", "det_score")}
                                 if second is not None else None)
    def box(value):
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value):
            return None
        values = tuple(float(v) for v in value)
        return values if values[2] > values[0] and values[3] > values[1] else None
    before, after = box(first_face.get("bbox")), box((second or {}).get("bbox"))
    trace["bbox_equal"] = before == after if before is not None and after is not None else None
    first_points, second_points = first_face.get("keypoints_5"), (second or {}).get("keypoints_5")
    trace["keypoints_equal"] = first_points == second_points if first_points is not None and second_points is not None else None
    trace["correspondence_status"] = ("exact_original_geometry" if trace["bbox_equal"] is True else
                                      "geometry_changed_unverified" if trace["bbox_equal"] is False else "unrecorded")
    trace["status"] = "failed" if error_type else "recorded" if second is not None else "incomplete"
    if error_type:
        trace["error_type"] = error_type
    trace["notice"] = "仅记录选脸与重检的几何对应，不证明真人身份或特征等价。"
    return trace


def parse_scale_chain(text: str) -> tuple[int, ...]:
    """解析形如 "1280,640,320" 的检测尺度链，去重并保持顺序。"""

    scales: list[int] = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise InputValidationError(f"检测尺度必须是整数: {token}") from exc
        if value < 64:
            raise InputValidationError(f"检测尺度不得小于 64: {value}")
        if value not in scales:
            scales.append(value)
    if not scales:
        raise InputValidationError("检测尺度链不得为空")
    return tuple(scales)


def attempt_scale_chain(
    searcher: Any,
    query: str,
    scales: Sequence[int],
    extractor_factory: Callable[[int], Any],
    *,
    top_k: int,
    progress_reporter: Any = None,
    emit_progress: Callable[..., None] | None = None,
    multi_face_policy: str = MULTI_FACE_STRICT,
    main_face_dir: Path | None = None,
    include_secondary_faces: bool = True,
    **search_kwargs: Any,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """按尺度链依次尝试；返回 (成功结果或 None, 逐档记录)。

    与冻结实现的差异：某一档「一个人脸都检不到」时继续下一档。
    多人脸默认仍立即上抛；只有显式选择 largest_face 才改走主脸选择。
    解码失败等其他错误保持冻结行为，不做重试。
    """

    def _emit(stage: str, status: str, message: str, **fields: Any) -> None:
        if emit_progress is not None:
            emit_progress(progress_reporter, stage, status, message, **fields)

    attempts: list[dict[str, Any]] = []
    for index, scale in enumerate(scales):
        entry: dict[str, Any] = {"scale": scale, "tier": index}
        # 只切换查询侧尺度；参考索引已按自己的完整固定策略建立。
        searcher.extractor = extractor_factory(scale)
        _emit(
            "identity_detection_scale",
            "running",
            f"正在按检测尺度 {scale} 提取查询图人脸。",
            progress_percent=None,
            details={"scale": scale, "tier": index, "chain": list(scales)},
            force=True,
        )
        try:
            result = searcher.search(
                query, top_k=top_k, progress_reporter=progress_reporter, **search_kwargs
            )
        except NoFaceError as exc:
            diagnostics = getattr(exc, "diagnostics", {}) or {}
            entry.update(
                {
                    "status": "error",
                    "error_type": "NoFaceError",
                    "face_count": diagnostics.get("face_count"),
                }
            )
            attempts.append(entry)
            _emit(
                "identity_scale_fallback",
                "warning",
                f"检测尺度 {scale} 未检出人脸。",
                progress_percent=None,
                details={"scale": scale, "tier": index, "next_tier": index + 1},
                force=True,
            )
            continue
        except MultipleFacesError as exc:
            if multi_face_policy != MULTI_FACE_LARGEST or main_face_dir is None:
                raise
            diagnostics = getattr(exc, "diagnostics", {}) or {}
            faces = list(diagnostics.get("detected_faces") or [])
            selected = select_main_face(faces)
            processing_trace: dict[str, Any] = {}
            main_query = write_main_face_query(query, faces, selected, Path(main_face_dir), trace=processing_trace)
            _emit(
                "identity_main_face_selected",
                "warning",
                f"查询图检出 {len(faces)} 张人脸，按面积最大选择主脸后比对。",
                progress_percent=None,
                details={"face_count": len(faces), "selected_bbox": faces[selected].get("bbox")},
                force=True,
            )
            try:
                result = searcher.search(
                    str(main_query), top_k=top_k, progress_reporter=progress_reporter, **search_kwargs
                )
            except Exception as selected_error:
                failure_trace = selection_trace(faces[selected], None, processing_trace, error_type=type(selected_error).__name__)
                if isinstance(getattr(selected_error, "diagnostics", None), dict):
                    selected_error.diagnostics["selection_trace"] = failure_trace
                raise
            annotate_main_face(result, query, faces, selected, main_query,
                               trace=selection_trace(faces[selected], result, processing_trace))
            if include_secondary_faces:
                from per_face import append_face_candidates

                append_face_candidates(result, query, faces, selected, Path(main_face_dir), searcher, search_kwargs)
        else:
            if include_secondary_faces and main_face_dir is not None:
                faces = (result.get("diagnostics") or {}).get("detected_faces") or []
                if len(faces) == 1:
                    from per_face import append_face_candidates

                    append_face_candidates(result, query, faces, 0, Path(main_face_dir), searcher, search_kwargs)
        entry.update({"status": "success", "face_count": 1})
        attempts.append(entry)
        return result, attempts
    return None, attempts


def annotate_main_face(
    result: dict[str, Any],
    query: str | Path,
    faces: Sequence[dict[str, Any]],
    selected: int,
    main_query: Path,
    *, trace: dict[str, Any] | None = None,
) -> None:
    """记录主脸选择的规则和依据；结果的查询图仍指向原始文件。"""

    selection = {
        "policy": MULTI_FACE_LARGEST,
        "rule": "面积最大的人脸；面积相同时取检测分高的",
        "face_count": len(faces),
        "selected_index": selected,
        "selected_bbox": faces[selected].get("bbox"),
        "other_bboxes": [face.get("bbox") for index, face in enumerate(faces) if index != selected],
        "masked_query_file": main_query.name,
        "selection_trace": deepcopy(trace) if trace is not None else selection_trace(faces[selected], result),
    }
    result["query_image"] = str(query)
    method = result.setdefault("method", {})
    if isinstance(method, dict):
        method["face_selection"] = selection
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        metrics["query_face_count_original"] = len(faces)
        metrics["main_face_selected"] = True
    warnings = result.setdefault("warnings", [])
    if isinstance(warnings, list):
        warnings.append(
            f"查询图检出 {len(faces)} 张人脸，系统按面积最大选择了一张比对；"
            "如果要查的不是这张脸，请裁剪后重新分析。"
        )


def annotate_result(
    result: dict[str, Any], scales: Sequence[int], attempts: list[dict[str, Any]]
) -> dict[str, Any]:
    """把尺度链与实际生效档位写入结果 JSON，供人工复核。"""

    used = attempts[-1]["scale"] if attempts else scales[0]
    method = result.get("method")
    if not isinstance(method, dict):
        method = {}
        result["method"] = method
    method["preprocessing"] = {
        "policy": FALLBACK_POLICY,
        "det_size_chain": list(scales),
        "library_det_size": scales[0],
        "library_detection_policy": reference_policy(scales),
        "detection_scale_used": used,
        "fallback_applied": used != scales[0],
        "attempts": attempts,
    }
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        metrics["detection_scale_used"] = used
        metrics["detection_attempt_count"] = len(attempts)
    return result


__all__ = [
    "DEFAULT_SCALE_CHAIN",
    "FALLBACK_POLICY",
    "MULTI_FACE_LARGEST",
    "MULTI_FACE_POLICIES",
    "MULTI_FACE_STRICT",
    "annotate_main_face",
    "annotate_result",
    "apply_detection_scale",
    "attempt_scale_chain",
    "parse_scale_chain",
    "select_main_face",
    "write_main_face_query",
]
