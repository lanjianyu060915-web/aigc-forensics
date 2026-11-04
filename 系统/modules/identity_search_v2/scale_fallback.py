# -*- coding: utf-8 -*-
"""身份检索的检测尺度回退策略（CLI worker 与常驻 worker 共用）。

证据见 docs/M2_人脸预处理_尺度回退实验.md：
冻结 worker 固定 det_size=(1280,1280)，InsightFace 会把整幅图直接缩放到该尺寸，
竖屏 720x1280 视频帧因此被横向拉伸 1.78 倍，人脸在检测网络里过大而漏检。

本模块只做两件事：

1. 把检测输入尺度切换到指定档位（复用同一个 FaceAnalysis 实例，不重复加载模型）；
2. 按尺度链依次尝试查询，只对「一个尺度都没检出人脸」触发下一档。

检索算法、库索引、特征口径、单脸约束都不在这里改动。

可选的主脸选择（--multi-face-policy largest_face）：查询图有多张人脸时，把面积最大以外的
人脸涂掉，再走同一条单脸检索。规则和被涂掉的区域都写进结果 JSON，不静默选择。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from identity_search.errors import InputValidationError, MultipleFacesError, NoFaceError


DEFAULT_SCALE_CHAIN = (1280, 640, 320)
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
    query: str | Path, faces: Sequence[dict[str, Any]], selected: int, output_dir: Path
) -> Path:
    """涂掉主脸以外的人脸，写成任务目录里的一张新查询图；原始查询图不动。"""

    import cv2
    import numpy as np

    data = np.fromfile(str(query), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise InputValidationError(f"无法解码查询图：{query}")
    height, width = image.shape[:2]
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
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "query_main_face_only.png"
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise InputValidationError("主脸查询图编码失败。")
    encoded.tofile(str(path))
    return path


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


def apply_detection_scale(extractor: Any, scale: int, ctx_id: int) -> Any:
    """切换检测输入尺度；复用已有 FaceAnalysis 实例，不重新加载模型。"""

    app = getattr(extractor, "app", None)
    if app is None:
        raise InputValidationError("特征提取器没有可切换的检测模型实例。")
    app.prepare(ctx_id=ctx_id, det_size=(scale, scale))
    if hasattr(extractor, "det_size"):
        extractor.det_size = (scale, scale)
    return extractor


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
        # 只替换查询侧检测尺度；库索引已按第一档建好，缓存键不受影响。
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
            main_query = write_main_face_query(query, faces, selected, Path(main_face_dir))
            _emit(
                "identity_main_face_selected",
                "warning",
                f"查询图检出 {len(faces)} 张人脸，按面积最大选择主脸后比对。",
                progress_percent=None,
                details={"face_count": len(faces), "selected_bbox": faces[selected].get("bbox")},
                force=True,
            )
            result = searcher.search(
                str(main_query), top_k=top_k, progress_reporter=progress_reporter, **search_kwargs
            )
            annotate_main_face(result, query, faces, selected, main_query)
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
