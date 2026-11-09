"""参考入库的固定尺度回退；只在未检出人脸时继续，不按识别分数选尺度。"""

from __future__ import annotations

from copy import deepcopy

from .errors import (IdentitySearchError, InputValidationError, NoFaceError,
                     MultipleFacesError, InvalidEmbeddingError)


DEFAULT_REFERENCE_SCALES = (1280, 640, 320)
REFERENCE_POLICY = "reference_no_face_scale_fallback"
REFERENCE_POLICY_VERSION = 1


def reference_policy(scales):
    values = tuple(scales)
    if (not values or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                          for v in values) or len(set(values)) != len(values)):
        raise InputValidationError("参考检测尺度链必须为不重复的正整数。")
    return {"policy": REFERENCE_POLICY, "version": REFERENCE_POLICY_VERSION,
            "scale_chain": list(values)}


def apply_detection_scale(extractor, scale, ctx_id):
    """复用检测会话；与既有查询侧切换实现保持相同语义。"""
    app = getattr(extractor, "app", None)
    if app is None:
        raise InputValidationError("特征提取器没有可切换的检测模型实例。")
    detector = getattr(app, "det_model", None)
    if detector is None:
        detector = getattr(app, "models", {}).get("detection")
    if detector is None:
        raise InputValidationError("特征提取器没有可切换的检测模型实例。")
    if getattr(detector, "static_input_size", None) is not None:
        return extractor
    size = (int(scale), int(scale))
    if (getattr(extractor, "det_size", None) == size
            and getattr(app, "det_size", None) == size
            and getattr(detector, "input_size", None) == size
            and getattr(detector, "input_sizes", None) == [size]):
        return extractor
    detector.input_sizes = [size]
    detector.input_size = size
    app.det_size = size
    extractor.det_size = size
    return extractor


def extract_reference(extractor, image_path, scales):
    """返回原提取结果及逐档记录；多脸和其他异常不重试，始终恢复原尺度。"""
    policy = reference_policy(scales)
    app = extractor.app
    detector = getattr(app, "det_model", None)
    if detector is None:
        detector = getattr(app, "models", {}).get("detection")
    if detector is None:
        raise InputValidationError("特征提取器没有可切换的检测模型实例。")
    saved = [(obj, key, hasattr(obj, key), deepcopy(getattr(obj, key, None)))
             for obj, keys in ((extractor, ("det_size",)), (app, ("det_size",)),
                               (detector, ("input_size", "input_sizes")))
             for key in keys]
    attempts = []
    diagnostic = {**policy, "attempts": attempts, "detection_scale_used": None}
    try:
        for scale in policy["scale_chain"]:
            apply_detection_scale(extractor, scale, -1)
            actual_size = getattr(detector, "static_input_size", None) or (scale, scale)
            attempt = {"scale": scale, "actual_input_size": list(actual_size)}
            try:
                feature, face = extractor.extract(image_path)
            except IdentitySearchError as exc:
                attempt.update(status="error", error_type=type(exc).__name__,
                               face_count=exc.diagnostics.get("face_count"))
                attempts.append(attempt)
                exc.diagnostics["reference_detection"] = deepcopy(diagnostic)
                # 固定输入模型不能切尺度，失败后不得重复调用同一输入尺寸。
                if (not isinstance(exc, NoFaceError)
                        or getattr(detector, "static_input_size", None) is not None
                        or scale == policy["scale_chain"][-1]):
                    raise
                continue
            attempt.update(status="success", face_count=1)
            attempts.append(attempt)
            diagnostic["detection_scale_used"] = list(actual_size)
            return feature, face, diagnostic
    finally:
        for obj, key, existed, value in saved:
            if existed:
                setattr(obj, key, value)
            elif hasattr(obj, key):
                delattr(obj, key)


def inspect_reference(extractor, image_path, scales):
    """质量检查调用同一提取路径，避免检查通过却被另一策略索引的情况。"""
    try:
        _, _, diagnostic = extract_reference(extractor, image_path, scales)
        return {"face_count": 1, "reference_detection": diagnostic, "issues": []}
    except (NoFaceError, MultipleFacesError, InvalidEmbeddingError) as exc:
        return {"face_count": exc.diagnostics.get("face_count"),
                "reference_detection": exc.diagnostics.get("reference_detection"),
                "issues": [str(exc)]}
