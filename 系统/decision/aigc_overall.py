# -*- coding: utf-8 -*-
"""把通用 AIGC 分支和人脸分支并入鉴伪结果，并按档位给出总体结论。

三路分数不相加：曾在两个数据集间试过加权融合，外部集 AUC 从 0.898 掉到 0.439。
总体档位只由档位推导，原有整图分支的字段保持不变，方便逐路复核。

规则依据（交叉评测，见 文档/评测）：
- 通用分支是主结论。它在换脸内容上接近随机，但对真实内容误报低。
- 人脸分支（CommunityForensics 打人脸裁剪图）能补换脸内容，所以可以把档位抬一级。
- 整图分支的边界只在高清人像上标定过，普通真实视频里 63–82% 都会触发“高风险”，
  DFDC 真人视频 AUC 只有 0.52，因此不影响总体档位，只作提示。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

BAND_LABELS = {"low": "证据不足", "medium": "需人工复核", "high": "高风险提示"}
_ORDER = {"low": 0, "medium": 1, "high": 2}
RULE_NAME = "general_face_band_rule_v2"
WHOLE_IMAGE_NOTE = "整图分支提示风险，但它的边界只在高清人像上标定过，普通内容上误报很高，这里只作提示。"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _seconds(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _whole_image_band(payload: Mapping[str, Any]) -> str | None:
    band = _mapping(payload.get("details")).get("band")
    if band in _ORDER:
        return str(band)
    # 冻结人脸裁剪分支没有分带，只有二值判定。
    if payload.get("status") == "success" and payload.get("is_deepfake") is True:
        return "medium"
    if payload.get("status") == "success" and payload.get("is_deepfake") is False:
        return "low"
    return None


def decide_overall(
    general_band: str | None,
    face_band: str | None,
    whole_image_band: str | None,
) -> tuple[str | None, str]:
    """返回（总体档位，理由）。通用分支定主结论，人脸分支最多抬一级，整图分支不影响档位。"""
    if general_band is None:
        if face_band is not None:
            capped = "medium" if face_band == "high" else face_band
            return capped, "通用 AIGC 分支没有可用档位，只有人脸分支给出了档位，高风险按需人工复核处理。"
        if whole_image_band is None:
            return None, "各分支都没有给出可用档位。"
        capped = "medium" if whole_image_band == "high" else whole_image_band
        return capped, "只有整图分支给出了档位。它的边界只在高清人像上标定过，高风险按需人工复核处理。"
    reasons: list[str] = []
    band = general_band
    if face_band in {"medium", "high"}:
        if general_band == "low":
            band = "medium"
            reasons.append("通用 AIGC 分支未见异常，但人脸分支提示风险，两路不一致，建议人工复核。")
        elif general_band == "medium" and face_band == "high":
            band = "high"
            reasons.append("通用 AIGC 分支需人工复核，人脸分支高风险，两路都提示风险。")
        else:
            reasons.append("通用 AIGC 分支和人脸分支都提示风险。")
    elif face_band == "low":
        reasons.append("以通用 AIGC 分支为主，人脸分支未见异常。")
    else:
        reasons.append("以通用 AIGC 分支为主。")
    if band == "low" and whole_image_band in {"medium", "high"}:
        reasons.append(WHOLE_IMAGE_NOTE)
    return band, "".join(reasons)


def _branch_summary(key: str, label: str, payload: Mapping[str, Any], band: str | None) -> dict[str, Any]:
    details = _mapping(payload.get("details"))
    return {
        "key": key,
        "label": label,
        "status": payload.get("status"),
        "band": band,
        "band_label": payload.get("band_label") or details.get("band_label") or BAND_LABELS.get(band or ""),
        "score": payload.get("score") if payload.get("score") is not None else payload.get("logit"),
        "logit": payload.get("logit"),
        "error_message": payload.get("error_message"),
    }


def attach_general_aigc(
    deepfake: Mapping[str, Any], general: Mapping[str, Any]
) -> dict[str, Any]:
    """返回新结果：原有字段保留，增加 general_aigc 和 overall；不修改入参。"""

    merged = dict(deepfake)
    general_data = dict(general)
    merged["general_aigc"] = general_data
    general_ok = general_data.get("status") == "success"
    whole_image_ok = deepfake.get("status") == "success"
    warnings = list(merged.get("warnings") or [])
    general_band = general_data.get("band") if general_ok and general_data.get("band") in _ORDER else None
    face = _mapping(general_data.get("face_branch"))
    face_band = face.get("band") if face.get("status") == "success" and face.get("band") in _ORDER else None
    whole_image_band = _whole_image_band(deepfake) if whole_image_ok else None

    if general_ok and not whole_image_ok:
        # 整图分支失败时任务仍有可用结论；失败原因保留，不当作没发生。
        warnings.append(f"整图分支未完成：{deepfake.get('error_message') or '未知原因'}")
        merged.update(
            status="success",
            error_message=None,
            legacy_error_message=deepfake.get("error_message"),
            score=general_data.get("score"),
            is_deepfake=None,
            label=general_band or "uncalibrated",
        )
        merged["details"] = {
            **_mapping(deepfake.get("details")),
            "band": general_band,
            "band_label": general_data.get("band_label"),
            "band_reason": general_data.get("band_reason"),
            "band_thresholds": general_data.get("band_thresholds"),
            "score_semantics": "通用 AIGC 分支 logit 的 sigmoid 展示值，不是校准过的概率。",
        }
    elif not general_ok:
        warnings.append(f"通用 AIGC 分支未完成：{general_data.get('error_message') or '未知原因'}")

    band, reason = decide_overall(general_band, face_band, whole_image_band)
    merged["warnings"] = warnings
    merged["runtime_seconds"] = _seconds(deepfake.get("runtime_seconds")) + _seconds(
        general_data.get("runtime_seconds")
    )
    merged["overall"] = {
        "band": band,
        "band_label": BAND_LABELS.get(band or "") or "未标定",
        "reason": reason,
        "rule": RULE_NAME,
        "branches": [
            _branch_summary("general_aigc", "通用 AIGC 检测", general_data, general_band),
            _branch_summary(
                "face",
                "人脸检测",
                {
                    **face,
                    "status": face.get("status", "skipped"),
                    "error_message": face.get("error_message") or ("这次没有运行" if not face else None),
                },
                face_band,
            ),
            _branch_summary("full_image", "整图检测（仅高清人像可信）", deepfake, whole_image_band),
        ],
    }
    return merged
