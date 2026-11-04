# -*- coding: utf-8 -*-
"""把通用 AIGC 分支并入鉴伪结果，并按两路分档给出总体档位。

两路分数不相加：曾在两个数据集间试过加权融合，外部集 AUC 从 0.898 掉到 0.439。
总体档位只由档位推导，原有分支的字段保持不变，方便逐路复核。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

BAND_LABELS = {"low": "证据不足", "medium": "需人工复核", "high": "高风险提示"}
_ORDER = {"low": 0, "medium": 1, "high": 2}
RULE_NAME = "two_branch_band_rule_v1"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _legacy_band(payload: Mapping[str, Any]) -> str | None:
    band = _mapping(payload.get("details")).get("band")
    if band in _ORDER:
        return str(band)
    # 冻结人脸裁剪分支没有分带，只有二值判定。
    if payload.get("status") == "success" and payload.get("is_deepfake") is True:
        return "medium"
    if payload.get("status") == "success" and payload.get("is_deepfake") is False:
        return "low"
    return None


def decide_overall(general_band: str | None, legacy_band: str | None) -> tuple[str | None, str]:
    """通用分支定主结论；另一路只能把“证据不足”抬到“需人工复核”，不能单独定高风险。

    依据：整图分支在 SWAN-DF 上 AUC 约 0.90，但案例 3 的 20 张原图有 15 张报高风险，
    DFDC 真人视频中高风险误报 53.3%，单独定高风险会大量误报。
    """
    if general_band is None and legacy_band is None:
        return None, "两路都没有给出可用档位。"
    if general_band is None:
        capped = "medium" if legacy_band == "high" else legacy_band
        return capped, "通用 AIGC 分支没有可用档位，只参考整图/人脸分支，高风险按需人工复核处理。"
    if legacy_band is None:
        return general_band, "只有通用 AIGC 分支给出了档位。"
    if general_band == "low" and _ORDER[legacy_band] >= _ORDER["medium"]:
        return "medium", "通用 AIGC 分支未见异常，但整图/人脸分支提示风险，两路不一致，建议人工复核。"
    if general_band == legacy_band:
        return general_band, "两路档位一致。"
    return general_band, "以通用 AIGC 分支为主；整图/人脸分支结果见明细。"


def _branch_summary(key: str, label: str, payload: Mapping[str, Any], band: str | None) -> dict[str, Any]:
    details = _mapping(payload.get("details"))
    return {
        "key": key,
        "label": label,
        "status": payload.get("status"),
        "band": band,
        "band_label": payload.get("band_label") or details.get("band_label") or BAND_LABELS.get(band or ""),
        "score": payload.get("score"),
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
    legacy_ok = deepfake.get("status") == "success"
    warnings = list(merged.get("warnings") or [])
    general_band = general_data.get("band") if general_ok and general_data.get("band") in _ORDER else None
    legacy_band = _legacy_band(deepfake) if legacy_ok else None

    if general_ok and not legacy_ok:
        # 原有分支失败时任务仍有可用结论；失败原因保留，不当作没发生。
        warnings.append(f"整图/人脸分支未完成：{deepfake.get('error_message') or '未知原因'}")
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

    band, reason = decide_overall(general_band, legacy_band)
    merged["warnings"] = warnings
    merged["overall"] = {
        "band": band,
        "band_label": BAND_LABELS.get(band or "") or "未标定",
        "reason": reason,
        "rule": RULE_NAME,
        "branches": [
            _branch_summary("general_aigc", "通用 AIGC 检测", general_data, general_band),
            _branch_summary("full_image_face", "整图/人脸检测", deepfake, legacy_band),
        ],
    }
    return merged
