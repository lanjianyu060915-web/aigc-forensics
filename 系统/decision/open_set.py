# -*- coding: utf-8 -*-
"""开放集三态判定层。

背景：v0.4.0 的身份检索与底图检索都是闭集 Top-K 排序，即使查询不属于库内，
系统也一定会返回一个「第一名」。本模块把排序结果转成四态结论：

- high_confidence        高可信候选
- insufficient_evidence  证据不足，建议人工复核
- no_reliable_candidate  当前库中未发现可靠候选
- unavailable            本次未能完成可比对检索（查询质量或库覆盖不足）

判据来源（详见 docs/M1_开放集三态与证据质量层.md）：

- 底图：只有 global_score 在库内命中与库外查询之间完全可分
  （库外 0.502-0.553，库内命中 0.968-0.998）。local_score、RANSAC 内点数、
  Top1-Top2 间隔在两组之间重叠，因此不作判据，只作为证据强度描述。
- 身份：Top-1 人员均值余弦与 Top1-Top2 间隔均可分（命中 0.559-0.770 /
  间隔 0.49-0.71；库外 0.063-0.099 / 间隔 0.0395）。

阈值来自独立校准集，测试集不参与选阈值。本模块不输出「综合概率」，
只输出证据质量分级与可复核的中间量。
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any


STATE_HIGH = "high_confidence"
STATE_INSUFFICIENT = "insufficient_evidence"
STATE_NONE = "no_reliable_candidate"
STATE_UNAVAILABLE = "unavailable"

STATE_LABELS = {
    STATE_HIGH: "高可信候选",
    STATE_INSUFFICIENT: "证据不足，建议人工复核",
    STATE_NONE: "当前库中未发现可靠候选",
    STATE_UNAVAILABLE: "本次无法完成可比对检索",
}

THRESHOLD_SCHEMA = "1.0"

#: 标定阈值配置的环境变量覆盖：用于把另一份标定结果接入同一套代码。
ENV_THRESHOLDS_PATH = "FSTS_OPEN_SET_THRESHOLDS"

#: 安装目录内的正式阈值配置（与 config.json 的 {APP_ROOT} 同层级）。
PACKAGED_THRESHOLDS_RELATIVE = (
    "packaging_v0.4",
    "resources",
    "decision",
    "open_set_thresholds.json",
)

UNCALIBRATED_CAVEAT = (
    "未找到正式标定配置，本次阈值来自内置占位值，结论只能用于内部调试，不得对外出具。"
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def app_root() -> Path:
    """安装根目录：decision/ 的上一层，与 config.json 的 {APP_ROOT} 一致。"""

    return Path(__file__).resolve().parents[1]


def resolve_thresholds_path() -> Path | None:
    """定位正式阈值配置：环境变量优先，其次安装目录内的打包资源。"""

    override = os.environ.get(ENV_THRESHOLDS_PATH)
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate
    packaged = app_root().joinpath(*PACKAGED_THRESHOLDS_RELATIVE)
    if packaged.is_file():
        return packaged
    return None


def _merge_thresholds(payload: Mapping[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(DEFAULT_THRESHOLDS))
    for module, values in _mapping(payload.get("modules")).items():
        if module in merged["modules"] and isinstance(values, Mapping):
            merged["modules"][module].update(dict(values))
    for key in ("source_version", "calibration_note"):
        if isinstance(payload.get(key), str):
            merged[key] = payload[key]
    return merged


@lru_cache(maxsize=8)
def _cached_thresholds(path_text: str, mtime_ns: int) -> dict[str, Any]:
    """按文件修改时间缓存标定配置，避免每次判定都读盘。"""

    return _merge_thresholds(
        json.loads(Path(path_text).read_text(encoding="utf-8"))
    )


def load_thresholds(path: str | Path | None = None) -> dict[str, Any]:
    """读取开放集阈值配置；缺文件时回落到内置默认值。"""

    if path is None:
        return json.loads(json.dumps(DEFAULT_THRESHOLDS))
    candidate = Path(path)
    if not candidate.is_file():
        return json.loads(json.dumps(DEFAULT_THRESHOLDS))
    try:
        stamp = candidate.stat().st_mtime_ns
    except OSError:
        stamp = 0
    return json.loads(json.dumps(_cached_thresholds(str(candidate), stamp)))


def active_thresholds() -> dict[str, Any]:
    """当前生效的阈值：正式标定配置优先，缺失时才用内置占位值。"""

    path = resolve_thresholds_path()
    if path is None:
        return load_thresholds(None)
    return load_thresholds(path)


def _threshold_context(
    thresholds: Mapping[str, Any] | None, module: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """返回 (模块阈值, 溯源信息)；thresholds 为 None 时使用当前生效的正式配置。"""

    if thresholds is None:
        resolved = active_thresholds()
        origin = "packaged-calibration" if resolve_thresholds_path() else "builtin-default"
    else:
        resolved = _mapping(thresholds)
        origin = "explicit"
    config = _mapping(_mapping(resolved.get("modules")).get(module))
    return config, {
        "config_origin": origin,
        "config_version": str(resolved.get("source_version") or "unknown"),
    }


def _verdict(
    *,
    module: str,
    state: str,
    primary: Mapping[str, Any],
    signals: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    reasons: list[str],
    caveats: list[str],
) -> dict[str, Any]:
    return {
        "module": module,
        "state": state,
        "state_label": STATE_LABELS[state],
        "decidable": state in (STATE_HIGH, STATE_INSUFFICIENT, STATE_NONE),
        "primary_signal": dict(primary),
        "signals": dict(signals),
        "thresholds": dict(thresholds),
        "reasons": reasons,
        "caveats": caveats,
    }


# --------------------------------------------------------------------------- #
# 底图检索
# --------------------------------------------------------------------------- #

def decide_source_open_set(
    result: Mapping[str, Any] | None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """底图检索的开放集三态判定。"""

    config, provenance = _threshold_context(thresholds, "source")
    high = _number(config.get("high_global_score"))
    low = _number(config.get("low_global_score"))
    if high is None:
        high = DEFAULT_THRESHOLDS["modules"]["source"]["high_global_score"]
    if low is None:
        low = DEFAULT_THRESHOLDS["modules"]["source"]["low_global_score"]

    data = _mapping(result)
    metrics = _mapping(data.get("metrics"))
    candidates = data.get("candidates")
    threshold_view = {"high_global_score": high, "low_global_score": low, **provenance}
    caveats = [
        "底图检索为闭集排序，任何查询都会返回 Top-K；本判定说明的是该候选能否单独支撑结论。",
        "判定依据是全局相似度，不是「来源确认」；结论仍需与原始文件比对。",
    ]
    if provenance["config_origin"] == "builtin-default":
        caveats.append(UNCALIBRATED_CAVEAT)

    if data.get("status") != "success":
        reason = "本次底图检索未成功完成，没有可判定的候选。"
        return _verdict(
            module="source",
            state=STATE_UNAVAILABLE,
            primary={"name": "global_score", "value": None},
            signals={"library_decoded": _integer(metrics.get("library_decoded"))},
            thresholds=threshold_view,
            reasons=[reason],
            caveats=caveats,
        )

    if not isinstance(candidates, list) or not candidates:
        return _verdict(
            module="source",
            state=STATE_NONE,
            primary={"name": "global_score", "value": None},
            signals={"returned_candidate_count": _integer(metrics.get("returned_candidate_count"))},
            thresholds=threshold_view,
            reasons=["检索已完成，但没有返回任何候选。"],
            caveats=caveats,
        )

    top = _mapping(candidates[0])
    global_score = _number(top.get("global_score"))
    local_score = _number(top.get("local_score"))
    final_score = _number(top.get("final_score"))
    inliers = _integer(top.get("ransac_inlier_count"))
    inlier_ratio = _number(top.get("ransac_inlier_ratio"))
    margin = _number(metrics.get("top1_top2_margin"))
    signals = {
        "global_score": global_score,
        "local_score": local_score,
        "final_score": final_score,
        "ransac_inlier_count": inliers,
        "ransac_inlier_ratio": inlier_ratio,
        "top1_top2_margin": margin,
        "good_match_count": _integer(top.get("good_match_count")),
        "library_decoded": _integer(metrics.get("library_decoded")),
    }

    if global_score is None:
        return _verdict(
            module="source",
            state=STATE_UNAVAILABLE,
            primary={"name": "global_score", "value": None},
            signals=signals,
            thresholds=threshold_view,
            reasons=["结果中缺少全局相似度，无法判定。"],
            caveats=caveats,
        )

    reasons = [
        f"该候选全局相似度为 {global_score:.6f}（高可信阈值 {high:.4f}，拒识阈值 {low:.4f}）。"
    ]
    if inliers is not None:
        reasons.append(f"几何复核内点数为 {inliers}；该量在本系统上不能区分库内外，仅作证据强度描述。")
    if margin is not None:
        reasons.append(
            f"Top1-Top2 间隔为 {margin:.6f}；同源视频多帧场景下该量会接近 0，因此不作为判据。"
        )

    caveats = list(caveats)
    caveats.append("local_score 与 RANSAC 内点在库外样本上也会偏高，不得单独据此认为已找到来源。")

    if global_score >= high:
        state = STATE_HIGH
        reasons.append("全局相似度高于高可信阈值，判为高可信候选。")
    elif global_score >= low:
        state = STATE_INSUFFICIENT
        reasons.append("全局相似度落在不确定带内，证据不足，建议人工复核。")
    else:
        state = STATE_NONE
        reasons.append("全局相似度低于拒识阈值，判为当前库中未发现可靠候选。")

    return _verdict(
        module="source",
        state=state,
        primary={"name": "global_score", "value": global_score},
        signals=signals,
        thresholds=threshold_view,
        reasons=reasons,
        caveats=caveats,
    )


# --------------------------------------------------------------------------- #
# 身份检索
# --------------------------------------------------------------------------- #

def decide_identity_open_set(
    result: Mapping[str, Any] | None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """人脸身份检索的开放集三态判定。"""

    config, provenance = _threshold_context(thresholds, "identity")
    high = _number(config.get("high_similarity"))
    low = _number(config.get("low_similarity"))
    margin_min = _number(config.get("min_top1_top2_margin"))
    if high is None:
        high = DEFAULT_THRESHOLDS["modules"]["identity"]["high_similarity"]
    if low is None:
        low = DEFAULT_THRESHOLDS["modules"]["identity"]["low_similarity"]
    if margin_min is None:
        margin_min = DEFAULT_THRESHOLDS["modules"]["identity"]["min_top1_top2_margin"]

    data = _mapping(result)
    metrics = _mapping(data.get("metrics"))
    candidates = data.get("candidates")
    threshold_view = {
        "high_similarity": high,
        "low_similarity": low,
        "min_top1_top2_margin": margin_min,
        **provenance,
    }
    caveats = [
        "身份检索为闭集排序，任何查询都会返回 Top-K；本判定说明的是该候选能否单独支撑结论。",
        "余弦相似度不是身份概率，高可信候选仍需人工比对确认。",
    ]
    if provenance["config_origin"] == "builtin-default":
        caveats.append(UNCALIBRATED_CAVEAT)

    if data.get("status") != "success":
        face_count = _integer(metrics.get("query_face_count"))
        if face_count is not None and face_count != 1:
            reason = (
                f"查询图检出人脸数为 {face_count}，身份检索要求恰好一张人脸，本次未能完成比对。"
            )
        else:
            reason = "本次身份检索未成功完成，没有可判定的候选。"
        return _verdict(
            module="identity",
            state=STATE_UNAVAILABLE,
            primary={"name": "mean_similarity", "value": None},
            signals={
                "query_face_count": face_count,
                "face_detection_score": _number(metrics.get("face_detection_score")),
                "valid_embedding_count": _integer(metrics.get("valid_embedding_count")),
                "library_identity_count": _integer(metrics.get("library_identity_count")),
            },
            thresholds=threshold_view,
            reasons=[reason],
            caveats=caveats + ["检索未完成时不能给出「库中无此人」的结论，只能说明本次无法比对。"],
        )

    library_identity_count = _integer(metrics.get("library_identity_count"))
    valid_embedding_count = _integer(metrics.get("valid_embedding_count"))
    if library_identity_count == 0 or valid_embedding_count == 0:
        return _verdict(
            module="identity",
            state=STATE_UNAVAILABLE,
            primary={"name": "mean_similarity", "value": None},
            signals={
                "library_identity_count": library_identity_count,
                "valid_embedding_count": valid_embedding_count,
            },
            thresholds=threshold_view,
            reasons=["身份库中没有形成任何有效特征，检索结果没有判别意义。"],
            caveats=caveats + ["库覆盖不足时，低分不能解释为「库中无此人」。"],
        )

    if not isinstance(candidates, list) or not candidates:
        return _verdict(
            module="identity",
            state=STATE_NONE,
            primary={"name": "mean_similarity", "value": None},
            signals={
                "library_identity_count": library_identity_count,
                "valid_embedding_count": valid_embedding_count,
            },
            thresholds=threshold_view,
            reasons=["检索已完成，但没有返回任何候选。"],
            caveats=caveats,
        )

    top = _mapping(candidates[0])
    score = _number(top.get("mean_similarity"))
    if score is None:
        score = _number(top.get("score"))
    best_reference = _number(top.get("best_reference_score"))
    margin = _number(metrics.get("top1_top2_margin"))
    signals = {
        "mean_similarity": score,
        "best_reference_score": best_reference,
        "top1_top2_margin": margin,
        "valid_reference_count": _integer(top.get("valid_reference_count")),
        "library_identity_count": library_identity_count,
        "valid_embedding_count": valid_embedding_count,
        "query_face_count": _integer(metrics.get("query_face_count")),
        "face_detection_score": _number(metrics.get("face_detection_score")),
    }

    if score is None:
        return _verdict(
            module="identity",
            state=STATE_UNAVAILABLE,
            primary={"name": "mean_similarity", "value": None},
            signals=signals,
            thresholds=threshold_view,
            reasons=["结果中缺少候选相似度，无法判定。"],
            caveats=caveats,
        )

    reasons = [
        f"第一候选人员均值为 {score:.6f}（高可信阈值 {high:.4f}，拒识阈值 {low:.4f}）。"
    ]
    if margin is not None:
        reasons.append(
            f"Top1-Top2 间隔为 {margin:.6f}（最低要求 {margin_min:.4f}）；"
            "间隔过小说明库内存在无法区分的多个候选。"
        )

    if score >= high and (margin is None or margin >= margin_min):
        state = STATE_HIGH
        reasons.append("相似度与间隔同时达标，判为高可信候选。")
    elif score < low:
        state = STATE_NONE
        reasons.append("相似度低于拒识阈值，判为当前库中未发现可靠候选。")
    else:
        state = STATE_INSUFFICIENT
        if score >= high and margin is not None and margin < margin_min:
            reasons.append("相似度达标但 Top1-Top2 间隔过小，降级为需人工复核。")
        else:
            reasons.append("相似度落在不确定带内，证据不足，建议人工复核。")

    return _verdict(
        module="identity",
        state=state,
        primary={"name": "mean_similarity", "value": score},
        signals=signals,
        thresholds=threshold_view,
        reasons=reasons,
        caveats=caveats,
    )


# 内置占位阈值：只在正式标定配置缺失时兜底。正式结论必须来自
# packaging_v0.4/resources/decision/open_set_thresholds.json（由标定脚本生成），
# 此时判定会额外给出「阈值未标定」的限制声明。
DEFAULT_THRESHOLDS: dict[str, Any] = {
    "schema_version": THRESHOLD_SCHEMA,
    "source_version": "uncalibrated-default",
    "calibration_note": "内置占位阈值，必须由标定脚本写入正式配置后使用。",
    "modules": {
        "source": {"high_global_score": 0.9450, "low_global_score": 0.5707},
        "identity": {
            "high_similarity": 0.5440,
            "low_similarity": 0.1480,
            "min_top1_top2_margin": 0.1000,
        },
    },
}


__all__ = [
    "STATE_HIGH",
    "STATE_INSUFFICIENT",
    "STATE_NONE",
    "STATE_UNAVAILABLE",
    "STATE_LABELS",
    "DEFAULT_THRESHOLDS",
    "ENV_THRESHOLDS_PATH",
    "active_thresholds",
    "app_root",
    "decide_identity_open_set",
    "decide_source_open_set",
    "load_thresholds",
    "resolve_thresholds_path",
]
