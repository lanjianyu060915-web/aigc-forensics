# -*- coding: utf-8 -*-
"""视频抽帧/选帧的确定性规则层（纯逻辑：不导入 OpenCV、不加载模型、不联网）。

职责边界：

- 解码与逐帧客观指标由 ``modules/video_frames_v2/run_video_frames_v2.py`` 产出；
- 本模块只做「指标 -> 质量分 -> 关键帧集合」的计算，便于单元测试与事后审计。

三条硬性设计约束：

1. 质量分是**工程启发式**，只用于排序与选帧，不参与任何检测或检索判定。
   结果 JSON 必须同时暴露分量，人工可以推翻排序（见 ``quality_components``）。
2. 身份检索要求「恰好一张人脸」，所以单人脸帧优先；这是硬约束偏好，不是
   质量修饰，且必须写进选择理由。
3. 所有函数都是确定性的：没有随机数，同一输入必然得到同一输出。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


SAMPLE_STRATEGY = "uniform_trimmed"
DEFAULT_TRIM_RATIO = 0.05
LEGACY_SELECTION_STRATEGY = "legacy_quality_v1"
EXPERIMENTAL_SELECTION_STRATEGY = "segmented_time_v1"

# 质量分权重：有人脸信息与未启用人脸检测两种情形分开，各自保持归一。
DEFAULT_WEIGHTS: dict[str, float] = {"face": 0.55, "sharpness": 0.30, "brightness": 0.15}
NO_FACE_WEIGHTS: dict[str, float] = {"face": 0.0, "sharpness": 0.65, "brightness": 0.35}

FACE_SCORE_SINGLE = 1.0
FACE_SCORE_MULTIPLE = 0.55
FACE_SCORE_NONE = 0.0
FACE_SCORE_UNKNOWN = 0.7

SELECTION_RULE = (
    "按质量分降序贪心选帧；单人脸帧优先；已选帧之间保持最小时间间隔；"
    "时间覆盖与画质冲突时以画质优先（低质量帧对三个模块都是噪声）。"
)
TIER_LABELS = (
    "单人脸且清晰度达标",
    "单人脸（清晰度不足）",
    "非单人脸且清晰度达标",
    "非单人脸（清晰度不足）",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def percentile(values: Sequence[float], q: float) -> float:
    """线性插值分位数，口径与 numpy.percentile 默认一致（避免引入依赖）。"""

    data = sorted(float(value) for value in values if _number(value) is not None)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * max(0.0, min(100.0, float(q))) / 100.0
    lower = int(math.floor(position))
    upper = min(lower + 1, len(data) - 1)
    fraction = position - lower
    return data[lower] * (1.0 - fraction) + data[upper] * fraction


def candidate_indices(
    frame_count: int,
    candidate_count: int,
    *,
    trim_ratio: float = DEFAULT_TRIM_RATIO,
) -> list[int]:
    """在去掉首尾 ``trim_ratio`` 的时间轴上均匀取 ``candidate_count`` 帧。

    首尾裁剪用于避开片头片尾黑场/转场；均匀取样不保证发现短暂变化，
    不用随机数，重跑得到完全相同的帧号。
    """

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("视频总帧数必须为正整数")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count <= 0:
        raise ValueError("候选帧数量必须为正整数")
    if candidate_count >= frame_count:
        return list(range(frame_count))
    trim = int(frame_count * float(trim_ratio))
    start = min(max(trim, 0), frame_count - 1)
    end = max(frame_count - 1 - trim, start)
    if candidate_count == 1:
        return [(start + end) // 2]
    span = end - start
    indices: list[int] = []
    for position in range(candidate_count):
        index = start + int(round(span * position / (candidate_count - 1)))
        if not indices or index != indices[-1]:
            indices.append(int(min(max(index, 0), frame_count - 1)))
    return indices


def _weights_for(candidates: Sequence[Mapping[str, Any]], weights: Mapping[str, float] | None) -> dict[str, float]:
    if weights is not None:
        return {key: float(value) for key, value in weights.items()}
    detector_used = any(item.get("face_count") is not None for item in candidates)
    return dict(DEFAULT_WEIGHTS if detector_used else NO_FACE_WEIGHTS)


def quality_components(
    candidate: Mapping[str, Any],
    *,
    sharpness_reference: float,
    weights: Mapping[str, float],
) -> dict[str, float]:
    """把单帧的客观指标折算成质量分与可复核的分量。"""

    sharpness = _number(candidate.get("sharpness")) or 0.0
    sharpness = max(0.0, sharpness)
    sharpness_ratio = (
        min(1.0, sharpness / float(sharpness_reference)) if float(sharpness_reference) > 0 else 0.0
    )
    brightness = _number(candidate.get("brightness"))
    brightness_score = (
        0.0 if brightness is None else max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
    )
    count = candidate.get("face_count")
    if count is None:
        face_score = FACE_SCORE_UNKNOWN
    elif int(count) == 1:
        face_score = FACE_SCORE_SINGLE
    elif int(count) >= 2:
        face_score = FACE_SCORE_MULTIPLE
    else:
        face_score = FACE_SCORE_NONE
    quality = (
        float(weights.get("face", 0.0)) * face_score
        + float(weights.get("sharpness", 0.0)) * sharpness_ratio
        + float(weights.get("brightness", 0.0)) * brightness_score
    )
    return {
        "quality_score": quality,
        "face_score": face_score,
        "sharpness_ratio": sharpness_ratio,
        "brightness_score": brightness_score,
    }


def score_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """为每个候选帧计算质量分；返回值与输入等长、顺序一致。"""

    values = [max(0.0, _number(item.get("sharpness")) or 0.0) for item in candidates]
    reference = percentile(values, 90.0)
    active = _weights_for(candidates, weights)
    scored: list[dict[str, Any]] = []
    for item in candidates:
        components = quality_components(item, sharpness_reference=reference, weights=active)
        scored.append(
            {
                "candidate": dict(item),
                "quality_score": components["quality_score"],
                "quality_components": {
                    "face_score": components["face_score"],
                    "sharpness_ratio": components["sharpness_ratio"],
                    "brightness_score": components["brightness_score"],
                },
                "quality_weights": dict(active),
                "sharpness_reference": reference,
            }
        )
    return scored


def _tier_of(candidate: Mapping[str, Any], scored: Mapping[str, Any], *, prefer_single_face: bool, min_sharpness_ratio: float) -> int:
    single = candidate.get("face_count") == 1
    sharpness_ok = scored["quality_components"]["sharpness_ratio"] >= float(min_sharpness_ratio)
    if prefer_single_face:
        if single and sharpness_ok:
            return 0
        if single:
            return 1
        return 2 if sharpness_ok else 3
    return 2 if sharpness_ok else 3


def select_key_frames(
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_frames: int,
    duration_seconds: float | None = None,
    min_gap_factor: float = 0.5,
    min_sharpness_ratio: float = 0.35,
    prefer_single_face: bool = True,
) -> dict[str, Any]:
    """选出用于三个模块的关键帧（确定性贪心，带时间间隔约束）。

    返回 ``policy`` / ``key_frames`` / ``candidates`` 三段：policy 说明规则与
    阈值，candidates 保留全部候选的质量分与入选状态，便于人工复核为什么
    某一帧被选中或被跳过。
    """

    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
        raise ValueError("关键帧数量必须为正整数")
    if not candidates:
        return {
            "policy": {
                "rule": SELECTION_RULE,
                "max_key_frames": max_frames,
                "min_temporal_gap_seconds": 0.0,
                "min_sharpness_ratio": float(min_sharpness_ratio),
                "prefer_single_face": bool(prefer_single_face),
                "weights": {},
                "sharpness_reference": 0.0,
                "empty_reason": "没有候选帧",
            },
            "key_frames": [],
            "candidates": [],
        }

    scored = score_candidates(candidates)
    timestamps = [float(item.get("timestamp_seconds") or 0.0) for item in candidates]
    duration = _number(duration_seconds)
    base_gap = (
        float(duration) / (max_frames * 2.0)
        if duration is not None and duration > 0 and max_frames > 1
        else 0.0
    )
    base_gap = base_gap * float(min_gap_factor)

    tiers: list[list[int]] = [[], [], [], []]
    for order, (candidate, score) in enumerate(zip(candidates, scored)):
        tiers[
            _tier_of(
                candidate,
                score,
                prefer_single_face=prefer_single_face,
                min_sharpness_ratio=min_sharpness_ratio,
            )
        ].append(order)
    for tier in tiers:
        tier.sort(key=lambda order: (-scored[order]["quality_score"], order))

    accepted: list[int] = []

    def fill(pool: Sequence[int], gap: float) -> None:
        for order in pool:
            if len(accepted) >= max_frames:
                return
            if order in accepted:
                continue
            if gap > 0 and any(
                abs(timestamps[order] - timestamps[other]) < gap for other in accepted
            ):
                continue
            accepted.append(order)

    def fill_all(gap: float) -> None:
        for tier in tiers:
            fill(tier, gap)

    fill_all(base_gap)
    target = min(max_frames, len(candidates))
    if len(accepted) < target:
        fill_all(base_gap / 2.0)
    if len(accepted) < target:
        fill_all(0.0)

    accepted.sort(key=lambda order: (scored[order]["quality_score"], -order), reverse=True)
    rank_of = {order: rank for rank, order in enumerate(accepted, start=1)}

    output_candidates: list[dict[str, Any]] = []
    key_frames: list[dict[str, Any]] = []
    for order, (candidate, score) in enumerate(zip(candidates, scored)):
        entry = dict(candidate)
        entry["quality_score"] = score["quality_score"]
        entry["quality_components"] = score["quality_components"]
        rank = rank_of.get(order)
        entry["selected"] = rank is not None
        entry["selection_rank"] = rank
        reasons: list[str] = []
        if candidate.get("face_count") == 1:
            reasons.append("单人脸帧优先（身份检索要求恰好一张人脸）")
        elif candidate.get("face_count") is not None and int(candidate.get("face_count") or 0) > 1:
            reasons.append(f"检出 {int(candidate['face_count'])} 张人脸，身份检索会跳过")
        elif candidate.get("face_count") == 0:
            reasons.append("未检出人脸")
        ratio = score["quality_components"]["sharpness_ratio"]
        reasons.append(f"清晰度为该视频 p90 的 {ratio:.2f} 倍")
        brightness = _number(candidate.get("brightness"))
        if brightness is not None:
            reasons.append(f"亮度均值 {brightness:.1f}")
        if rank is not None:
            nearest = min(
                (abs(timestamps[order] - timestamps[other]) for other in accepted if other != order),
                default=None,
            )
            if nearest is not None:
                reasons.append(f"与已选帧的最小时间间隔 {nearest:.2f}s")
        entry["selection_reasons"] = reasons
        output_candidates.append(entry)
        if rank is not None:
            key_frames.append({**entry, "frame_rank": rank})
    key_frames.sort(key=lambda item: item["frame_rank"])

    return {
        "policy": {
            "rule": SELECTION_RULE,
            "max_key_frames": max_frames,
            "selected_frames": len(key_frames),
            "min_temporal_gap_seconds": round(base_gap, 3),
            "min_sharpness_ratio": float(min_sharpness_ratio),
            "prefer_single_face": bool(prefer_single_face),
            "weights": dict(scored[0]["quality_weights"]) if scored else {},
            "sharpness_reference": round(float(scored[0]["sharpness_reference"]), 3) if scored else 0.0,
            "tiers": list(TIER_LABELS),
        },
        "key_frames": key_frames,
        "candidates": output_candidates,
    }


def summarize_frame_coverage(
    frames: Sequence[Mapping[str, Any]],
    *,
    frame_count: int | None = None,
    fps: float | None = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Describe discrete observations; never equate their span with inspected video.

    Coordinates are the requested frame indices. Decoder position observations are
    separate evidence and are not upgraded to verified presentation timestamps.
    """
    indices = sorted({
        int(item["frame_index"]) for item in frames
        if isinstance(item.get("frame_index"), int)
        and not isinstance(item.get("frame_index"), bool) and item["frame_index"] >= 0
    })
    times = sorted({value for item in frames
                    if (value := _number(item.get("timestamp_seconds"))) is not None and value >= 0})
    total = frame_count if isinstance(frame_count, int) and not isinstance(frame_count, bool) and frame_count > 0 else None
    nominal_fps = _number(fps)
    duration = _number(duration_seconds)
    gaps = [b - a for a, b in zip(indices, indices[1:])]
    time_gaps = [b - a for a, b in zip(times, times[1:])]
    end = total if total is not None else None
    edge_ranges = None
    if total is not None:
        edge_ranges = ({"leading": [0, min(indices[0], total)],
                        "trailing": [min(indices[-1] + 1, total), end]}
                       if indices else {"leading": [0, total], "trailing": None})
    return {
        "scope": "discrete_frame_samples",
        "continuous_time_coverage": False,
        "coordinate_semantics": "requested_frame_indices; backend positions are separate unverified observations",
        "entries": len(frames),
        "frame_ids": [item.get("frame_id") for item in frames],
        "unique_requested_frame_count": len(indices),
        "requested_frame_indices": indices,
        "video_frame_count": total,
        "timestamp_sources": sorted({str(item.get("timestamp_source") or "unspecified") for item in frames}),
        "timestamp_seconds": times,
        "frame_index_gaps": gaps,
        "max_frame_index_gap": max(gaps) if gaps else None,
        "timestamp_gaps_seconds": [round(value, 6) for value in time_gaps],
        "max_timestamp_gap_seconds": round(max(time_gaps), 6) if time_gaps else None,
        "edge_unsampled_frame_ranges_half_open": edge_ranges,
        "nominal_fps": nominal_fps if nominal_fps is not None and nominal_fps > 0 else None,
        "duration_seconds": duration if duration is not None and duration >= 0 else None,
        "limitations": ["Samples and their first-to-last span do not prove continuous inspection or absence of transient manipulation."],
    }


def select_segmented_key_frames(
    candidates: Sequence[Mapping[str, Any]], *, max_frames: int,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Opt-in research candidate: one quality-ranked sample per temporal bin.

    This is not used by the legacy selector and has no claimed detection benefit.
    Empty bins are recorded; remaining budget is filled by quality deterministically.
    """
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
        raise ValueError("关键帧数量必须为正整数")
    policy: dict[str, Any] = {
        "strategy": EXPERIMENTAL_SELECTION_STRATEGY, "experimental": True,
        "independent_effect_validation": "not_performed",
        "rule": "时间分桶；每桶按既有质量分选择一帧；空桶记录；剩余预算按质量补齐。",
        "max_key_frames": max_frames, "selected_frames": 0,
    }
    if not candidates:
        return {"policy": {**policy, "empty_reason": "没有候选帧"}, "key_frames": [], "candidates": []}
    times = [_number(item.get("timestamp_seconds")) for item in candidates]
    if any(value is None or value < 0 for value in times):
        raise ValueError("分段候选需要每帧有效且非负的时间坐标；不得以未知时间冒充覆盖")
    times = [float(value) for value in times]
    scored = score_candidates(candidates)
    bins_count = min(max_frames, len(candidates))
    duration = _number(duration_seconds)
    end = max(max(times), duration if duration is not None and duration > 0 else 0.0)
    bins: list[list[int]] = [[] for _ in range(bins_count)]
    for order, timestamp in enumerate(times):
        bucket = min(bins_count - 1, int(timestamp / end * bins_count)) if end > 0 else 0
        bins[bucket].append(order)
    def quality_order(order: int):
        frame_index = candidates[order].get("frame_index")
        return (-scored[order]["quality_score"], times[order],
                frame_index if isinstance(frame_index, int) else order, order)
    accepted = [min(pool, key=quality_order) for pool in bins if pool]
    remaining = sorted((order for order in range(len(candidates)) if order not in accepted), key=quality_order)
    accepted.extend(remaining[:bins_count - len(accepted)])
    accepted.sort(key=lambda order: (times[order], quality_order(order)))
    ranks = {order: rank for rank, order in enumerate(accepted, 1)}
    output: list[dict[str, Any]] = []
    for order, candidate in enumerate(candidates):
        output.append({**candidate, "quality_score": scored[order]["quality_score"],
                       "quality_components": scored[order]["quality_components"],
                       "selected": order in ranks, "selection_rank": ranks.get(order),
                       "selection_reasons": ["独立候选：时间分桶后按固定质量排序；尚未验证鉴伪效果。"]})
    selected = [{**output[order], "frame_rank": ranks[order]} for order in accepted]
    policy.update(selected_frames=len(selected), temporal_bins=bins_count,
                  empty_temporal_bins=[index for index, pool in enumerate(bins) if not pool],
                  temporal_domain_seconds=[0.0, end],
                  weights=dict(scored[0]["quality_weights"]))
    return {"policy": policy, "key_frames": selected, "candidates": output}


__all__ = [
    "DEFAULT_TRIM_RATIO",
    "DEFAULT_WEIGHTS",
    "NO_FACE_WEIGHTS",
    "SAMPLE_STRATEGY",
    "SELECTION_RULE",
    "TIER_LABELS",
    "candidate_indices",
    "percentile",
    "quality_components",
    "score_candidates",
    "select_key_frames",
    "summarize_frame_coverage",
    "select_segmented_key_frames",
    "LEGACY_SELECTION_STRATEGY",
    "EXPERIMENTAL_SELECTION_STRATEGY",
]

