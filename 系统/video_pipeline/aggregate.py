# -*- coding: utf-8 -*-
"""视频任务的多帧聚合层（纯逻辑，可单元测试）。

三个模块在视频模式下的聚合口径（全部写进结果 JSON，避免黑箱）：

- 鉴伪：帧级分数取**中位数**得到视频级分数；视频级分带由帧级分带的
  一致性规则给出（``band_majority``）。单位换脸视频通常整段被篡改，
  要求「多数帧落在高风险带」才判高风险，是为了压低单帧偶发高分造成的
  误报——M3.1 已证明强退化输入下真人单帧误报率可达 0.56。
- 身份：按人员聚合，取该人员在「有脸帧」上的平均余弦相似度排序，并公开
  支持帧数与逐帧分数；参考图级字段取自得分最高的那一帧。
- 底图：只跑一帧（最可靠帧）。底图检索在 CPU 上昂贵，逐帧跑没有收益，
  而同源视频的相邻帧近乎同图（M1 §2.3）。

本层不产生「综合可信度」，不跨模块做数值相加。
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any


DEEPFAKE_RULE = "band_majority"
# 有独立视频校准集时的口径：视频级分数（帧级中位数）直接与校准集真人视频分布的
# 分位点比较，不再依赖帧级分带计数。只有当前输入上全部帧都由二代分支判定、且内核
# 给出了成对阈值时才启用，否则一律退回 ``band_majority``。
CALIBRATED_VIDEO_RULE = "median_video_thresholds"
IDENTITY_RULE = "mean_across_frames"
SOURCE_RULE = "primary_frame"

# 任务级与鉴伪专用的边界说明：视频级分带与二次编码衰减只对鉴伪成立，
# 身份与底图共用同一套帧，只能继承与模块无关的那一条。
VIDEO_LIMITATIONS_DEEPFAKE = [
    "视频级分带边界来自图像标定集，没有可用的视频级标定：M4.1 在 DFDC 抽样的真人视频上测得本口径的中/高风险误报率为 53.3%，跨语料的绝对阈值不可移植。",
    "同一 M4.1 实验中，同一原视频内换脸版分数更高的配对比例为 70.9%（79 对，符号检验 p=1.3e-04）：检测器对换脸本身敏感，但视频间方差会淹没该信号，分带只能当作筛查提示。",
    "帧级分带沿用整图标定集边界；M3.1 已证明视频二次编码近似会使留出集 AUC 降到 0.8175，视频输入必须视为更弱证据。",
]
VIDEO_LIMITATIONS_SHARED = [
    "关键帧是有损抽样：未选中的帧没有参与任何模块，报告中的帧覆盖情况以 frame_selection.json 为准。",
]
VIDEO_LIMITATIONS = [*VIDEO_LIMITATIONS_DEEPFAKE, *VIDEO_LIMITATIONS_SHARED]

DEFAULT_BAND_LABELS = {"low": "证据不足", "medium": "需人工复核", "high": "高风险提示"}
DEFAULT_BAND_REASONS = {
    "high": "多数关键帧的深伪风险分落在校准集真人分布上界之外，提示高风险，仍需人工复核。",
    "medium": "关键帧中存在高风险分帧或较多样本落在高危尾部，证据不足以直接判定，建议人工复核。",
    "low": "关键帧分数与已知真人分布无显著差异，当前证据不足以支持深伪判断。",
}
# 校准口径下视频级分数是帧级中位数，措辞必须跟着换：说「关键帧分数」会让人误以为
# 结论来自某一帧。
CALIBRATED_VIDEO_BAND_REASONS = {
    "high": "视频级中位分高于独立视频校准集真人视频分布的 p99，提示高风险，仍需人工复核。",
    "medium": "视频级中位分落在独立视频校准集真人视频分布的 p95~p99 之间，证据不足以直接判定，建议人工复核。",
    "low": "视频级中位分不高于独立视频校准集真人视频分布的 p95，当前证据不足以支持深伪判断。",
}

# 冻结内核（v0.4.0 空间 + 频域加权融合）没有校准分带，只能按阈值给出二值判定，
# 因此视频级结论必须换一套措辞，避免把「没有超过阈值」写成「与真人分布无显著差异」。
FROZEN_BAND_REASONS = {
    "high": (
        "存在关键帧被冻结内核判定为深伪风险（冻结阈值二值判定，无校准分带），提示高风险，仍需人工复核。"
    ),
    "medium": (
        "冻结内核的逐帧判定不一致（部分帧超过判定阈值、部分帧未超过），证据不足以直接判定，建议人工复核。"
    ),
    "low": (
        "全部关键帧都没有超过冻结阈值；但冻结内核在整图换脸场景的判别力接近随机"
        "（M3：SWAN-DF AUC 0.4594），该结果只能记为「当前证据不足」，不能作为未换脸的结论。"
    ),
}

FROZEN_BRANCH_WARNING = (
    "本轮有 {count}/{total} 个关键帧由冻结内核（v0.4.0 空间 + 频域融合）给出判定："
    "冻结结果没有校准分带，视频级分带退化为冻结阈值的二值判定；"
    "该内核在整图换脸场景接近随机水平，本次视频级鉴伪结论只能视为占位结果。"
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def _first_number(*values: Any) -> float | None:
    """返回第一个可解释为有限数值的参数；用于兼容不同内核的字段名。"""

    for value in values:
        number = _number(value)
        if number is not None:
            return number
    return None


def _candidate_name(candidate: Mapping[str, Any]) -> str | None:
    """取候选的可读标识：优先显式 ID，其次库内文件名（不外泄库绝对路径）。"""

    for key in ("candidate_id", "source_id", "file_name", "identity_id"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("file_path", "reference_image", "reference_relative_path"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.replace("\\", "/").rstrip("/").split("/")[-1]
    return None


def _frame_meta(entry: Mapping[str, Any]) -> dict[str, Any]:
    frame = entry.get("frame")
    return dict(frame) if isinstance(frame, Mapping) else {}


def _frame_label(frame: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frame_rank": frame.get("frame_rank"),
        "file_name": frame.get("file_name"),
        "timestamp_seconds": frame.get("timestamp_seconds"),
        "frame_index": frame.get("frame_index"),
    }


def _looks_absolute(value: str) -> bool:
    """识别 Windows 盘符或 UNC 绝对路径；这类路径必须原样保留。"""

    if len(value) >= 2 and value[1] == ":" and value[0].isalpha():
        return True
    return value.startswith("//")


def rebase_frame_artifacts(
    payload: Mapping[str, Any],
    *,
    frame_dir: Any,
    prefix: str,
) -> dict[str, Any]:
    """把单帧结果里指向该帧输出目录的相对路径改写成任务目录相对路径。

    只改写「在该帧输出目录下确实存在对应文件」的字符串，避免误改库内相对
    路径（例如参考图路径）。
    """

    from pathlib import Path

    root = Path(frame_dir)
    normalized_prefix = str(prefix).replace("\\", "/").strip("/")

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            candidate = value.replace("\\", "/").strip()
            if (
                not candidate
                or candidate.startswith("/")
                or ".." in candidate.split("/")
                or _looks_absolute(candidate)
            ):
                return value
            if normalized_prefix and candidate.startswith(normalized_prefix + "/"):
                return value
            try:
                exists = (root / candidate).is_file()
            except OSError:
                exists = False
            return f"{normalized_prefix}/{candidate}" if exists else value
        if isinstance(value, Mapping):
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return walk(dict(payload))


def _frame_band(
    payload: Mapping[str, Any], details: Mapping[str, Any]
) -> tuple[str, str, bool | None]:
    """返回该帧的 ``(band, band_source, is_deepfake)``。

    二代内核会在 ``details`` 里给出校准分带；冻结内核没有分带，只能退化为
    冻结阈值的二值判定。分带来源必须写进结果，避免两种口径被混为一谈。
    """

    band = str(details.get("band") or "").strip().lower()
    if band in {"high", "medium", "low"}:
        return band, "calibrated_band", band == "high"
    decision = payload.get("is_deepfake")
    if isinstance(decision, bool):
        return ("high" if decision else "low"), "frozen_threshold", decision
    return "", "unavailable", None


def _frame_branch(payload: Mapping[str, Any], details: Mapping[str, Any]) -> str:
    """取该帧实际使用的检测分支名；冻结内核没有分支名时给出稳定标识。"""

    method = _mapping(payload.get("method"))
    for value in (details.get("routed_branch"), details.get("branch"), method.get("branch")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    if details.get("native_status") is not None:
        return "frozen_v040_spatial_frequency_fusion"
    return ""


def aggregate_deepfake_frames(
    frame_results: Sequence[Mapping[str, Any]],
    *,
    query_image: str,
    band_labels: Mapping[str, str] | None = None,
    extra_limitations: Sequence[str] = (),
    video_band_thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """把逐帧鉴伪结果聚合成一份视频级结果（返回 canonical payload）。"""

    labels = dict(DEFAULT_BAND_LABELS)
    if isinstance(band_labels, Mapping):
        labels.update({str(key): str(value) for key, value in band_labels.items()})

    scored: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for entry in frame_results:
        frame = _frame_meta(entry)
        payload = entry.get("payload")
        payload = dict(payload) if isinstance(payload, Mapping) else None
        score = _number((payload or {}).get("score"))
        status = str((payload or {}).get("status") or "error")
        if payload is None or status != "success" or score is None:
            failures.append(
                {
                    **_frame_label(frame),
                    "error_message": str(
                        (payload or {}).get("error_message") or entry.get("error") or "未返回结果"
                    ),
                }
            )
            continue
        details = _mapping(payload.get("details"))
        band, band_source, is_deepfake = _frame_band(payload, details)
        scored.append(
            {
                "frame": frame,
                "payload": payload,
                "score": float(score),
                "band": band,
                "band_label": details.get("band_label"),
                "band_source": band_source,
                "is_deepfake": is_deepfake,
                "branch": _frame_branch(payload, details),
                "thresholds": _mapping(details.get("band_thresholds")),
                "input_sha256": payload.get("provenance", {}).get("input_sha256")
                if isinstance(payload.get("provenance"), Mapping)
                else None,
                "runtime_seconds": _number(payload.get("runtime_seconds")) or 0.0,
                "limitations": list(payload.get("limitations") or []),
            }
        )

    runtime_seconds = sum(item["runtime_seconds"] for item in scored) + sum(
        _number(entry.get("runtime_seconds")) or 0.0 for entry in frame_results if entry.get("payload") is None
    )

    if not scored:
        payload = {
            "schema_version": "3.0",
            "module": "deepfake",
            "status": "error",
            "query_image": str(query_image),
            "is_deepfake": None,
            "score": None,
            "label": None,
            "runtime_seconds": runtime_seconds,
            "details": {
                "branch": "vit_full_image_video",
                "video_aggregation": {"rule": DEEPFAKE_RULE, "frames_scored": 0},
                "frame_failures": failures,
            },
            "metrics": {"frames_scored": 0, "frames_total": len(frame_results)},
            "explanation": {},
            "visual_evidence": {},
            "artifact_manifest": [],
            "progress_events": [],
            "warnings": [],
            "limitations": list(VIDEO_LIMITATIONS),
            "runtime": {"elapsed_seconds": runtime_seconds},
            "provenance": {},
            "error_message": "全部关键帧的深伪鉴伪均未成功，视频级结论不可用。",
            "method": {
                "name": "二代整图深伪风险甄别（视频多帧聚合）",
                "branch": "vit_full_image_video",
                "fusion_method": "帧级分数取中位数；分带按帧级分带一致性规则",
            },
        }
        return payload

    scores = [item["score"] for item in scored]
    frames_scored = len(scores)
    frames_total = len(frame_results)
    median_score = float(statistics.median(scores))
    band_counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    band_sources = {"calibrated_band": 0, "frozen_threshold": 0, "unavailable": 0}
    branch_counts: dict[str, int] = {}
    for item in scored:
        band_counts[item["band"] if item["band"] in band_counts else "unknown"] += 1
        source_key = str(item["band_source"] or "unavailable")
        band_sources[source_key if source_key in band_sources else "unavailable"] += 1
        branch_key = str(item["branch"] or "unknown")
        branch_counts[branch_key] = branch_counts.get(branch_key, 0) + 1
    frozen_frames = band_sources["frozen_threshold"]
    band_rule = "frozen_threshold" if frozen_frames else "calibrated_band"
    if frozen_frames and frozen_frames < frames_scored:
        band_rule = "mixed"

    video_thresholds = _mapping(video_band_thresholds)
    video_low_max = _number(video_thresholds.get("low_max"))
    video_medium_max = _number(video_thresholds.get("medium_max"))
    if (
        frozen_frames == 0
        and video_low_max is not None
        and video_medium_max is not None
        and video_medium_max > video_low_max
    ):
        video_rule = CALIBRATED_VIDEO_RULE
        if median_score > video_medium_max:
            band_key = "high"
        elif median_score > video_low_max:
            band_key = "medium"
        else:
            band_key = "low"
    else:
        video_rule = DEEPFAKE_RULE
        video_low_max = None
        video_medium_max = None
        if band_counts["high"] * 2 > frames_scored:
            band_key = "high"
        elif band_counts["high"] >= 1 or band_counts["medium"] * 2 >= frames_scored:
            band_key = "medium"
        else:
            band_key = "low"

    per_frame = [
        {
            **_frame_label(item["frame"]),
            "score": round(item["score"], 6),
            "band": item["band"] or None,
            "band_label": item["band_label"] or labels.get(item["band"] or "", None),
            "band_source": item["band_source"],
            "branch": item["branch"] or None,
            "is_deepfake": item["is_deepfake"],
            "input_sha256": item["input_sha256"],
            "runtime_seconds": round(item["runtime_seconds"], 4),
        }
        for item in scored
    ]
    thresholds = scored[0]["thresholds"]
    high_frames = band_counts["high"]
    medium_frames = band_counts["medium"]
    low_frames = band_counts["low"]
    if frozen_frames >= frames_scored:
        band_reason = FROZEN_BAND_REASONS[band_key]
    elif video_rule == CALIBRATED_VIDEO_RULE:
        band_reason = CALIBRATED_VIDEO_BAND_REASONS[band_key]
    else:
        band_reason = DEFAULT_BAND_REASONS[band_key]
    if frozen_frames >= frames_scored:
        consistency = (
            f"共 {frames_scored}/{frames_total} 个关键帧完成鉴伪（冻结内核二值判定）："
            f"超过冻结阈值 {high_frames} 帧、未超过 {low_frames} 帧"
            + (f"、失败 {len(failures)} 帧" if failures else "")
            + f"；中位分 {median_score:.6f}。"
        )
    elif video_rule == CALIBRATED_VIDEO_RULE:
        consistency = (
            f"共 {frames_scored}/{frames_total} 个关键帧完成鉴伪："
            f"高风险 {high_frames} 帧、需复核 {medium_frames} 帧、证据不足 {low_frames} 帧"
            + (f"、失败 {len(failures)} 帧" if failures else "")
            + f"；中位分 {median_score:.6f}，对照独立视频校准集真人视频分位点"
            f"（p95={video_low_max:.6g}、p99={video_medium_max:.6g}）后落在"
            f"「{labels.get(band_key, band_key)}」带。"
        )
    else:
        consistency = (
            f"共 {frames_scored}/{frames_total} 个关键帧完成鉴伪："
            f"高风险 {high_frames} 帧、需复核 {medium_frames} 帧、证据不足 {low_frames} 帧"
            + (f"、失败 {len(failures)} 帧" if failures else "")
            + f"；中位分 {median_score:.6f}。"
        )
    limitations: list[str] = []
    for text in [*VIDEO_LIMITATIONS, *extra_limitations]:
        if isinstance(text, str) and text.strip() and text not in limitations:
            limitations.append(text)
    for item in scored:
        for text in item["limitations"]:
            if isinstance(text, str) and text.strip() and text not in limitations:
                limitations.append(text)

    warnings: list[str] = []
    if failures:
        warnings.append(
            "以下关键帧未取得有效鉴伪结果，未参与聚合："
            + "；".join(
                f"{item.get('file_name') or item.get('frame_rank')}（{item['error_message']}）"
                for item in failures
            )
        )
    if frames_scored < frames_total:
        warnings.append("视频级结论只覆盖成功鉴伪的关键帧，帧覆盖不足会削弱结论强度。")
    if frozen_frames:
        warnings.append(FROZEN_BRANCH_WARNING.format(count=frozen_frames, total=frames_total))

    payload = {
        "schema_version": "3.0",
        "module": "deepfake",
        "status": "success",
        "query_image": str(query_image),
        "is_deepfake": band_key == "high",
        "score": median_score,
        "label": "fake" if band_key == "high" else "real",
        "runtime_seconds": runtime_seconds,
        "details": {
            "score_semantics": (
                "视频级分数为全部成功关键帧深伪风险分的中位数，不是概率、不是系统准确率。"
            ),
            "branch": "video_multi_frame_aggregation",
            "band": band_key,
            "band_label": labels.get(band_key, band_key),
            "band_reason": band_reason,
            "band_rule": band_rule,
            "band_thresholds": thresholds,
            "video_band_rule": video_rule,
            "video_band_thresholds": (
                {
                    "statistic": "median",
                    "low_max": video_low_max,
                    "medium_max": video_medium_max,
                }
                if video_rule == CALIBRATED_VIDEO_RULE
                else {}
            ),
            "frames_scored": frames_scored,
            "frames_total": frames_total,
            "frame_band_counts": band_counts,
            "frame_band_sources": band_sources,
            "frame_branches": branch_counts,
            "score_statistics": {
                "median": round(median_score, 6),
                "mean": round(float(statistics.fmean(scores)), 6),
                "max": round(max(scores), 6),
                "min": round(min(scores), 6),
            },
            "per_frame": per_frame,
            "frame_failures": failures,
            "consistency_statement": consistency,
            "routing_note": (
                "视频模式下逐帧独立路由；本次帧级分支分布见 frame_branches，"
                "逐帧明细见 per_frame 与各帧结果文件。"
            ),
            "calibration_set": scored[0]["payload"].get("details", {}).get("calibration_set"),
            "validation_set": scored[0]["payload"].get("details", {}).get("validation_set"),
        },
        "metrics": {
            "frames_scored": frames_scored,
            "frames_total": frames_total,
            "high_band_frames": band_counts["high"],
            "video_score_median": median_score,
            "video_score_max": max(scores),
            "frozen_threshold_frames": frozen_frames,
        },
        "explanation": {
            "summary": (
                f"视频级深伪风险分带为「{labels.get(band_key, band_key)}」：{band_reason}"
            ),
            "basis": (
                "帧级分数来自冻结内核（没有校准分带）；视频级分带由冻结阈值的二值判定按一致性规则导出。"
                if frozen_frames
                else (
                    "帧级分数来自二代整图检测器；视频级分带由帧级中位数对照独立视频校准集的真人视频分位点给出。"
                    if video_rule == CALIBRATED_VIDEO_RULE
                    else "帧级分数来自二代整图检测器；视频级分带由帧级分带的一致性规则给出。"
                )
            ),
            "consistency": consistency,
        },
        "visual_evidence": {
            "module": "deepfake",
            "status": "video_timeline",
            "frames": per_frame,
        },
        "artifact_manifest": list(scored[0]["payload"].get("artifact_manifest") or []),
        "frame_evidence": per_frame,
        "progress_events": [],
        "warnings": warnings,
        "limitations": limitations,
        "runtime": {
            "elapsed_seconds": runtime_seconds,
            "frames_scored": frames_scored,
            "aggregation": "median_of_frame_scores",
        },
        "provenance": {
            "frame_scores": per_frame,
            "model_sha256": (
                scored[0]["payload"].get("provenance", {}).get("model_sha256")
                if isinstance(scored[0]["payload"].get("provenance"), Mapping)
                else None
            ),
            "aggregation_rule": video_rule,
        },
        "error_message": None,
        "method": {
            "name": "二代整图深伪风险甄别（视频多帧聚合）",
            "branch": "video_multi_frame_aggregation",
            "model": (
                "CommunityForensics-DeepfakeDet-ViT (ViT-Small/384, ONNX Runtime)"
                if not frozen_frames
                else "v0.4.0 冻结融合（MobileNetV3 空间 + 频域统计；无校准分带）"
            ),
            "fusion_method": (
                "帧级分数取中位数；视频级分带按帧级分带多数一致性规则"
                if video_rule == DEEPFAKE_RULE
                else "帧级分数取中位数；视频级中位分对照独立视频校准集真人视频分位点分带"
            ),
            "frame_branches": branch_counts,
        },
    }
    return payload


def aggregate_identity_frames(
    frame_results: Sequence[Mapping[str, Any]],
    *,
    query_image: str,
    extra_limitations: Sequence[str] = (),
) -> dict[str, Any]:
    """把逐帧身份检索结果按人员聚合（跨帧候选聚合，不做嵌入级聚合）。"""

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    runtime_seconds = 0.0
    for entry in frame_results:
        frame = _frame_meta(entry)
        payload = entry.get("payload")
        payload = dict(payload) if isinstance(payload, Mapping) else None
        runtime_seconds += _number((payload or {}).get("runtime_seconds")) or 0.0
        candidates = (payload or {}).get("candidates")
        if payload is None or str(payload.get("status")) != "success" or not isinstance(candidates, list):
            failures.append(
                {
                    **_frame_label(frame),
                    "error_message": str(
                        (payload or {}).get("error_message") or entry.get("error") or "未返回结果"
                    ),
                }
            )
            continue
        successes.append({"frame": frame, "payload": payload, "candidates": list(candidates)})

    if not successes:
        return {
            "schema_version": "3.0",
            "module": "identity",
            "status": "error",
            "query_image": str(query_image),
            "candidates": [],
            "runtime_seconds": runtime_seconds,
            "error_message": "全部关键帧的身份检索均未成功，视频级候选不可用。",
            "details": {
                "video_aggregation": {"rule": IDENTITY_RULE, "frames_with_faces": 0},
                "frame_failures": failures,
            },
            "metrics": {"frames_analyzed": len(frame_results), "frames_with_faces": 0},
            "method": {"name": "insightface_closed_set_person_mean_cosine（视频多帧聚合）"},
            "explanation": {},
            "visual_evidence": {},
            "artifact_manifest": [],
            "warnings": [],
            "limitations": list(VIDEO_LIMITATIONS_SHARED),
            "runtime": {"elapsed_seconds": runtime_seconds},
            "provenance": {},
        }

    per_identity: dict[str, dict[str, Any]] = {}
    for item in successes:
        frame = item["frame"]
        for candidate in item["candidates"]:
            if not isinstance(candidate, Mapping):
                continue
            identity_id = str(candidate.get("identity_id") or candidate.get("identity_name") or "")
            if not identity_id:
                continue
            score = _number(candidate.get("mean_similarity"))
            if score is None:
                score = _number(candidate.get("score"))
            if score is None:
                continue
            record = per_identity.setdefault(
                identity_id,
                {
                    "identity_id": identity_id,
                    "identity_name": candidate.get("identity_name") or identity_id,
                    "observations": [],
                    "best": None,
                },
            )
            observation = {
                **_frame_label(frame),
                "score": float(score),
                "candidate": dict(candidate),
            }
            record["observations"].append(observation)
            if record["best"] is None or float(score) > float(record["best"]["score"]):
                record["best"] = observation

    if not per_identity:
        return {
            "schema_version": "3.0",
            "module": "identity",
            "status": "error",
            "query_image": str(query_image),
            "candidates": [],
            "runtime_seconds": runtime_seconds,
            "error_message": "关键帧检索完成但没有返回任何候选，无法形成视频级排序。",
            "details": {
                "video_aggregation": {"rule": IDENTITY_RULE, "frames_with_faces": len(successes)},
                "frame_failures": failures,
            },
            "metrics": {"frames_analyzed": len(frame_results), "frames_with_faces": len(successes)},
            "method": {"name": "insightface_closed_set_person_mean_cosine（视频多帧聚合）"},
            "explanation": {},
            "visual_evidence": {},
            "artifact_manifest": [],
            "warnings": [],
            "limitations": list(VIDEO_LIMITATIONS_SHARED),
            "runtime": {"elapsed_seconds": runtime_seconds},
            "provenance": {},
        }

    frames_with_faces = len(successes)
    ranked: list[dict[str, Any]] = []
    for identity_id, record in per_identity.items():
        observations = sorted(record["observations"], key=lambda item: item["frame_rank"] or 0)
        values = [item["score"] for item in observations]
        mean_value = float(statistics.fmean(values))
        best = dict(record["best"])
        entry = dict(best["candidate"])
        entry.update(
            {
                "identity_id": identity_id,
                "identity_name": record["identity_name"],
                "score": mean_value,
                "mean_similarity": mean_value,
                "mean_across_frames": mean_value,
                "max_across_frames": float(max(values)),
                "frame_support": len(observations),
                "frames_with_faces": frames_with_faces,
                "frame_scores": [
                    {
                        **_frame_label(item),
                        "score": round(item["score"], 6),
                    }
                    for item in observations
                ],
                "best_frame": _frame_label(best),
                "best_frame_score": float(best["score"]),
            }
        )
        ranked.append(entry)
    ranked.sort(
        key=lambda item: (
            -float(item["mean_across_frames"]),
            -int(item["frame_support"]),
            -float(item["max_across_frames"]),
            str(item.get("identity_id")),
        )
    )
    top_score = float(ranked[0]["mean_across_frames"])
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
        item["gap_from_top1"] = round(top_score - float(item["mean_across_frames"]), 6)
    margin = (
        float(ranked[0]["mean_across_frames"]) - float(ranked[1]["mean_across_frames"])
        if len(ranked) > 1
        else float(ranked[0]["mean_across_frames"])
    )

    best_frame_payload = max(
        successes,
        key=lambda item: _number((item["candidates"][0] or {}).get("mean_similarity"))
        or _number((item["candidates"][0] or {}).get("score"))
        or 0.0,
    )
    base_payload = best_frame_payload["payload"]
    base_metrics = _mapping(base_payload.get("metrics"))
    base_details = _mapping(base_payload.get("details"))
    per_frame_margins = [
        {
            **_frame_label(item["frame"]),
            "top1_top2_margin": _number(_mapping(item["payload"].get("metrics")).get("top1_top2_margin")),
            "top1_identity": (item["candidates"][0] or {}).get("identity_id")
            if isinstance(item["candidates"][0], Mapping)
            else None,
        }
        for item in successes
    ]
    limitations: list[str] = []
    for source in [
        *VIDEO_LIMITATIONS_SHARED,
        *extra_limitations,
        *[str(text) for text in (base_payload.get("limitations") or [])],
    ]:
        if isinstance(source, str) and source.strip() and source not in limitations:
            limitations.append(source)

    warnings = list(base_payload.get("warnings") or [])
    if failures:
        warnings.append(
            "以下关键帧没有参与身份聚合（未检出人脸或检索失败）："
            + "；".join(
                f"{item.get('file_name') or item.get('frame_rank')}（{item['error_message']}）"
                for item in failures
            )
        )
    warnings.append(
        "视频级候选排序为跨帧平均余弦相似度（候选级聚合），未做嵌入级跨帧聚合；"
        "参考图级字段取自得分最高的那一帧。"
    )

    payload = {
        "schema_version": "3.0",
        "module": "identity",
        "status": "success",
        "query_image": str(query_image),
        "candidates": ranked,
        "runtime_seconds": runtime_seconds,
        "error_message": None,
        "method": {
            **_mapping(base_payload.get("method")),
            "name": "insightface_closed_set_person_mean_cosine（视频多帧候选聚合）",
            "video_aggregation": IDENTITY_RULE,
        },
        "metrics": {
            **base_metrics,
            "top1_top2_margin": margin,
            "video_top1_top2_margin": margin,
            "frames_analyzed": len(frame_results),
            "frames_with_faces": frames_with_faces,
            "per_frame_margins": per_frame_margins,
        },
        "explanation": {
            "summary": "这是当前身份库中的候选排序（跨帧聚合），不是最终身份确认。",
            "ranking_basis": (
                "每个关键帧单独检索后，按人员聚合：同一人员在所有有脸帧上的余弦相似度取平均后排序；"
                "支持帧数与逐帧分数一并公开。"
            ),
            "top1_reason": (
                f"{ranked[0].get('identity_name')} 的跨帧平均余弦相似度为 "
                f"{float(ranked[0]['mean_across_frames']):.6f}，支持帧数 "
                f"{ranked[0]['frame_support']}/{frames_with_faces}，与第二候选差值为 {margin:.6f}。"
            ),
        },
        "visual_evidence": {
            **_mapping(base_payload.get("visual_evidence")),
            "module": "identity",
            "status": "video_aggregated",
            "frames": [
                {
                    **_frame_label(item["frame"]),
                    "top1_identity": (item["candidates"][0] or {}).get("identity_id")
                    if isinstance(item["candidates"][0], Mapping)
                    else None,
                    "query_face_count": _mapping(item["payload"].get("metrics")).get("query_face_count"),
                }
                for item in successes
            ],
        },
        "artifact_manifest": list(base_payload.get("artifact_manifest") or []),
        "frame_evidence": [
            {
                **_frame_label(item["frame"]),
                "status": "success",
                "top1_identity": (item["candidates"][0] or {}).get("identity_id")
                if isinstance(item["candidates"][0], Mapping)
                else None,
                "top1_score": _number((item["candidates"][0] or {}).get("mean_similarity"))
                if isinstance(item["candidates"][0], Mapping)
                else None,
                "top1_top2_margin": _number(_mapping(item["payload"].get("metrics")).get("top1_top2_margin")),
                "detection_scale_used": _mapping(item["payload"].get("metrics")).get("detection_scale_used"),
            }
            for item in successes
        ]
        + [{**item, "status": "error"} for item in failures],
        "progress_events": [],
        "warnings": warnings,
        "limitations": limitations,
        "runtime": {
            "elapsed_seconds": runtime_seconds,
            "frames_analyzed": len(frame_results),
            "frames_with_faces": frames_with_faces,
            "aggregation": IDENTITY_RULE,
            "device": _mapping(base_payload.get("runtime")).get("device"),
        },
        "details": {
            **base_details,
            "video_aggregation": {
                "rule": IDENTITY_RULE,
                "frames_analyzed": len(frame_results),
                "frames_with_faces": frames_with_faces,
                "ranking_basis": "跨帧平均余弦相似度（候选级聚合）",
                "reference_fields_source": best_frame_payload["frame"].get("file_name"),
                "note": "候选级聚合；嵌入级跨帧聚合列为后续工作。",
            },
            "frame_failures": failures,
        },
        "provenance": {
            **_mapping(base_payload.get("provenance")),
            "video_frames": [_frame_label(item["frame"]) for item in successes],
            "aggregation_rule": IDENTITY_RULE,
        },
    }
    return payload


def annotate_single_frame(
    payload: Mapping[str, Any],
    *,
    frame: Mapping[str, Any],
    role: str,
    rule: str = SOURCE_RULE,
    extra_limitations: Sequence[str] = (),
) -> dict[str, Any]:
    """给「只跑一帧」的模块结果补上视频语境（底图检索用）。"""

    output = dict(payload)
    frame_label = _frame_label(frame)
    details = _mapping(output.get("details"))
    limitations = [str(item) for item in (output.get("limitations") or []) if isinstance(item, str)]
    for text in [*VIDEO_LIMITATIONS_SHARED, *extra_limitations]:
        if text not in limitations:
            limitations.append(text)
    details["video_aggregation"] = {
        "rule": rule,
        "role": role,
        "frame": frame_label,
        "note": "视频模式下该模块只使用最可靠的一帧；逐帧检索在 CPU 上昂贵且同源相邻帧近乎同图。",
    }
    output["details"] = details
    output["limitations"] = limitations
    output["frame_evidence"] = [{**frame_label, "status": str(output.get("status") or "error")}]
    runtime = _mapping(output.get("runtime"))
    runtime.setdefault("frames_analyzed", 1)
    output["runtime"] = runtime
    return output


def build_video_summary(
    *,
    selection: Mapping[str, Any],
    module_payloads: Mapping[str, Mapping[str, Any]],
    input_file: str,
    sha256: str | None = None,
) -> dict[str, Any]:
    """汇总整个视频任务的视频级视图（写进 video_summary.json 与统一结果）。"""

    key_frames = [
        {
            "frame_rank": item.get("frame_rank"),
            "file_name": item.get("file_name"),
            "sha256": item.get("sha256"),
            "timestamp_seconds": item.get("timestamp_seconds"),
            "quality_score": item.get("quality_score"),
            "face_count": item.get("face_count"),
            "selection_reasons": list(item.get("selection_reasons") or []),
        }
        for item in (selection.get("key_frames") or [])
        if isinstance(item, Mapping)
    ]
    modules: dict[str, Any] = {}
    deepfake = _mapping(module_payloads.get("deepfake"))
    if deepfake:
        details = _mapping(deepfake.get("details"))
        modules["deepfake"] = {
            "status": deepfake.get("status"),
            "score": _number(deepfake.get("score")),
            "band": details.get("band"),
            "band_label": details.get("band_label"),
            "band_rule": details.get("band_rule"),
            "frames_scored": details.get("frames_scored"),
            "frame_band_counts": _mapping(details.get("frame_band_counts")),
            "frame_band_sources": _mapping(details.get("frame_band_sources")),
            "frame_branches": _mapping(details.get("frame_branches")),
            "rule": details.get("video_band_rule"),
        }
    identity = _mapping(module_payloads.get("identity"))
    if identity:
        candidates = identity.get("candidates") if isinstance(identity.get("candidates"), list) else []
        top = _mapping(candidates[0]) if candidates else {}
        modules["identity"] = {
            "status": identity.get("status"),
            "frames_with_faces": _mapping(identity.get("metrics")).get("frames_with_faces"),
            "top1_identity": top.get("identity_id"),
            "top1_identity_name": top.get("identity_name"),
            "top1_mean_similarity": _number(top.get("mean_similarity")),
            "top1_frame_support": top.get("frame_support"),
            "top1_top2_margin": _number(_mapping(identity.get("metrics")).get("top1_top2_margin")),
            "rule": _mapping(identity.get("details")).get("video_aggregation", {}).get("rule")
            if isinstance(_mapping(identity.get("details")).get("video_aggregation"), Mapping)
            else None,
        }
    source = _mapping(module_payloads.get("source"))
    if source:
        candidates = source.get("candidates") if isinstance(source.get("candidates"), list) else []
        top = _mapping(candidates[0]) if candidates else {}
        modules["source"] = {
            "status": source.get("status"),
            "top1_candidate": _candidate_name(top),
            "top1_score": _first_number(
                top.get("final_score"), top.get("score"), top.get("global_score")
            ),
            "top1_global_score": _number(top.get("global_score")),
            "top1_local_score": _number(top.get("local_score")),
            "rule": SOURCE_RULE,
        }
    return {
        "schema_version": "1.0",
        "input_type": "video",
        "video_file": str(input_file),
        "video_sha256": sha256,
        "video": _mapping(selection.get("video")),
        "frame_selection": _mapping(selection.get("selection"))
        or _mapping(selection.get("policy")),
        "key_frames": key_frames,
        "modules": modules,
        "limitations": list(VIDEO_LIMITATIONS),
    }


__all__ = [
    "DEEPFAKE_RULE",
    "DEFAULT_BAND_LABELS",
    "IDENTITY_RULE",
    "SOURCE_RULE",
    "VIDEO_LIMITATIONS",
    "VIDEO_LIMITATIONS_DEEPFAKE",
    "VIDEO_LIMITATIONS_SHARED",
    "aggregate_deepfake_frames",
    "aggregate_identity_frames",
    "annotate_single_frame",
    "build_video_summary",
    "rebase_frame_artifacts",
]
