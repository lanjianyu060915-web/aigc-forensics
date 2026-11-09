"""通用 AIGC 分支：把引擎分数整理成结果 JSON，并按校准边界分档。

这里不做真假判定：分档边界来自真实样本分数的分位点，只表示“这个分数在真实样本里有多少见”。
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
BAND_LABELS = {"low": "证据不足", "medium": "需人工复核", "high": "高风险提示"}
UNCALIBRATED_LABEL = "未标定"

LIMITATIONS = [
    "模型输出是筛查分数，不是准确率、概率或司法鉴定结论。",
    "证据不足不等于内容真实，只说明模型没有发现足够的风险信号。",
    "分档边界来自特定数据集的真实样本，换一批来源后误报率会变化，结果只作线索。",
    "训练媒体和部分权重的许可限于研究用途。",
]


def input_kind(path: Path) -> str:
    """图片走单帧路径，其余按视频解码；解码失败由引擎报错。"""
    return "image" if path.suffix.casefold() in IMAGE_SUFFIXES else "video"


def load_calibration(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def band_of(logit: float, thresholds: dict[str, Any] | None) -> tuple[str | None, str, str]:
    """返回（档位键、档位名、理由）；没有边界时不分档，不用默认值冒充标定结果。"""
    if not thresholds or "low_max" not in thresholds or "medium_max" not in thresholds:
        return None, UNCALIBRATED_LABEL, "这类输入还没有校准边界，只报告原始分数。"
    low_max, medium_max = float(thresholds["low_max"]), float(thresholds["medium_max"])
    if logit >= medium_max:
        return "high", BAND_LABELS["high"], f"分数 {logit:.3f} 高于真实样本分布上界 {medium_max:.3f}。"
    if logit >= low_max:
        return (
            "medium",
            BAND_LABELS["medium"],
            f"分数 {logit:.3f} 落在真实样本高分尾部（{low_max:.3f}–{medium_max:.3f}）。",
        )
    return "low", BAND_LABELS["low"], f"分数 {logit:.3f} 低于真实样本高分尾部的下界 {low_max:.3f}。"


def face_branch_payload(
    logits: list[float], crop_files: list[str], calibration: dict[str, Any]
) -> dict[str, Any]:
    """人脸裁剪上的 CommunityForensics 分数；多张裁剪取均值后分档。"""
    if not logits:
        return {"status": "no_face", "logit": None, "band": None, "band_label": "未检出人脸"}
    logit = sum(logits) / len(logits)
    thresholds = calibration.get("face_crop") if isinstance(calibration.get("face_crop"), dict) else None
    band, band_label, band_reason = band_of(logit, thresholds)
    return {
        "status": "success",
        "logit": logit,
        "band": band,
        "band_label": band_label,
        "band_reason": band_reason,
        "band_thresholds": (
            {
                "low_max": thresholds["low_max"],
                "medium_max": thresholds["medium_max"],
                "calibration_set": thresholds.get("calibration_set", {}),
            }
            if band is not None and thresholds is not None
            else None
        ),
        "crops": [
            {"file": name, "logit": value}
            for name, value in zip(crop_files, logits, strict=True)
        ],
    }


def enrich_face_branch(
    original: dict[str, Any],
    faces: list[dict[str, Any]],
    scores: list[float],
    calibration: dict[str, Any],
    *,
    backend: str = "cf384",
    band_aggregation: str = "largest",
    min_face_side: int = 64,
    face_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """扩充逐脸信息；默认旧后端最大脸路径保留所有已有字段。"""
    if backend not in {"cf384", "gend"} or band_aggregation not in {"largest", "max_all"}:
        raise ValueError("人脸后端或定档口径无效")
    if len(faces) != len(scores) or not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("逐脸记录与有效分数不一致")
    if not isinstance(min_face_side, int) or isinstance(min_face_side, bool) or min_face_side < 1:
        raise ValueError("人脸最小短边必须为正整数")
    legacy = backend == "cf384" and band_aggregation == "largest"
    result = dict(original)
    per_face = []
    grouped: dict[str, list[dict[str, Any]]] = OrderedDict()
    for record, value in zip(faces, scores, strict=True):
        row = {key: value for key, value in record.items() if key != "path"}
        short_side = int(row.get("short_edge_px", 0))
        too_small = short_side < min_face_side
        eligible = bool(row.get("legacy_selected")) if legacy else not too_small
        row.update(score=float(value), score_kind="logit" if backend == "cf384" else "model_probability",
                   score_status="success",
                   too_small=too_small, eligible_for_band=eligible, used_for_band=False,
                   reliability_note="人脸过小，结果不可靠" if too_small else "未触发小脸限制")
        per_face.append(row)
        grouped.setdefault(str(row.get("source") or "image"), []).append(row)
    frame_scores = []
    for source, rows in grouped.items():
        eligible = [row for row in rows if row["eligible_for_band"]]
        if band_aggregation == "largest":
            eligible = [row for row in eligible if row.get("legacy_selected") or row.get("face_id") == 1]
        chosen = max(eligible, key=lambda row: row["score"]) if eligible else None
        if chosen is not None:
            chosen["used_for_band"] = True
        frame_scores.append({"source": source, "timestamp_seconds": rows[0].get("timestamp_seconds"),
                             "frame_id": rows[0].get("frame_id"),
                             "score": chosen["score"] if chosen is not None else None,
                             "selected_face_id": chosen["face_id"] if chosen is not None else None,
                             "selected_face_uid": chosen.get("face_uid") if chosen is not None else None})
    for row in per_face:
        row["aggregation_exclusion_reason"] = (None if row["used_for_band"] else
            "too_small" if not legacy and row["too_small"] else "not_selected_by_frozen_rule")
    if not legacy:
        chosen_scores = [row["score"] for row in frame_scores if row["score"] is not None]
        score = sum(chosen_scores) / len(chosen_scores) if chosen_scores else None
        thresholds = calibration.get("face_crop") if backend == "cf384" else (
            (calibration.get("aggregations") or {}).get(band_aggregation))
        thresholds = thresholds if isinstance(thresholds, dict) else None
        band, label, reason = band_of(score, thresholds) if score is not None else (
            None, "无可参与定档人脸", "没有满足可靠性条件的人脸，本分支不参与总体档位。")
        result.update(status="success" if score is not None else ("no_eligible_face" if per_face else "no_face"),
                      logit=score, band=band, band_label=label, band_reason=reason,
                      band_thresholds={"low_max": thresholds["low_max"], "medium_max": thresholds["medium_max"],
                                       "calibration_set": thresholds.get("calibration_set", {})}
                      if band is not None and thresholds is not None else None)
        if backend == "gend":
            result["score"] = score
    result.update(backend=backend, band_aggregation=band_aggregation, video_aggregation="mean_of_frame_scores",
                  min_face_side=min_face_side, per_face=per_face, frame_scores=frame_scores,
                  highest_score_face=dict(max(per_face, key=lambda row: row["score"])) if per_face else None,
                  compatibility_note=("原后端最大脸模式保留旧版40像素门槛以恢复旧结果；40–63像素人脸仍标为不可靠。"
                                      if legacy else None))
    result["face_audit"] = merge_face_check_results(face_audit, per_face)
    return result


def merge_face_check_results(
    face_audit: dict[str, Any] | None, per_face: list[dict[str, Any]], *,
    inference_status: str = "success", error_type: str | None = None,
) -> dict[str, Any]:
    """Attach scoring to unique task-local records without inventing missing coverage.

    A failed model call can use this helper with an empty per_face list. It never
    calculates a band or changes a score. Numeric face_id alone is not a join key.
    """
    if inference_status not in {"success", "failed", "not_run"}:
        raise ValueError("逐脸推理状态无效")
    result = deepcopy(face_audit) if isinstance(face_audit, dict) else {
        "schema_version": "1.0", "status": "unrecorded", "scope": "task_local_face_checks",
        "frame_records": [], "face_checks": [], "coverage": {"count_status": "unrecorded",
        "detected_total": None, "processed_count": None, "truncated_count": None, "failed_count": None},
        "identity_coverage": {"status": "unrecorded"}, "pair_coverage": {"status": "unrecorded"}}
    checks = result.setdefault("face_checks", [])
    score_rows: dict[str, list[dict[str, Any]]] = {}
    for row in per_face:
        uid = row.get("face_uid")
        if isinstance(uid, str) and uid:
            score_rows.setdefault(uid, []).append(row)
    check_counts: dict[str, int] = {}
    for row in checks:
        uid = row.get("face_uid")
        if isinstance(uid, str):
            check_counts[uid] = check_counts.get(uid, 0) + 1
    unmatched = []
    used_uids = set()
    for check in checks:
        uid = check.get("face_uid")
        matches = score_rows.get(uid, [])
        if matches and len(matches) == 1 and check_counts.get(uid) == 1 and check.get("crop_status") == "success":
            scored = matches[0]
            if (scored.get("frame_id") != check.get("frame_id") or scored.get("bbox") != check.get("bbox")
                    or scored.get("source_id") != check.get("source_id")):
                check.update(score_status="unmatched", score=None, used_for_band=False,
                             score_binding_reason="coordinate_or_source_mismatch")
                continue
            value = scored.get("score")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                check.update(score_status="failed", score=None, used_for_band=False,
                             score_binding_reason="invalid_score")
                continue
            for key in ("score", "score_kind", "score_status", "eligible_for_band", "used_for_band",
                        "aggregation_exclusion_reason", "reliability_note", "too_small"):
                if key in scored:
                    check[key] = deepcopy(scored[key])
            used_uids.add(uid)
        elif matches:
            check.update(score_status="ambiguous", score=None, used_for_band=False,
                         score_binding_reason="duplicate_or_uncropped_record")
        elif check.get("crop_status") == "success":
            check.update(score_status="failed" if inference_status == "failed" else "not_run",
                         score=None, used_for_band=False,
                         score_binding_reason="model_failed" if inference_status == "failed" else "score_unrecorded")
            if error_type:
                check["model_error_type"] = error_type
    for row in per_face:
        if row.get("face_uid") not in used_uids:
            unmatched.append({key: deepcopy(row.get(key)) for key in ("face_uid", "face_id", "source", "frame_id")})
    result["score_binding"] = {"matched_count": len(used_uids), "unmatched_scores": unmatched}
    result["inference_status"] = inference_status
    coverage = result.setdefault("coverage", {})
    valid_scores = [row for row in per_face if isinstance(row.get("score"), (int, float))
                    and not isinstance(row.get("score"), bool) and math.isfinite(row["score"])]
    coverage.update(module="face_aigc", scored_count=len(valid_scores),
                    used_for_band_count=sum(bool(row.get("used_for_band")) for row in per_face),
                    bound_score_count=len(used_uids),
                    model_failed_count=sum(row.get("score_status") == "failed" for row in checks))
    # Unknown crop coverage must stay unknown even when scores were supplied.
    if inference_status == "failed":
        result["status"] = "inference_failed"
    elif unmatched:
        result["status"] = "incomplete"
    return result


def error_payload(query: Path, message: str, *, started: float, error_type: str) -> dict[str, Any]:
    runtime = time.perf_counter() - started
    return {
        "schema_version": "1.0",
        "module": "general_aigc",
        "status": "error",
        "input_file": query.name,
        "input_kind": input_kind(query),
        "score": None,
        "logit": None,
        "band": None,
        "band_label": None,
        "band_reason": None,
        "runtime_seconds": runtime,
        "details": {"error_type": error_type},
        "warnings": [],
        "limitations": list(LIMITATIONS),
        "error_message": message,
    }


def success_payload(
    query: Path,
    result: Any,
    *,
    score: float,
    model: dict[str, Any],
    calibration: dict[str, Any],
    device: str,
    device_name: str,
    started: float,
) -> dict[str, Any]:
    thresholds = calibration.get(result.kind) if isinstance(calibration.get(result.kind), dict) else None
    band, band_label, band_reason = band_of(result.logit, thresholds)
    warnings = list(result.warnings)
    if band is None:
        warnings.append("这类输入还没有校准边界，只能参考原始分数。")
    return {
        "schema_version": "1.0",
        "module": "general_aigc",
        "status": "success",
        "input_file": query.name,
        "input_kind": result.kind,
        "score": score,
        "logit": result.logit,
        "band": band,
        "band_label": band_label,
        "band_reason": band_reason,
        "band_thresholds": (
            {
                "low_max": thresholds["low_max"],
                "medium_max": thresholds["medium_max"],
                "statistic": "融合 logit",
                "calibration_set": thresholds.get("calibration_set", {}),
            }
            if band is not None and thresholds is not None
            else None
        ),
        "branches": {
            "clip_original_logit": result.clip_original_logit,
            "clip_refit_logit": result.clip_refit_logit,
            "clip_logit": result.clip_logit,
            "community_logit": result.community_logit,
            "formula": result.formula,
        },
        "frames": [
            {
                "timestamp_seconds": timestamp,
                "clip_logit": sum(crops) / len(crops),
                "crop_logits": list(crops),
            }
            for timestamp, crops in zip(result.frame_timestamps, result.frame_logits, strict=True)
        ],
        "community_frames": [
            {"timestamp_seconds": timestamp, "logit": logit}
            for timestamp, logit in zip(
                result.community_timestamps, result.community_frame_logits, strict=True
            )
        ],
        "media": {
            "width": result.width,
            "height": result.height,
            "duration_seconds": result.duration_seconds,
            "decode_mode": result.decode_mode,
        },
        "model": model,
        "runtime": {
            "device": device,
            "device_name": device_name,
            "frame_batch_size": result.frame_batch_size,
            "decode_seconds": result.decode_seconds,
            "inference_seconds": result.inference_seconds,
        },
        "runtime_seconds": time.perf_counter() - started,
        "warnings": warnings,
        "limitations": list(LIMITATIONS),
        "error_message": None,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
