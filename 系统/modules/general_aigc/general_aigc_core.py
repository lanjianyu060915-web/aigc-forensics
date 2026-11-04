"""通用 AIGC 分支：把引擎分数整理成结果 JSON，并按校准边界分档。

这里不做真假判定：分档边界来自真实样本分数的分位点，只表示“这个分数在真实样本里有多少见”。
"""

from __future__ import annotations

import json
import os
import time
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
