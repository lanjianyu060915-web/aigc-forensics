"""为已有总体档位补充测试集统计和提示，不产生或修改判定。"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any


EVIDENCE_PATH = Path(__file__).with_name("band_evidence.json")
_FACE_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "models" / "video-c38-refit3-cf384" / "calibration.json"
UNAVAILABLE_NOTE = "当前人脸配置没有对应的总体档位统计，不能引用原默认配置的测试比例。"
DISCLAIMER = (
    "下表是测试集上的统计：伪造样本落入这一档的比例，除以真实样本落入这一档的比例。"
    "它说明这一档的分量，不是本案的概率，也不是正式的似然比，换一批数据会变化。"
    "结果只作侦查线索，不替代人工复核和司法鉴定。"
)
LOW_NOTE = "证据不足不能排除伪造，约三分之二的换脸测试图都落在这一档。"
SMALL_SAMPLE_NOTE = "样本少，只能看个大概"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def legacy_overall_statistics_match(general: Mapping[str, Any] | None) -> bool:
    """只判断旧统计能否展示，不修改分数、档位或总体规则。"""
    if not isinstance(general, Mapping):
        return False
    # 旧版曾没有人脸分支键；显式空值或无效对象不属于这种旧结构。
    if "face_branch" not in general:
        return True
    face = general["face_branch"]
    if not isinstance(face, Mapping) or not face:
        return False
    backend_present, aggregation_present = "backend" in face, "band_aggregation" in face
    if backend_present or aggregation_present:
        if not (backend_present and aggregation_present
                and face["backend"] == "cf384" and face["band_aggregation"] == "largest"):
            return False
    else:
        # 原第二版实际保存的旧形状；第三轮字段缺配置不能伪装成旧任务。
        legacy_keys = {"status", "logit", "band", "band_label", "band_reason", "band_thresholds", "crops", "records"}
        if set(face) - legacy_keys or face.get("status") not in {"success", "no_face"}:
            return False
    if "band_thresholds" not in face:
        # 原默认无脸分支不保存阈值，保持这一有实证的兼容例外。
        return face.get("status") == "no_face"
    thresholds = face["band_thresholds"]
    if not isinstance(thresholds, Mapping):
        return False
    try:
        frozen = json.loads(_FACE_CALIBRATION_PATH.read_text(encoding="utf-8"))["face_crop"]
    except (OSError, UnicodeError, ValueError, TypeError, KeyError):
        return False
    if not isinstance(frozen, Mapping):
        return False
    for key in ("low_max", "medium_max"):
        actual, expected = thresholds.get(key), frozen.get(key)
        if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual)
                or isinstance(expected, bool) or not isinstance(expected, (int, float))
                or not math.isfinite(expected) or actual != expected):
            return False
    return True


def load_band_evidence(path: str | Path = EVIDENCE_PATH) -> dict[str, Any]:
    """统计文件不可用时返回空结果，检测流程不依赖它。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) and isinstance(data.get("scenarios"), list) else {}


def get_band_evidence(band: Any, data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """返回当前档位在五类测试场景中的统计，不把它解释为个案概率。"""
    if band not in ("low", "medium", "high"):
        return {}
    source = load_band_evidence() if data is None else _mapping(data)
    scenarios = source.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        return {}
    rows = []
    required = {"real_count", "real_total", "real_ratio", "fake_count", "fake_total", "fake_ratio", "ratio", "small_sample"}
    for scenario in scenarios:
        scenario = _mapping(scenario)
        stats = _mapping(_mapping(scenario.get("bands")).get(band))
        if not required.issubset(stats) or not scenario.get("label") or not scenario.get("source_sha256"):
            return {}
        numbers = [stats[key] for key in required - {"small_sample", "ratio"}]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in numbers):
            return {}
        ratio = stats["ratio"]
        if ratio is not None and (isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio)):
            return {}
        rows.append({
            "scenario": scenario.get("key"), "label": scenario["label"],
            "source_file": scenario.get("source_file"), "source_sha256": scenario["source_sha256"],
            **stats,
        })
    return {"band": band, "rows": rows, "disclaimer": DISCLAIMER, "low_note": LOW_NOTE if band == "low" else ""}


def build_hints(band: Any, media: Mapping[str, Any] | None, provenance: Mapping[str, Any] | None, visible_label: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    """按既定条件补充低分辨率和文件标识提示。"""
    media = _mapping(media)
    provenance = _mapping(provenance)
    hints = []
    dimensions = [media.get("width"), media.get("height")]
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 for value in dimensions):
        short_edge = min(dimensions)
        if short_edge <= 256:
            hints.append({"code": "low_resolution", "message": (
                f"输入短边只有 {short_edge:g} 像素。图片测试里，缩小到 256 像素再压缩后 AUC 从 0.886 降到 0.752，"
                "这次结论的可靠性可能更低，建议找更高清的原件。"
            )})
    if provenance.get("status") == "success" and provenance.get("verdict") in {"labelled_ai", "labelled_possible", "generator_hint"} and band == "low":
        label = provenance.get("verdict_label") or provenance["verdict"]
        hints.append({"code": "provenance_conflict", "message": (
            f"文件标识提示可能是生成内容（{label}），但检测档位是证据不足，两者不一致。"
            "标识可以伪造也可以被抹掉，建议人工复核。"
        )})
    visible = _mapping(visible_label)
    if visible.get("verdict") == "visible_label_found" and band == "low":
        from reporting.text_safety import redact_local_paths
        texts = [str(item.get("text")) for item in visible.get("hits") or [] if isinstance(item, Mapping) and item.get("text")]
        text = redact_local_paths("、".join(dict.fromkeys(texts))[:200]) or "生成相关"
        hints.append({"code": "visible_label_conflict", "message": (
            f"画面上识别到“{text}”字样，但检测档位是证据不足，两者不一致。"
            "可见文字可以被添加也可以被裁掉，建议人工复核。"
        )})
    return hints


def attach_evidence(deepfake: Mapping[str, Any], provenance: Mapping[str, Any] | None, data: Mapping[str, Any] | None = None, *, visible_label: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """仅追加说明字段，保留所有原有值和输入对象。"""
    merged = dict(deepfake)
    overall = _mapping(merged.get("overall"))
    if not overall:
        return merged
    general = _mapping(merged.get("general_aigc"))
    applicable = legacy_overall_statistics_match(merged.get("general_aigc", {}))
    overall["band_evidence"] = (get_band_evidence(overall.get("band"), data) if applicable
        else {"status": "unavailable", "rows": [], "note": UNAVAILABLE_NOTE})
    overall["hints"] = build_hints(overall.get("band"), _mapping(general.get("media")), provenance, visible_label)
    if not applicable:
        overall["hints"].append({"code": "overall_statistics_unavailable", "message": UNAVAILABLE_NOTE})
    merged["overall"] = overall
    return merged


def evidence_table(evidence: Mapping[str, Any]) -> list[dict[str, str]]:
    """界面和报告共用的中文统计行。"""
    rows = []
    for value in evidence.get("rows") or []:
        item = _mapping(value)
        try:
            rows.append({
                "测试场景": str(item["label"]),
                "真实样本": f"{item['real_count']}/{item['real_total']}（{100 * item['real_ratio']:.1f}%）",
                "伪造样本": f"{item['fake_count']}/{item['fake_total']}（{100 * item['fake_ratio']:.1f}%）",
                "比例之比": "无法计算" if item["ratio"] is None else f"{item['ratio']:.2f}",
                "说明": SMALL_SAMPLE_NOTE if item["small_sample"] else "",
            })
        except (KeyError, TypeError, ValueError):
            return []
    return rows


def evidence_lines(overall: Mapping[str, Any]) -> list[str]:
    """旧任务不含新增字段时，输出为空。"""
    lines = []
    for hint in overall.get("hints") or []:
        if isinstance(hint, Mapping) and hint.get("message"):
            lines.append(str(hint["message"]))
    evidence = _mapping(overall.get("band_evidence"))
    rows = evidence_table(evidence)
    if rows:
        lines.append("档位证据强度：")
        lines.extend("；".join(f"{key}：{value}" for key, value in row.items() if value) for row in rows)
        lines.append(DISCLAIMER)
        if evidence.get("low_note"):
            lines.append(LOW_NOTE)
    return lines
