# -*- coding: utf-8 -*-
"""证据质量层：把三个模块的结果整理成「证据矩阵」，不做跨模块数值相加。

设计要点：
- 三个模块的数值性质不同（鉴伪是模型风险分、身份是余弦相似度、素材是自定义
  融合分），因此本层只做证据质量分级与证据罗列，不产生「综合可信度」。
- 每个模块的等级必须可追溯到结果文件中的具体字段。
- 判定结论复用 decision.open_set 的三态结果。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from decision.open_set import (
    STATE_HIGH,
    STATE_INSUFFICIENT,
    STATE_NONE,
    STATE_UNAVAILABLE,
    decide_identity_open_set,
    decide_source_open_set,
)


SCHEMA_VERSION = "1.0"

BAND_GRADES = {"low": "低", "medium": "中", "high": "高"}


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


def _deepfake_block(result: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(result)
    details = _mapping(data.get("details"))
    if data.get("status") != "success":
        return {
            "available": False,
            "grade": "unavailable",
            "grade_label": "未完成",
            "primary_value": None,
            "primary_label": "深伪风险筛查分",
            "text": "本次AIGC鉴伪未成功完成。",
        }
    overall = _mapping(data.get("overall"))
    if overall.get("band") in BAND_GRADES:
        general = _mapping(data.get("general_aigc"))
        return {
            "available": True,
            "grade": str(overall["band"]),
            "grade_label": str(overall.get("band_label") or BAND_GRADES[str(overall["band"])]),
            "primary_value": _number(general.get("logit")),
            "primary_label": "通用 AIGC 融合分数",
            "branch": "two_branch",
            "band_reason": overall.get("reason"),
            "text": f"总体档位：{overall.get('band_label')}",
        }
    band = details.get("band")
    if isinstance(band, str) and band in BAND_GRADES:
        grade = band
        label = str(details.get("band_label") or BAND_GRADES[band])
        branch = details.get("branch") or _mapping(data.get("method")).get("branch")
        text = f"证据分带：{label}"
        if branch == "v040_face_crop":
            text = "由 v0.4.0 冻结人脸裁剪分支判定（本次输入低于路由阈值）"
            grade = "unavailable"
            label = "由回退分支判定"
        return {
            "available": True,
            "grade": grade,
            "grade_label": label,
            "primary_value": _number(data.get("score")),
            "primary_label": "深伪风险筛查分",
            "branch": branch,
            "band_reason": details.get("band_reason"),
            "text": text,
        }
    return {
        "available": True,
        "grade": "unavailable",
        "grade_label": "旧版判定口径",
        "primary_value": _number(data.get("score")),
        "primary_label": "融合风险分",
        "branch": None,
        "text": "本次结果由 v0.4.0 双分支融合口径给出，不含证据分带。",
    }


def _input_quality_block(
    deepfake: Mapping[str, Any], identity: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    identity_metrics = _mapping(identity.get("metrics"))
    source_metrics = _mapping(source.get("metrics"))
    faces = _integer(identity_metrics.get("query_face_count"))
    detection = _number(identity_metrics.get("face_detection_score"))
    notes: list[str] = []
    if identity.get("status") != "success":
        if faces is not None and faces != 1:
            notes.append(f"身份检索查询图检出人脸数为 {faces}，未满足单人脸要求。")
        else:
            notes.append("身份检索未完成，查询图人脸质量未形成可用证据。")
    else:
        notes.append("身份检索查询图满足单人脸要求。")
    if detection is not None:
        notes.append(f"查询图人脸检测分为 {detection:.4f}。")
    if source_metrics.get("query_face_detected") is False:
        notes.append("素材检索未在查询图中检出人脸，掩膜未生效，全局特征包含人脸区域。")
    query_keypoints = _integer(_mapping(source.get("metrics")).get("query_keypoint_count"))
    if query_keypoints is not None:
        notes.append(f"素材检索查询图关键点数为 {query_keypoints}。")
    if identity.get("status") == "success" and source_metrics.get("query_face_detected") is not False:
        grade = "良好"
    elif identity.get("status") == "success" or source_metrics.get("query_face_detected"):
        grade = "一般"
    else:
        grade = "不足"
    if detection is not None and detection < 0.6:
        grade = "一般" if grade == "良好" else grade
        notes.append("人脸检测分偏低，查询质量对检索结果影响较大。")
    return {"grade": grade, "faces_detected": faces, "face_detection_score": detection, "notes": notes}


def _library_coverage_block(identity: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    identity_metrics = _mapping(identity.get("metrics"))
    source_metrics = _mapping(source.get("metrics"))
    identities = _integer(identity_metrics.get("library_identity_count"))
    valid_embeddings = _integer(identity_metrics.get("valid_embedding_count"))
    skipped = _integer(identity_metrics.get("skipped_library_image_count"))
    decoded = _integer(source_metrics.get("library_decoded"))
    library_files = _integer(source_metrics.get("library_file_count"))
    notes: list[str] = []
    if identities is not None:
        notes.append(f"身份库 {identities} 个身份，形成有效特征 {valid_embeddings} 条。")
    if skipped:
        notes.append(f"身份库有 {skipped} 张参考图未形成有效特征，对应身份可能覆盖不全。")
    if library_files is not None:
        notes.append(f"素材库 {library_files} 张，已解码 {decoded} 张。")
    covered = bool(identities) and bool(valid_embeddings)
    return {
        "identity_identities": identities,
        "identity_valid_embeddings": valid_embeddings,
        "identity_skipped_images": skipped,
        "source_files": library_files,
        "source_decoded": decoded,
        "complete": covered,
        "notes": notes,
    }


def _geometry_block(source: Mapping[str, Any]) -> dict[str, Any]:
    candidates = source.get("candidates")
    top = _mapping(candidates[0]) if isinstance(candidates, list) and candidates else {}
    inliers = _integer(top.get("ransac_inlier_count"))
    ratio = _number(top.get("ransac_inlier_ratio"))
    applied = top.get("ransac_applied")
    if inliers is None:
        return {
            "applicable": False,
            "grade": "不适用",
            "inlier_count": None,
            "inlier_ratio": None,
            "note": "本次素材检索没有可用的几何复核结果。",
        }
    if inliers >= 200:
        grade = "强"
    elif inliers >= 40:
        grade = "中"
    else:
        grade = "弱"
    return {
        "applicable": True,
        "grade": grade,
        "ransac_applied": applied,
        "inlier_count": inliers,
        "inlier_ratio": ratio,
        "note": (
            "几何内点数与内点率在本系统上无法区分库内命中与库外查询"
            "（库外样本内点 4-186、库内命中 161-1459 存在重叠），"
            "因此只作为证据强度描述，不参与来源判定。"
        ),
    }


def build_evidence_matrix(
    deepfake_result: Mapping[str, Any] | None = None,
    identity_result: Mapping[str, Any] | None = None,
    source_result: Mapping[str, Any] | None = None,
    *,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造跨模块证据矩阵；不产生综合概率。"""

    deepfake = _mapping(deepfake_result)
    identity = _mapping(identity_result)
    source = _mapping(source_result)

    identity_decision = decide_identity_open_set(identity, thresholds)
    source_decision = decide_source_open_set(source, thresholds)

    state_grades = {
        STATE_HIGH: "high",
        STATE_INSUFFICIENT: "medium",
        STATE_NONE: "low",
        STATE_UNAVAILABLE: "unavailable",
    }

    modules = {
        "deepfake": _deepfake_block(deepfake),
        "identity": {
            "available": identity.get("status") == "success",
            "grade": state_grades.get(identity_decision["state"], "unavailable"),
            "grade_label": identity_decision["state_label"],
            "primary_value": identity_decision["primary_signal"].get("value"),
            "primary_label": "第一候选人员均值余弦",
            "decision": identity_decision,
        },
        "source": {
            "available": source.get("status") == "success",
            "grade": state_grades.get(source_decision["state"], "unavailable"),
            "grade_label": source_decision["state_label"],
            "primary_value": source_decision["primary_signal"].get("value"),
            "primary_label": "第一候选全局相似度",
            "decision": source_decision,
        },
    }

    open_set_notes: list[str] = []
    for key in ("identity", "source"):
        decision = modules[key]["decision"]
        if decision["state"] == STATE_NONE:
            open_set_notes.append(
                f"{'身份' if key == 'identity' else '素材'}检索："
                "当前候选低于拒识阈值，属于库外风险。"
            )
        elif decision["state"] == STATE_INSUFFICIENT:
            open_set_notes.append(
                f"{'身份' if key == 'identity' else '素材'}检索：候选落在不确定带内，需人工复核。"
            )
        elif decision["state"] == STATE_UNAVAILABLE:
            open_set_notes.append(
                f"{'身份' if key == 'identity' else '素材'}检索：本次无法完成比对，"
                "不能据此得出「库中无该对象」的结论。"
            )

    warnings: list[str] = []
    coverage = _library_coverage_block(identity, source)
    if not coverage["complete"]:
        warnings.append("身份库覆盖不足，低分不能解释为「库中无此人」。")

    return {
        "schema_version": SCHEMA_VERSION,
        "modules": modules,
        "input_quality": _input_quality_block(deepfake, identity, source),
        "library_coverage": coverage,
        "geometry": _geometry_block(source),
        "open_set_risk": {
            "level": "存在" if open_set_notes else "未发现",
            "notes": open_set_notes,
        },
        "warnings": warnings,
        "notes": [
            "本矩阵不提供跨模块综合概率：三个模块的数值性质不同，不能相加或合并。",
            "每一级结论都可以在对应模块结果文件中逐项复核。",
        ],
    }


__all__ = ["SCHEMA_VERSION", "build_evidence_matrix"]
