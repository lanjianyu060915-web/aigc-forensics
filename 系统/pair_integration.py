"""将冻结的配对实验用于候选对比，失败不改变原模块状态。"""

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from case_libraries import checked_reference


_COMPARATORS: dict[tuple[str, str, str], Any] = {}
_LOCK = threading.RLock()
MANUAL_NOTICE = "当前版本未给出伪造类型结论；配对图及数值仅供人工对比。"
SCOPE_NOTICE = "只核验素材检索的前三个候选，真实底图可能排在后面。"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_comparator(rule_path: Path, model_dir: Path, cache_dir: Path):
    from modules.pair_compare.pair_compare import PairComparator
    # 模型和规则各自保留实例；防止多个前端请求共用可变检测器。
    key = (_sha256(rule_path), str(model_dir.resolve()), str(cache_dir.resolve()))
    if key not in _COMPARATORS:
        os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
        _COMPARATORS.clear()
        _COMPARATORS[key] = PairComparator(rule_path, model_dir, cache_dir)
    return _COMPARATORS[key]


def _save_artifacts(directory: Path, artifacts: dict, result: dict) -> dict:
    from modules.pair_compare.pair_compare import save_artifacts
    return save_artifacts(directory, artifacts, result)


def _save_face_crops(directory: Path, query: Any, reference: Any, faces: list[dict]) -> dict:
    import cv2
    import numpy as np

    paths = {}
    for index, face in enumerate(faces):
        for label, picture, bbox_key in (("query", query.image, "query_bbox"),
                                         ("reference", reference.image, "reference_bbox")):
            box = face.get(bbox_key)
            if not isinstance(box, list) or len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, picture.shape[1]), min(y2, picture.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue
            ok, encoded = cv2.imencode(".png", np.ascontiguousarray(picture[y1:y2, x1:x2]))
            if ok:
                path = directory / f"face_{index}_{label}.png"
                encoded.tofile(path)
                paths[f"face_{index}_{label}"] = str(path)
    return paths


def _manual_observations(compared: Mapping[str, Any]) -> dict[str, Any]:
    """人工模式只发布测量和对应关系，不发布未经验收的实验分类。"""
    allowed = {
        "query_face_count", "reference_face_count", "alignment", "metrics",
        "query_sha256", "reference_sha256", "comparison_seconds", "uncached_pair_seconds",
        "preparation_seconds", "cache_hits", "elapsed_wall_seconds",
    }
    output = {key: value for key, value in compared.items() if key in allowed}
    alignment = compared.get("alignment") or {}
    output["alignment_status"] = (
        "failed" if compared.get("code") == "alignment_failed" else
        "completed" if alignment.get("homography") is not None else "unavailable")
    face_keys = {
        "query_face_id", "reference_face_id", "query_bbox", "reference_bbox",
        "query_bbox_original", "reference_bbox_original", "mapped_query_bbox",
        "match_iou", "cosine_similarity", "face_region", "surround_region",
        "query_det_score", "reference_det_score",
    }
    output["faces"] = []
    for face in compared.get("faces", []):
        row = {key: value for key, value in face.items() if key in face_keys}
        row["match_status"] = "matched" if row.get("reference_face_id") is not None and row.get("query_face_id") is not None else "unmatched"
        output["faces"].append(row)
    return output


def run_pair_comparison(*, query_path: str | Path, source_result: Mapping[str, Any],
                        task_dir: str | Path, source_library: str | Path,
                        rule_path: str | Path, model_dir: str | Path,
                        cache_dir: str | Path, scope: str = "image",
                        timestamp_seconds: float | None = None,
                        expected_rule_sha256: str | None = None,
                        record_clue_assessment: bool = False,
                        comparator: Any = None, artifact_writer: Any = None) -> dict[str, Any]:
    """完整处理前三个候选；现阶段只交付人工比对，不启用自动类型结论。"""
    started = time.perf_counter()
    task = Path(task_dir).resolve()
    result: dict[str, Any] = {
        "schema_version": "1.0", "status": "success", "scope": scope,
        "timestamp_seconds": timestamp_seconds, "conclusion_enabled": False,
        "mode": "manual_review", "notice": MANUAL_NOTICE,
        "limitations": [SCOPE_NOTICE], "candidates": [], "artifact_manifest": [],
        "rule_sha256": None, "runtime_seconds": 0.0,
    }
    if scope == "frame":
        result["limitations"].append("配对只代表首要帧，不是整段视频的结论。")
    candidates = list(source_result.get("candidates") or [])[:3]
    if not candidates or source_result.get("status") != "success":
        result.update(status="skipped", notice="本次没有可供配对核验的素材候选。")
        return result
    writer = artifact_writer or _save_artifacts
    try:
        with _LOCK:
            result["rule_sha256"] = _sha256(Path(rule_path))
            if expected_rule_sha256 and result["rule_sha256"] != expected_rule_sha256:
                raise ValueError("配对规则摘要与冻结记录不一致，已跳过配对。")
            engine = comparator or _load_comparator(Path(rule_path), Path(model_dir), Path(cache_dir))
            result["model_load_seconds"] = getattr(engine, "model_load_seconds", None)
            prepared_query = engine.prepare(Path(query_path))
            result["query_face_count"] = len(prepared_query.faces)
            for index, candidate in enumerate(candidates, 1):
                candidate = dict(candidate) if isinstance(candidate, Mapping) else {}
                row: dict[str, Any] = {
                    "rank": candidate.get("rank", index),
                    "candidate_path": candidate.get("file_path"),
                    "candidate_id": str(candidate.get("file_path") or index),
                    "status": "error", "artifacts": {}, "comparison": None,
                }
                try:
                    path = checked_reference(source_library, str(candidate.get("file_path") or ""))
                    if not path.is_file():
                        raise ValueError("候选不是可读取的图片文件。")
                    prepared_reference = engine.prepare(path)
                    compared, visual = engine.compare_prepared(prepared_query, prepared_reference,
                                                                 return_artifacts=True)
                    if record_clue_assessment:
                        # 新研判只留存原规则的真实输出；原人工测量视图、类型结论开关不变。
                        row["clue_assessment"] = {
                            "code": compared.get("code"), "reason": compared.get("reason"),
                            "rule_sha256": result["rule_sha256"], "usage": "review_floor_only",
                            "faces": [{key: face.get(key) for key in (
                                "query_face_id", "reference_face_id", "query_bbox_original", "reference_bbox_original",
                                "status", "reason", "cosine_similarity", "face_region", "surround_region")}
                                      for face in compared.get("faces", []) if isinstance(face, Mapping)],
                        }
                    compared = _manual_observations(compared)
                    # 对齐失败也保留输入并排图，不能把几何失败显示成无关图片。
                    visual.setdefault("query", prepared_query.image)
                    visual.setdefault("reference", prepared_reference.image)
                    directory = task / "pair_compare" / f"candidate_{index}"
                    directory.mkdir(parents=True, exist_ok=True)
                    artifacts = writer(directory, visual, compared)
                    artifacts.update(_save_face_crops(directory, prepared_query, prepared_reference,
                                                     compared.get("faces", [])))
                    relative = {}
                    for label, value in artifacts.items():
                        artifact_path = checked_reference(task, value)
                        relpath = artifact_path.relative_to(task).as_posix()
                        relative[label] = relpath
                        result["artifact_manifest"].append({
                            "path": relpath, "module": "source", "kind": "pair_compare",
                            "description": f"候选 {index} 的配对对比图", "status": "generated"})
                    detail_path = directory / "result.json"
                    if detail_path.is_file():
                        result["artifact_manifest"].append({
                            "path": detail_path.relative_to(task).as_posix(), "module": "source",
                            "kind": "pair_compare_detail", "description": f"候选 {index} 的配对数值",
                            "status": "generated"})
                    row.update(status="success", comparison=compared, artifacts=relative,
                               candidate_name=path.name, candidate_sha256=_sha256(path))
                except Exception as exc:
                    row["error_message"] = f"配对未完成：{exc}"
                result["candidates"].append(row)
    except Exception as exc:
        result.update(status="error", error_message=f"配对模块未完成：{exc}")
    completed = sum(row["status"] == "success" for row in result["candidates"])
    if result["status"] != "error" and completed < len(candidates):
        result["status"] = "partial_success" if completed else "error"
    result["runtime_seconds"] = round(time.perf_counter() - started, 6)
    return result


def pair_report_lines(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    lines = [MANUAL_NOTICE, SCOPE_NOTICE]
    from reporting.text_safety import redact_local_paths
    if value.get("status") in {"error", "failed", "partial_success"}:
        reason = value.get("error_message") or value.get("notice") or "部分候选未形成有效结果"
        lines.append("配对比对未完成：" + redact_local_paths(str(reason)))
    elif value.get("status") == "skipped":
        reason = value.get("notice") or value.get("error_message") or "本次未执行配对"
        lines.append("配对比对未运行：" + redact_local_paths(str(reason)))
    if value.get("scope") == "frame":
        lines.append(f"核验时间点：{value.get('timestamp_seconds')} 秒，只代表首要帧，不代表整段视频。")
    for row in value.get("candidates", []):
        comparison = row.get("comparison") or {}
        alignment = comparison.get("alignment") or {}
        # 尚未通过全部验收时，报告展示观测数值，不展示实验分类标签。
        name = row.get("candidate_name") or Path(str(row.get("candidate_path") or "未记录")).name
        status = "完成" if row.get("status") == "success" else "未完成"
        alignment_label = {"failed": "失败，暂不能判断是否同源", "completed": "完成", "unavailable": "未形成可比对的对齐"}.get(
            comparison.get("alignment_status"), "未记录")
        lines.append(f"候选 {row.get('rank')}：{name}；配对执行状态：{status}；"
                     f"对齐状态：{alignment_label}；对齐内点：{alignment.get('inliers', '未记录')}。")
    return lines
