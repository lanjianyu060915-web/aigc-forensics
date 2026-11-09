"""溯源升级的接线薄层：案件库快照、缓存隔离和可选人工配对。"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from case_libraries import case_source_cache, describe_case_libraries
from pair_integration import MANUAL_NOTICE, SCOPE_NOTICE, run_pair_comparison


PathResolver = Callable[[str | Path, Path], Path]
REGISTRY_ROOT = "{DATA_ROOT}/runtime/cache"
SOURCE_CACHE_ROOT = "{DATA_ROOT}/runtime/cache/source"


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _library_value(override: str | Path | None, section: Mapping[str, Any]) -> str | Path:
    value = override if override is not None else section.get("library")
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("本次比对库目录未配置或为空。")
    return value


def prepare_case_context(*, config: Mapping[str, Any], config_path: Path,
                         resolve_path: PathResolver,
                         source_library: str | Path | None = None,
                         identity_library: str | Path | None = None,
                         selected_modules: Collection[str] | None = None) -> dict[str, Any]:
    """返回独立的素材配置和库记录，不改变传入配置及任何判定阈值。

    调用方只把 ``case_libraries`` 写入报告；``source_section`` 用于执行
    素材检索。快照失败时禁用该副本中的缓存，检索仍由原有模块独立执行。
    """
    source_section = _section(config, "source")
    selected = ({"source", "identity"} if selected_modules is None else
                {selected_modules} if isinstance(selected_modules, str) else
                set(selected_modules))
    enabled = _section(config, "case_libraries").get("enabled") is True
    active = enabled and bool(selected & {"source", "identity"})
    context: dict[str, Any] = {
        "enabled": active, "source_section": source_section, "case_libraries": None,
    }
    if not active:
        return context
    try:
        source_root = resolve_path(_library_value(source_library, source_section), config_path)
        identity_root = resolve_path(
            _library_value(identity_library, _section(config, "identity")), config_path)
        metadata = describe_case_libraries(
            cache_root=resolve_path(REGISTRY_ROOT, config_path),
            source_root=source_root, identity_root=identity_root)
        metadata = deepcopy(metadata)
        metadata["status"] = "success"
        # 旧建库记录的时间只对匹配的内容有效，不能代表已经变更的当前库。
        libraries = metadata.get("libraries") or {}
        if not all(libraries.get(kind, {}).get("build_record_matches")
                   for kind in ("source", "identity")):
            metadata["built_at"] = None
        if "source" in selected:
            digest = libraries["source"]["manifest_sha256"]
            cache_root = resolve_path(source_section.get("feature_cache") or SOURCE_CACHE_ROOT,
                                      config_path)
            cache_path = case_source_cache(cache_root, source_root, digest)
            # 防止配置把缓存放入任一案内库，或已有链接把隔离目录导向库内。
            resolved_cache = cache_path.resolve(strict=False)
            if any(resolved_cache.is_relative_to(Path(root).resolve(strict=True))
                   for root in (source_root, identity_root)):
                raise ValueError("素材特征缓存目录不能放在案件库里面。")
            source_section["feature_cache"] = str(cache_path)
        context["case_libraries"] = metadata
    except Exception as exc:
        if "source" in selected:
            source_section["feature_cache"] = None
        context["case_libraries"] = {
            "status": "error", "name": "本次比对库", "built_at": None,
            "libraries": {},
            "error_message": f"案件库信息记录未完成：{type(exc).__name__}: {exc}",
        }
    return context


def _frame_timestamp(source_result: Mapping[str, Any]) -> float | None:
    frames = source_result.get("frame_evidence")
    if not isinstance(frames, list) or not frames or not isinstance(frames[0], Mapping):
        return None
    value = frames[0].get("timestamp_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _pair_error(scope: str, timestamp: float | None, error: Exception) -> dict[str, Any]:
    limitations = [SCOPE_NOTICE]
    if scope == "frame":
        limitations.append("配对只代表首要帧，不是整段视频的结论。")
    return {
        "schema_version": "1.0", "status": "error", "scope": scope,
        "timestamp_seconds": timestamp, "conclusion_enabled": False,
        "mode": "manual_review", "forgery_type": None, "notice": MANUAL_NOTICE,
        "limitations": limitations, "candidates": [], "artifact_manifest": [],
        "rule_sha256": None, "runtime_seconds": 0.0,
        "error_message": f"配对接入未完成：{type(error).__name__}: {error}",
    }


def _write_pair_result(task_dir: Path, result: Mapping[str, Any]) -> None:
    target = task_dir / "pair_compare_result.json"
    temporary = target.with_name(f".{target.stem}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def run_source_pair(*, config: Mapping[str, Any], config_path: Path,
                    resolve_path: PathResolver, task_dir: str | Path,
                    query_path: str | Path, source_result: Mapping[str, Any],
                    source_library: str | Path | None = None,
                    input_kind: str = "image") -> dict[str, Any] | None:
    """运行可选配对并留存结果；禁用时返回空值且不产生任何文件。

    配置键为 ``enabled``、``expected_rule_sha256``、``rule_path``、
    ``model_dir``、``cache_dir``。冻结摘要必填，其余路径有产品默认值。
    """
    section = _section(config, "pair_compare")
    if section.get("enabled") is not True:
        return None
    scope = "frame" if input_kind == "video" else "image"
    timestamp = _frame_timestamp(source_result) if scope == "frame" else None
    task = Path(task_dir)
    actual_rule_sha256 = None
    try:
        expected = section.get("expected_rule_sha256")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
            raise ValueError("必须配置有效的冻结规则 SHA-256，未执行配对。")
        rule_path = resolve_path(section.get("rule_path") or
                                 "{APP_ROOT}/modules/pair_compare/rule.json", config_path)
        actual_rule_sha256 = hashlib.sha256(rule_path.read_bytes()).hexdigest()
        if actual_rule_sha256 != expected.lower():
            raise ValueError("配对规则摘要与冻结记录不一致，未执行配对。")
        identity_section = _section(config, "identity")
        model_value = section.get("model_dir")
        if not model_value and identity_section.get("model_root"):
            model_value = resolve_path(identity_section["model_root"], config_path) / "models" / "buffalo_l"
        if not isinstance(model_value, (str, Path)) or not str(model_value).strip():
            raise ValueError("配对所需的人脸模型目录未配置。")
        result = run_pair_comparison(
            query_path=query_path, source_result=source_result, task_dir=task,
            source_library=resolve_path(
                _library_value(source_library, _section(config, "source")), config_path),
            rule_path=rule_path,
            model_dir=resolve_path(model_value, config_path),
            cache_dir=resolve_path(section.get("cache_dir") or
                                   "{DATA_ROOT}/runtime/cache/pair_compare", config_path),
            scope=scope, timestamp_seconds=timestamp, expected_rule_sha256=expected.lower(),
            record_clue_assessment=isinstance(config.get("llm"), Mapping))
        if not isinstance(result, Mapping):
            raise ValueError("配对模块没有返回结构化结果。")
        result = dict(result)
        result.update(conclusion_enabled=False, mode="manual_review", forgery_type=None,
                      scope=scope, timestamp_seconds=timestamp)
    except Exception as exc:
        result = _pair_error(scope, timestamp, exc)
        result["rule_sha256"] = actual_rule_sha256
    try:
        _write_pair_result(task, result)
    except Exception as exc:
        result["status"] = "error"
        result["persistence_error"] = f"配对结果保存失败：{type(exc).__name__}: {exc}"
    return result


def case_report_lines(metadata: Any) -> list[str]:
    """仅输出库名、数量和内容摘要，不把库的绝对路径写进报告正文。"""
    if not isinstance(metadata, Mapping):
        return []
    if metadata.get("status") == "error":
        return ["案件库信息记录未完成，详见任务日志。"]
    lines = [f"案件库：{metadata.get('name') or '本次比对库'}"]
    libraries = metadata.get("libraries") or {}
    for kind, label in (("source", "素材库"), ("identity", "身份库")):
        item = libraries.get(kind)
        if not isinstance(item, Mapping):
            continue
        built_at = item.get("built_at") if item.get("build_record_matches") else None
        lines.append(
            f"{label}：{item.get('name') or '未记录'}；文件数：{item.get('file_count', '未记录')}；"
            f"文件清单 SHA-256：{item.get('manifest_sha256') or '未记录'}；"
            f"建库时间：{built_at or '未记录与当前内容匹配的建库时间'}。")
    return lines
