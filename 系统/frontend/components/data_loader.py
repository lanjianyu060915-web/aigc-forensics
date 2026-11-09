from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal

from components.path_security import PathContainmentError, ResourcePathPolicy


ResultKind = Literal["identity", "source"]
ResultStatus = Literal["success", "error", "skipped"]
MAX_RESULT_BYTES = 5 * 1024 * 1024
MAX_REPORT_BYTES = 20 * 1024 * 1024

COMMON_FIELDS = {
    "status",
    "query_image",
    "candidates",
    "runtime_seconds",
    "error_message",
}
IDENTITY_FIELDS = {
    "rank",
    "identity_id",
    "identity_name",
    "score",
    "reference_image",
}
SOURCE_FIELDS = {
    "rank",
    "file_path",
    "global_score",
    "local_score",
    "final_score",
    "visualization_path",
}
DETECTION_FIELDS = {
    "status",
    "query_image",
    "is_deepfake",
    "score",
    "label",
    "runtime_seconds",
    "details",
    "error_message",
}
STRUCTURED_V3_FIELDS = {
    "method",
    "metrics",
    "runtime",
    "provenance",
    "visual_evidence",
}
NON_MATERIALIZED_ARTIFACT_STATUSES = {
    "not_generated",
    "unavailable",
    "skipped",
    "failed",
    "error",
    "missing",
}


class ResultDataError(ValueError):
    """A result file is missing, malformed, unsafe, or violates its schema."""


@dataclass(frozen=True)
class TaskResults:
    task_dir: Path
    deepfake_result: dict[str, Any]
    identity_result: dict[str, Any]
    source_result: dict[str, Any]
    path_policy: ResourcePathPolicy
    legacy_dual_module: bool = False
    task_status: dict[str, Any] | None = None
    unified_result: dict[str, Any] | None = None
    final_report: dict[str, Any] | None = None
    artifact_manifest: dict[str, Any] | None = None


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResultDataError(f"{field} 必须是非空字符串。")
    return value


def _require_number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultDataError(f"{field} 必须是有限数值。")
    number = float(value)
    if not math.isfinite(number):
        raise ResultDataError(f"{field} 必须是有限数值。")
    if nonnegative and number < 0:
        raise ResultDataError(f"{field} 不能为负数。")
    return number


def _require_rank(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ResultDataError(f"{field} 必须是从 1 开始的正整数。")
    return value


def _read_json(json_path: Path, *, max_bytes: int = MAX_RESULT_BYTES) -> dict[str, Any]:
    path = Path(json_path)
    if not path.is_file():
        raise ResultDataError(f"结果文件不存在：{path}")
    try:
        if path.stat().st_size > max_bytes:
            raise ResultDataError(
                f"结果文件超过 {max_bytes // (1024 * 1024)} MiB 安全上限：{path}"
            )
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except UnicodeDecodeError as exc:
        raise ResultDataError("结果文件必须使用 UTF-8 编码。") from exc
    except json.JSONDecodeError as exc:
        raise ResultDataError(
            f"JSON 格式错误：第 {exc.lineno} 行，第 {exc.colno} 列。"
        ) from exc
    except OSError as exc:
        raise ResultDataError(f"无法读取结果文件：{exc}") from exc
    if not isinstance(data, dict):
        raise ResultDataError("JSON 顶层必须是对象。")
    return data


def _read_optional_json(
    json_path: Path, *, max_bytes: int = MAX_RESULT_BYTES
) -> dict[str, Any] | None:
    return _read_json(json_path, max_bytes=max_bytes) if Path(json_path).is_file() else None


def _validate_v3_extensions(data: dict[str, Any], *, label: str) -> None:
    """Validate optional v3 containers while retaining the v2 field contract."""

    schema_version = data.get("schema_version")
    if schema_version is not None and not isinstance(schema_version, str):
        raise ResultDataError(f"{label}.schema_version 必须是字符串。")
    for field in STRUCTURED_V3_FIELDS:
        value = data.get(field)
        if value is not None and not isinstance(value, dict):
            raise ResultDataError(f"{label}.{field} 必须是对象。")
    explanation = data.get("explanation")
    if explanation is not None and not (
        isinstance(explanation, dict)
        or (
            isinstance(explanation, list)
            and all(isinstance(item, str) for item in explanation)
        )
    ):
        raise ResultDataError(f"{label}.explanation 必须是对象或字符串数组。")
    for field in ("warnings", "limitations"):
        value = data.get(field)
        if value is None:
            continue
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ResultDataError(f"{label}.{field} 必须是字符串数组。")


def _validate_artifact_manifest(
    data: dict[str, Any],
    path_policy: ResourcePathPolicy,
    *,
    label: str,
) -> None:
    """Validate optional v0.4 task-relative evidence records.

    Older tasks legitimately omit the manifest.  New entries may point only
    inside the approved task/sample visualization roots; missing files remain
    loadable so the UI can present an explicit non-fatal evidence warning.
    """

    manifest = data.get("artifact_manifest")
    if manifest is None:
        visual = data.get("visual_evidence")
        manifest = visual.get("artifact_manifest") if isinstance(visual, dict) else None
    if manifest is None:
        return
    if not isinstance(manifest, list):
        raise ResultDataError(f"{label}.artifact_manifest 必须是数组。")
    for index, item in enumerate(manifest):
        prefix = f"{label}.artifact_manifest[{index}]"
        if not isinstance(item, dict):
            raise ResultDataError(f"{prefix} 必须是对象。")
        path_value = item.get("path") or item.get("relative_path")
        if not isinstance(path_value, str) or not path_value.strip():
            status = str(item.get("status") or "").strip().casefold()
            if status not in NON_MATERIALIZED_ARTIFACT_STATUSES:
                raise ResultDataError(
                    f"{prefix}.path 为空时，status 必须明确表示该证据未生成或不可用。"
                )
        else:
            try:
                path_policy.resolve(path_value, "visualization", label=f"{prefix}.path")
            except PathContainmentError as exc:
                raise ResultDataError(str(exc)) from exc
        sha256 = item.get("sha256")
        if sha256 is not None and (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in sha256)
        ):
            raise ResultDataError(f"{prefix}.sha256 必须是64位十六进制字符串。")
        size_bytes = item.get("size_bytes")
        if size_bytes is not None and (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise ResultDataError(f"{prefix}.size_bytes 必须是非负整数。")


def _validate_common(data: dict[str, Any], path_policy: ResourcePathPolicy) -> None:
    missing = COMMON_FIELDS - data.keys()
    if missing:
        raise ResultDataError(f"JSON 缺少字段：{'、'.join(sorted(missing))}")

    status = data["status"]
    if status not in {"success", "error", "skipped"}:
        raise ResultDataError("status 只允许 'success'、'error' 或 'skipped'。")
    query_image = _require_nonempty_string(data["query_image"], "query_image")
    if not isinstance(data["candidates"], list):
        raise ResultDataError("candidates 必须是数组。")
    _require_number(data["runtime_seconds"], "runtime_seconds", nonnegative=True)

    error_message = data["error_message"]
    if error_message is not None and not isinstance(error_message, str):
        raise ResultDataError("error_message 必须是字符串或 null。")
    if status == "error" and (not isinstance(error_message, str) or not error_message.strip()):
        raise ResultDataError("status=error 时 error_message 必须是非空字符串。")
    if status == "error" and data["candidates"]:
        raise ResultDataError("status=error 时 candidates 必须为空数组。")
    if status == "skipped" and data["candidates"]:
        raise ResultDataError("status=skipped 时 candidates 必须为空数组。")

    try:
        path_policy.resolve(query_image, "query", label="query_image")
    except PathContainmentError as exc:
        raise ResultDataError(str(exc)) from exc
    _validate_v3_extensions(data, label="result")
    _validate_artifact_manifest(data, path_policy, label="result")


def _validate_ranks(candidates: list[dict[str, Any]]) -> None:
    ranks = [candidate["rank"] for candidate in candidates]
    expected = list(range(1, len(candidates) + 1))
    if ranks != expected:
        raise ResultDataError(
            f"候选 rank 必须按数组顺序连续为 1..N；实际为 {ranks}。"
        )


def _validate_identity_candidate(
    candidate: Any,
    index: int,
    path_policy: ResourcePathPolicy,
) -> dict[str, Any]:
    prefix = f"candidates[{index}]"
    if not isinstance(candidate, dict):
        raise ResultDataError(f"{prefix} 必须是对象。")
    missing = IDENTITY_FIELDS - candidate.keys()
    if missing:
        raise ResultDataError(f"{prefix} 缺少字段：{'、'.join(sorted(missing))}")
    _require_rank(candidate["rank"], f"{prefix}.rank")
    _require_nonempty_string(candidate["identity_id"], f"{prefix}.identity_id")
    _require_nonempty_string(candidate["identity_name"], f"{prefix}.identity_name")
    _require_number(candidate["score"], f"{prefix}.score")
    path_value = _require_nonempty_string(
        candidate["reference_image"], f"{prefix}.reference_image"
    )
    try:
        path_policy.resolve(path_value, "identity", label=f"{prefix}.reference_image")
        best_reference = candidate.get("best_reference_image")
        if best_reference is not None:
            best_reference = _require_nonempty_string(
                best_reference, f"{prefix}.best_reference_image"
            )
            path_policy.resolve(
                best_reference,
                "identity",
                label=f"{prefix}.best_reference_image",
            )
    except PathContainmentError as exc:
        raise ResultDataError(str(exc)) from exc
    for count_field in ("reference_count",):
        value = candidate.get(count_field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ResultDataError(f"{prefix}.{count_field} 必须是非负整数。")
    for score_field in ("mean_similarity", "best_reference_score"):
        if candidate.get(score_field) is not None:
            _require_number(candidate[score_field], f"{prefix}.{score_field}")
    reference_scores = candidate.get("reference_scores")
    if reference_scores is not None:
        if not isinstance(reference_scores, list):
            raise ResultDataError(f"{prefix}.reference_scores 必须是数组。")
        for reference_index, item in enumerate(reference_scores):
            item_prefix = f"{prefix}.reference_scores[{reference_index}]"
            if not isinstance(item, dict):
                raise ResultDataError(f"{item_prefix} 必须是对象。")
            reference_path = _require_nonempty_string(
                item.get("reference_image"), f"{item_prefix}.reference_image"
            )
            _require_number(item.get("similarity"), f"{item_prefix}.similarity")
            try:
                path_policy.resolve(
                    reference_path,
                    "identity",
                    label=f"{item_prefix}.reference_image",
                )
            except PathContainmentError as exc:
                raise ResultDataError(str(exc)) from exc
    return candidate


def _validate_source_candidate(
    candidate: Any,
    index: int,
    path_policy: ResourcePathPolicy,
) -> dict[str, Any]:
    prefix = f"candidates[{index}]"
    if not isinstance(candidate, dict):
        raise ResultDataError(f"{prefix} 必须是对象。")
    missing = SOURCE_FIELDS - candidate.keys()
    if missing:
        raise ResultDataError(f"{prefix} 缺少字段：{'、'.join(sorted(missing))}")
    _require_rank(candidate["rank"], f"{prefix}.rank")
    _require_nonempty_string(candidate["file_path"], f"{prefix}.file_path")
    _require_nonempty_string(
        candidate["visualization_path"], f"{prefix}.visualization_path"
    )
    for score_name in ("global_score", "local_score", "final_score"):
        _require_number(candidate[score_name], f"{prefix}.{score_name}")
    try:
        path_policy.resolve(
            candidate["file_path"], "source", label=f"{prefix}.file_path"
        )
        path_policy.resolve(
            candidate["visualization_path"],
            "visualization",
            label=f"{prefix}.visualization_path",
        )
    except PathContainmentError as exc:
        raise ResultDataError(str(exc)) from exc
    for count_field in (
        "query_keypoint_count",
        "candidate_keypoint_count",
        "good_match_count",
        "ransac_inlier_count",
    ):
        value = candidate.get(count_field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ResultDataError(f"{prefix}.{count_field} 必须是非负整数。")
    for score_field in (
        "ransac_inlier_ratio",
        "lowe_ratio",
        "ransac_threshold",
        "global_weight",
        "local_weight",
    ):
        if candidate.get(score_field) is not None:
            _require_number(candidate[score_field], f"{prefix}.{score_field}")
    components = candidate.get("local_score_components")
    if components is not None and not isinstance(components, dict):
        raise ResultDataError(f"{prefix}.local_score_components 必须是对象。")
    return candidate


def _validate_source_extensions(
    data: dict[str, Any], path_policy: ResourcePathPolicy
) -> None:
    """Validate persisted source visual paths without trusting v3 diagnostics."""

    for container_name in ("metrics", "diagnostics"):
        container = data.get(container_name)
        if not isinstance(container, dict):
            continue
        masked_query_path = container.get("masked_query_path")
        if masked_query_path is None:
            continue
        value = _require_nonempty_string(
            masked_query_path, f"{container_name}.masked_query_path"
        )
        try:
            path_policy.resolve(
                value,
                "visualization",
                label=f"{container_name}.masked_query_path",
            )
        except PathContainmentError as exc:
            raise ResultDataError(str(exc)) from exc


def load_result(
    json_path: Path,
    kind: ResultKind,
    path_policy: ResourcePathPolicy,
) -> dict[str, Any]:
    data = _read_json(json_path)
    return validate_result_data(data, kind, path_policy)


def validate_result_data(
    data: dict[str, Any],
    kind: ResultKind,
    path_policy: ResourcePathPolicy,
) -> dict[str, Any]:
    """Validate an already-decoded payload without weakening file loading."""

    if not isinstance(data, dict):
        raise ResultDataError("JSON 顶层必须是对象。")
    _validate_common(data, path_policy)
    candidates = data["candidates"]
    if kind == "identity":
        for index, candidate in enumerate(candidates):
            _validate_identity_candidate(candidate, index, path_policy)
    elif kind == "source":
        for index, candidate in enumerate(candidates):
            _validate_source_candidate(candidate, index, path_policy)
        _validate_source_extensions(data, path_policy)
    else:
        raise ResultDataError(f"未知结果类型：{kind}")
    _validate_ranks(candidates)
    return data


def load_identity_result(
    json_path: Path,
    path_policy: ResourcePathPolicy,
) -> dict[str, Any]:
    return load_result(json_path, "identity", path_policy)


def load_source_result(
    json_path: Path,
    path_policy: ResourcePathPolicy,
) -> dict[str, Any]:
    return load_result(json_path, "source", path_policy)


def validate_detection_data(
    data: dict[str, Any],
    path_policy: ResourcePathPolicy,
) -> dict[str, Any]:
    """Validate the AIGC/deepfake result without changing native uncertainty.

    ``LOW_CONFIDENCE`` and ``INSUFFICIENT_EVIDENCE`` are represented by a
    successful module call with ``is_deepfake=null``.  They intentionally stay
    uncertain here; the loader never maps them to ``real``.
    """

    if not isinstance(data, dict):
        raise ResultDataError("鉴伪 JSON 顶层必须是对象。")
    missing = DETECTION_FIELDS - data.keys()
    if missing:
        raise ResultDataError(f"鉴伪 JSON 缺少字段：{'、'.join(sorted(missing))}")

    status = data["status"]
    if status not in {"success", "error", "skipped"}:
        raise ResultDataError("鉴伪 status 只允许 'success'、'error' 或 'skipped'。")

    query_image = _require_nonempty_string(data["query_image"], "query_image")
    try:
        path_policy.resolve(query_image, "query", label="query_image")
    except PathContainmentError as exc:
        raise ResultDataError(str(exc)) from exc

    is_deepfake = data["is_deepfake"]
    if is_deepfake is not None and not isinstance(is_deepfake, bool):
        raise ResultDataError("is_deepfake 必须是布尔值或 null。")

    score = data["score"]
    if score is not None:
        number = _require_number(score, "score")
        if not 0.0 <= number <= 1.0:
            raise ResultDataError("score 必须在 0 到 1 之间。")

    label = data["label"]
    if label is not None and not isinstance(label, str):
        raise ResultDataError("label 必须是字符串或 null。")
    _require_number(data["runtime_seconds"], "runtime_seconds", nonnegative=True)
    if not isinstance(data["details"], dict):
        raise ResultDataError("details 必须是对象。")

    error_message = data["error_message"]
    if error_message is not None and not isinstance(error_message, str):
        raise ResultDataError("鉴伪 error_message 必须是字符串或 null。")
    if status == "success" and error_message is not None:
        raise ResultDataError("鉴伪 status=success 时 error_message 必须为 null。")
    if status == "error" and (
        not isinstance(error_message, str) or not error_message.strip()
    ):
        raise ResultDataError("鉴伪 status=error 时 error_message 必须是非空字符串。")
    if status == "skipped":
        if is_deepfake is not None or score is not None:
            raise ResultDataError(
                "鉴伪 status=skipped 时 is_deepfake 和 score 必须为 null。"
            )
        if label not in {None, "skipped"}:
            raise ResultDataError("鉴伪 status=skipped 时 label 只能为 null 或 'skipped'。")
    _validate_v3_extensions(data, label="detection_result")
    _validate_artifact_manifest(data, path_policy, label="detection_result")
    return data


def load_detection_result(
    json_path: Path,
    path_policy: ResourcePathPolicy,
) -> dict[str, Any]:
    return validate_detection_data(_read_json(json_path), path_policy)


def skipped_detection_result(query_image: str, reason: str) -> dict[str, Any]:
    """Create the explicit compatibility marker for an old two-module task."""

    return {
        "status": "skipped",
        "query_image": query_image,
        "is_deepfake": None,
        "score": None,
        "label": "skipped",
        "runtime_seconds": 0.0,
        "details": {"reason": reason, "compatibility_mode": "legacy_dual_module"},
        "error_message": None,
    }


def load_task_results(
    task_dir: Path,
    *,
    identity_roots: tuple[Path, ...] = (),
    source_roots: tuple[Path, ...] = (),
    sample_roots: tuple[Path, ...] = (),
) -> TaskResults:
    task_path = Path(task_dir).expanduser().resolve(strict=False)
    if not task_path.is_dir():
        raise ResultDataError(f"task_dir 不存在或不是目录：{task_path}")

    policy = ResourcePathPolicy(
        task_roots=(task_path,),
        identity_roots=identity_roots,
        source_roots=source_roots,
        sample_roots=sample_roots,
    )
    identity = load_identity_result(task_path / "identity_result.json", policy)
    source = load_source_result(task_path / "source_result.json", policy)
    detection_path = task_path / "detection_result.json"
    legacy_dual_module = not detection_path.is_file()
    if legacy_dual_module:
        query_image = str(identity.get("query_image") or source.get("query_image") or "")
        deepfake = skipped_detection_result(
            query_image,
            "旧版双模块 task 未包含 detection_result.json。",
        )
        validate_detection_data(deepfake, policy)
    else:
        deepfake = load_detection_result(detection_path, policy)
    task_status = _read_optional_json(task_path / "task_status.json")
    unified_result = _read_optional_json(task_path / "unified_result.json")
    final_report = _read_optional_json(
        task_path / "final_report.json", max_bytes=MAX_REPORT_BYTES
    )
    if final_report is not None:
        if str(final_report.get("schema_version") or "") != "3.0":
            raise ResultDataError("final_report.json schema_version 必须为 '3.0'。")
        if not isinstance(final_report.get("task"), dict):
            raise ResultDataError("final_report.json 缺少 task 对象。")
    artifact_manifest = _read_optional_json(task_path / "artifact_manifest.json")
    if artifact_manifest is not None:
        artifacts = artifact_manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise ResultDataError("artifact_manifest.json 缺少 artifacts 数组。")
        for index, item in enumerate(artifacts):
            if not isinstance(item, dict):
                raise ResultDataError(f"artifact_manifest.artifacts[{index}] 必须是对象。")
            path_value = _require_nonempty_string(
                item.get("path"), f"artifact_manifest.artifacts[{index}].path"
            )
            try:
                policy.resolve(
                    path_value,
                    "visualization",
                    label=f"artifact_manifest.artifacts[{index}].path",
                )
            except PathContainmentError as exc:
                raise ResultDataError(str(exc)) from exc
    return TaskResults(
        task_dir=task_path,
        deepfake_result=deepfake,
        identity_result=identity,
        source_result=source,
        path_policy=policy,
        legacy_dual_module=legacy_dual_module,
        task_status=task_status,
        unified_result=unified_result,
        final_report=final_report,
        artifact_manifest=artifact_manifest,
    )
