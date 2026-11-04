"""Build one auditable report model and render JSON, TXT, and HTML from it."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
from typing import Any
import uuid

from .explainability import build_explanations, display_name, redact_text
from decision.evidence_matrix import build_evidence_matrix


DEFAULT_PRODUCT_NAME = "AI换脸涉案图像智能鉴伪与原始素材溯源系统"
DEFAULT_DISCLAIMER = (
    "本系统用于深度伪造风险筛查、身份候选检索和底图候选检索，仅提供侦查线索和辅助研判信息，"
    "不替代原始介质固定、哈希校验、人工复核或司法鉴定。"
)
_SAFE_COMPONENT_RE = re.compile(r"[^0-9A-Za-z._\-\u3400-\u9fff]+")
_MAX_REPORT_BYTES = 20 * 1024 * 1024


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _lookup(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if data.get(key) is not None:
            return data[key]
    for container_name in ("metrics", "details", "diagnostics", "runtime", "provenance"):
        container = data.get(container_name)
        if isinstance(container, Mapping):
            for key in keys:
                if container.get(key) is not None:
                    return container[key]
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 6) -> str:
    number = _number(value)
    return "未记录" if number is None else f"{number:.{digits}f}"


def _text(value: Any, fallback: str = "未记录") -> str:
    return str(value).strip() if value is not None and str(value).strip() else fallback


def _yes_no(value: Any) -> str:
    if value is True:
        return "是"
    if value is False:
        return "否"
    return "未记录"


def _structured_text(value: Any) -> str:
    if isinstance(value, Mapping):
        if not value:
            return _text(None)
        return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        if not value:
            return _text(None)
        return json.dumps(_json_value(value), ensure_ascii=False)
    return _text(value)


def _safe_component(value: Any) -> str:
    rendered = _SAFE_COMPONENT_RE.sub("_", str(value or "task")).strip(" ._")
    return (rendered or "task")[:96]


def report_file_names(task_id: Any) -> dict[str, str]:
    """Return the only report filenames permitted inside a task directory."""

    safe_id = _safe_component(task_id)
    return {
        "json": "final_report.json",
        "txt": f"检测结果_{safe_id}.txt",
        "html": f"检测结果_{safe_id}.html",
    }


def _selected_modules(
    task_status: Mapping[str, Any],
    unified_result: Mapping[str, Any],
    raw_results: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    raw = task_status.get("selected_modules") or unified_result.get("selected_modules")
    if isinstance(raw, list):
        selected = [name for name in ("deepfake", "identity", "source") if name in raw]
        if selected:
            return selected
    return [
        name
        for name in ("deepfake", "identity", "source")
        if str(raw_results[name].get("status") or "skipped") != "skipped"
    ]


def _derived_status(raw_results: Mapping[str, Mapping[str, Any]]) -> str:
    statuses = [str(raw_results[name].get("status") or "error") for name in ("deepfake", "identity", "source")]
    ran = [status for status in statuses if status != "skipped"]
    if not ran:
        return "skipped"
    successes = sum(status == "success" for status in ran)
    errors = sum(status == "error" for status in ran)
    if successes and errors:
        return "partial_success"
    if errors:
        return "error"
    return "success"


def _module_runtime(
    name: str,
    task_status: Mapping[str, Any],
    unified_result: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> float | None:
    metadata = task_status.get(name)
    if isinstance(metadata, Mapping):
        runtime = _number(metadata.get("runtime_seconds"))
        if runtime is not None:
            return runtime
    unified_runtimes = unified_result.get("runtime_seconds")
    if isinstance(unified_runtimes, Mapping):
        runtime = _number(unified_runtimes.get(name))
        if runtime is not None:
            return runtime
    return _number(raw_result.get("runtime_seconds"))


def _query_value(raw_results: Mapping[str, Mapping[str, Any]], task_status: Mapping[str, Any]) -> Any:
    for source in (
        task_status.get("query_image"),
        raw_results["deepfake"].get("query_image"),
        raw_results["identity"].get("query_image"),
        raw_results["source"].get("query_image"),
    ):
        if source:
            return source
    return None


def _image_metadata(
    raw_results: Mapping[str, Mapping[str, Any]],
    computed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sources = [raw_results["deepfake"], raw_results["identity"], raw_results["source"]]
    width = height = image_format = input_sha256 = None
    for source in sources:
        width = width or _lookup(source, "image_width", "width")
        height = height or _lookup(source, "image_height", "height")
        image_format = image_format or _lookup(source, "image_format", "format", "input_format")
        input_sha256 = input_sha256 or _lookup(source, "input_sha256", "input_file_sha256", "query_sha256")
        size = _lookup(source, "image_size", "dimensions")
        if isinstance(size, Mapping):
            width = width or size.get("width")
            height = height or size.get("height")
        elif isinstance(size, (list, tuple)) and len(size) >= 2:
            width = width or size[0]
            height = height or size[1]
        for container_name in ("metrics", "details"):
            container = source.get(container_name)
            if not isinstance(container, Mapping):
                continue
            media = container.get("media_metadata")
            if isinstance(media, Mapping):
                width = width or media.get("width")
                height = height or media.get("height")
                image_format = image_format or media.get("format")
    computed_data = _mapping(computed)
    computed_sha256 = computed_data.get("input_sha256")
    sha256_mismatch = bool(
        input_sha256
        and computed_sha256
        and str(input_sha256).lower() != str(computed_sha256).lower()
    )
    return {
        "input_sha256": input_sha256 or computed_sha256,
        "width": width or computed_data.get("width"),
        "height": height or computed_data.get("height"),
        "format": image_format or computed_data.get("format"),
        "sha256_source": "module_result" if input_sha256 else computed_data.get("sha256_source"),
        "computed_input_sha256": computed_sha256,
        "sha256_mismatch": sha256_mismatch,
    }


def _task_input_metadata(task_root: Path, query_value: Any) -> dict[str, Any]:
    """Hash and inspect only the task-contained query copy when available."""

    candidates: list[Path] = []
    if isinstance(query_value, str) and query_value.strip():
        supplied = Path(query_value).expanduser()
        candidates.append(supplied if supplied.is_absolute() else task_root / supplied)
    candidates.append(task_root / "query.jpg")
    candidates.append(task_root / "query.png")
    root = task_root.resolve(strict=False)
    query_path: Path | None = None
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.is_file():
            query_path = resolved
            break
    if query_path is None:
        return {}
    digest = hashlib.sha256()
    try:
        with query_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        return {"metadata_warning": f"任务查询文件哈希读取失败：{type(exc).__name__}"}
    metadata: dict[str, Any] = {
        "input_sha256": digest.hexdigest(),
        "sha256_source": "task_query_file",
        "format": query_path.suffix.lstrip(".").upper() or None,
    }
    try:
        from PIL import Image

        with Image.open(query_path) as image:
            metadata.update(width=image.width, height=image.height, format=image.format)
    except (ImportError, OSError, ValueError):
        # Hashing remains available even when optional image metadata parsing is not.
        pass
    return metadata


def _video_input(status_data: Mapping[str, Any]) -> dict[str, Any] | None:
    """视频任务：把视频文件、哈希与关键帧清单原样带进报告输入段。"""

    status_input = _mapping(status_data.get("input"))
    if str(status_input.get("type") or "") != "video":
        return None
    frames = status_input.get("key_frames")
    frames = frames if isinstance(frames, list) else []
    return {
        "video_file": status_input.get("video_file"),
        "video_sha256": status_input.get("video_sha256"),
        "frame_selection": status_input.get("frame_selection"),
        "key_frame_count": len(frames),
        "primary_frame": status_input.get("primary_frame"),
        "key_frames": [
            {
                "frame_rank": item.get("frame_rank"),
                "path": item.get("path"),
                "timestamp_seconds": item.get("timestamp_seconds"),
                "quality_score": item.get("quality_score"),
                "face_count": item.get("face_count"),
                "sha256": item.get("sha256"),
            }
            for item in frames
            if isinstance(item, Mapping)
        ],
    }


def build_final_report(
    task_id: str,
    *,
    task_status: Mapping[str, Any] | None = None,
    deepfake_result: Mapping[str, Any] | None = None,
    identity_result: Mapping[str, Any] | None = None,
    source_result: Mapping[str, Any] | None = None,
    unified_result: Mapping[str, Any] | None = None,
    product_name: str = DEFAULT_PRODUCT_NAME,
    product_version: str = "0.3",
    generated_at: str | None = None,
    input_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the single structured source used by every report renderer."""

    status_data = _mapping(task_status)
    unified = _mapping(unified_result)
    raw_results = {
        "deepfake": _mapping(deepfake_result),
        "identity": _mapping(identity_result),
        "source": _mapping(source_result),
    }
    explanations = build_explanations(
        raw_results["deepfake"], raw_results["identity"], raw_results["source"]
    )
    created_at = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    query_value = _query_value(raw_results, status_data)
    selected = _selected_modules(status_data, unified, raw_results)
    overall_status = str(
        status_data.get("status")
        or unified.get("status")
        or _derived_status(raw_results)
    )
    image_metadata = _image_metadata(raw_results, input_metadata)
    runtimes = {
        name: _module_runtime(name, status_data, unified, raw_results[name])
        for name in ("deepfake", "identity", "source")
    }
    total_runtime = _number(status_data.get("runtime_seconds"))
    if total_runtime is None:
        total_runtime = _number(_lookup(unified, "total_runtime_seconds"))
    if total_runtime is None and isinstance(unified.get("runtime_seconds"), Mapping):
        total_runtime = _number(unified["runtime_seconds"].get("total"))
    if total_runtime is None and all(value is not None for value in runtimes.values()):
        total_runtime = sum(value for value in runtimes.values() if value is not None)
    video_input = _video_input(status_data)

    report = {
        "schema_version": "3.0",
        "report_type": "face_source_trace_detection_report",
        "product": {"name": product_name, "version": str(product_version)},
        "task": {
            "task_id": str(task_id),
            "status": overall_status,
            "started_at": status_data.get("started_at"),
            "finished_at": status_data.get("finished_at") or created_at,
            "report_generated_at": created_at,
            "selected_modules": selected,
            "runtime_seconds": {**runtimes, "total": total_runtime},
        },
        "input": {
            "file_name": display_name(query_value),
            "query_image": query_value,
            **image_metadata,
            **({"video": video_input} if video_input else {}),
        },
        "modules": {
            name: {
                "status": str(raw_results[name].get("status") or "error"),
                "result": deepcopy(_json_value(raw_results[name])),
                "explanation": deepcopy(_json_value(explanations[name])),
            }
            for name in ("deepfake", "identity", "source")
        },
        "summary": {
            "deepfake": explanations["deepfake"]["summary"],
            "identity": explanations["identity"]["summary"],
            "source": explanations["source"]["summary"],
            "cross_module_fusion": None,
            "note": "三类证据并列呈现，不计算跨模块综合概率。",
        },
        "disclaimer": DEFAULT_DISCLAIMER,
        "provenance": {
            "task_status": deepcopy(_json_value(status_data)),
            "unified_result": deepcopy(_json_value(unified)),
            "report_generator": "deterministic_template_v3",
            "external_language_model_used": False,
        },
    }
    return _json_value(report)


def _append_list(lines: list[str], values: Any, *, prefix: str = "- ") -> None:
    if not isinstance(values, list) or not values:
        lines.append(prefix + "无")
        return
    for value in values:
        lines.append(prefix + redact_text(value))


def _module(report: Mapping[str, Any], name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    block = _mapping(_mapping(report.get("modules")).get(name))
    return _mapping(block.get("result")), _mapping(block.get("explanation"))


def _evidence_matrix_lines(
    deepfake: Mapping[str, Any],
    identity: Mapping[str, Any],
    source: Mapping[str, Any],
) -> list[str]:
    """构造证据质量矩阵章节；只做逐模块分级，不合并跨模块数值。"""

    matrix = build_evidence_matrix(deepfake, identity, source)
    module_labels = {"deepfake": "AIGC鉴伪", "identity": "人脸身份检索", "source": "原始底图溯源"}
    lines = [
        "",
        "五、证据质量矩阵（研判卡）",
        "说明：本矩阵只做逐模块证据分级，不做跨模块数值相加，不生成综合可信概率。",
    ]
    for module_key, module_label in module_labels.items():
        block = _mapping(_mapping(matrix.get("modules")).get(module_key))
        lines.append(
            f"- {module_label}：{_text(block.get('grade_label'))}"
            f"（{_text(block.get('primary_label'))}：{_fmt(block.get('primary_value'))}）"
        )
    quality = _mapping(matrix.get("input_quality"))
    coverage = _mapping(matrix.get("library_coverage"))
    geometry = _mapping(matrix.get("geometry"))
    open_set = _mapping(matrix.get("open_set_risk"))
    lines.append(f"- 查询图像质量：{_text(quality.get('grade'))}")
    lines.append(
        "- 身份库覆盖："
        f"{'已确认' if coverage.get('complete') else '覆盖不足'}"
        f"（{_text(coverage.get('identity_identities'))} 个身份 / "
        f"{_text(coverage.get('identity_valid_embeddings'))} 条有效特征）"
    )
    lines.append(
        f"- 几何一致性：{_text(geometry.get('grade'))}"
        f"（RANSAC 内点 {_text(geometry.get('inlier_count'))}）"
    )
    lines.append(f"- 开放集风险：{_text(open_set.get('level'))}")
    for note in open_set.get("notes") or []:
        lines.append(f"  - {redact_text(note)}")
    return lines


def render_text_report(report: Mapping[str, Any]) -> str:
    """Render a UTF-8-friendly Chinese report without ordinary-view full paths."""

    product = _mapping(report.get("product"))
    task = _mapping(report.get("task"))
    input_data = _mapping(report.get("input"))
    runtimes = _mapping(task.get("runtime_seconds"))
    selected = task.get("selected_modules") if isinstance(task.get("selected_modules"), list) else []
    lines = [
        _text(product.get("name"), DEFAULT_PRODUCT_NAME),
        "统一检测与检索结果报告",
        "=" * 34,
        "",
        "一、基本信息",
        f"产品版本：{_text(product.get('version'))}",
        f"任务编号：{_text(task.get('task_id'))}",
        f"检测时间：{_text(task.get('finished_at'))}",
        f"任务状态：{_text(task.get('status'))}",
        f"查询文件：{display_name(input_data.get('file_name') or input_data.get('query_image'))}",
        f"输入文件SHA-256：{_text(input_data.get('input_sha256'))}",
        f"输入哈希来源：{_text(input_data.get('sha256_source'))}",
        f"输入哈希校验：{'不一致，需人工复核' if input_data.get('sha256_mismatch') else '未发现不一致'}",
        f"图片尺寸：{_text(input_data.get('width'))} × {_text(input_data.get('height'))}",
        f"图片格式：{_text(input_data.get('format'))}",
        f"选择模块：{'、'.join(str(value) for value in selected) or '未记录'}",
        f"总耗时：{_fmt(runtimes.get('total'))} 秒",
        f"AIGC模块耗时：{_fmt(runtimes.get('deepfake'))} 秒",
        f"身份搜索耗时：{_fmt(runtimes.get('identity'))} 秒",
        f"底图搜索耗时：{_fmt(runtimes.get('source'))} 秒",
    ]
    video_input = _mapping(input_data.get("video"))
    if video_input:
        frame_count = _text(video_input.get("key_frame_count"))
        video_file = display_name(video_input.get("video_file"))
        video_digest = _text(video_input.get("video_sha256"))
        frame_evidence = _text(video_input.get("frame_selection"))
        lines.extend(
            [
                f"输入类型：视频（使用 {frame_count} 个关键帧；报告查询图为视频首要帧）",
                f"视频文件：{video_file}",
                f"视频SHA-256：{video_digest}",
                f"逐帧证据：{frame_evidence}",
            ]
        )

    deepfake_raw, deepfake = _module(report, "deepfake")
    dt = _mapping(deepfake.get("technical"))
    df_formula = _mapping(dt.get("formula"))
    lines.extend(
        [
            "",
            "二、AI换脸/深伪风险甄别",
            f"模块状态：{_text(deepfake.get('status'))}",
            f"结论：{_text(deepfake.get('headline'))}",
            f"检测方法：{_text(dt.get('method_name'))}",
        ]
    )
    if dt.get("branch") == "video_multi_frame_aggregation":
        band_thresholds = _mapping(dt.get("band_thresholds"))
        frame_counts = _mapping(dt.get("frame_band_counts"))
        band_sources = _mapping(dt.get("frame_band_sources"))
        primary_frame = _mapping(dt.get("primary_frame"))
        lines.extend(
            [
                f"生效分支：{_text(dt.get('branch_name'))}",
                f"模型名称：{_text(dt.get('model_name'))}",
                f"模型版本：{_structured_text(dt.get('model_version'))}",
                f"视频级深伪风险筛查分：{_fmt(dt.get('risk_score'))}",
                f"视频级分带：{_text(dt.get('band_label'))}",
                "分带边界："
                f"低/中 {_fmt(band_thresholds.get('low_max'))} · "
                f"中/高 {_fmt(band_thresholds.get('medium_max'))}",
                "关键帧统计："
                f"已评 {_text(dt.get('frames_scored'))}/{_text(dt.get('frames_total'))} 帧 · "
                f"高风险 {_text(frame_counts.get('high'))} · "
                f"需复核 {_text(frame_counts.get('medium'))} · "
                f"证据不足 {_text(frame_counts.get('low'))}",
                "分带来源："
                f"校准分带 {_text(band_sources.get('calibrated_band'))} 帧 · "
                f"冻结阈值 {_text(band_sources.get('frozen_threshold'))} 帧",
                f"聚合口径：{_text(dt.get('video_band_rule'))}",
                f"帧级一致性：{_text(dt.get('consistency_statement'))}",
                "首要帧："
                f"{_text(primary_frame.get('file_name'))} · "
                f"{_fmt(primary_frame.get('timestamp_seconds'))}s · "
                f"分数 {_fmt(primary_frame.get('score'))} · "
                f"分带 {_text(primary_frame.get('band'))}",
                f"首要帧哈希：{_text(dt.get('input_sha256'))}",
                "说明：视频级结论由帧级分数与确定性一致性规则导出，不是概率，也不是对整段视频的鉴定结论。",
                "判断依据：",
            ]
        )
    elif dt.get("branch") == "vit_full_image":
        band_thresholds = _mapping(dt.get("band_thresholds"))
        lines.extend(
            [
                f"生效分支：{_text(dt.get('branch_name'))}",
                f"模型名称：{_text(dt.get('model_name'))}",
                f"模型版本：{_structured_text(dt.get('model_version'))}",
                f"深伪风险筛查分：{_fmt(dt.get('risk_score'))}",
                f"证据分带：{_text(dt.get('band_label'))}",
                "分带边界："
                f"低/中 {_fmt(band_thresholds.get('low_max'))} · "
                f"中/高 {_fmt(band_thresholds.get('medium_max'))}",
                "路由依据："
                f"输入有效短边 {_text(dt.get('short_edge_px'))} px · "
                f"路由阈值 {_text(dt.get('routing_threshold_px'))} px",
                f"输入尺寸：{_structured_text(dt.get('original_size'))}",
                f"运行设备：{_text(dt.get('actual_device'))}",
                f"输入哈希：{_text(dt.get('input_sha256'))}",
                f"模型文件哈希：{_text(dt.get('model_sha256'))}",
                "说明：二代整图分支为单模型输出，不产生空间域/频域加权融合分解。",
                "判断依据：",
            ]
        )
    else:
        lines.extend(
            [
                f"空间域模型：{_text(dt.get('spatial_model_name'))}",
                f"频域模型：{_text(dt.get('frequency_model_name'))}",
                f"模型名称：{_text(dt.get('model_name'))}",
                f"模型版本：{_structured_text(dt.get('model_version'))}",
                f"风险分数：{_fmt(dt.get('fusion_score'))}",
                f"空间域分数：{_fmt(dt.get('spatial_score'))}",
                f"频域分数：{_fmt(dt.get('frequency_score'))}",
                f"融合公式代入：{_text(df_formula.get('substitution'))}",
                f"判定阈值：{_fmt(dt.get('decision_threshold'))}",
                f"与阈值差值：{_fmt(dt.get('threshold_margin'))}",
                f"运行设备：{_text(dt.get('actual_device'))}",
                f"有效人脸或有效帧数量：{_text(dt.get('valid_face_or_frame_count'))}",
                f"输入哈希：{_text(dt.get('input_sha256'))}",
                f"模型文件哈希：{_structured_text(dt.get('model_hashes'))}",
                f"融合配置哈希：{_text(dt.get('fusion_config_sha256'))}",
                "判断依据：",
            ]
        )
    _append_list(lines, deepfake.get("basis"))
    lines.append("警告：")
    _append_list(lines, deepfake.get("warnings"))
    lines.append("边界说明：")
    _append_list(lines, deepfake.get("limitations"))

    identity_raw, identity = _module(report, "identity")
    it = _mapping(identity.get("technical"))
    identity_candidates = it.get("candidates") if isinstance(it.get("candidates"), list) else []
    lines.extend(
        [
            "",
            "三、人脸身份搜索",
            f"模块状态：{_text(identity.get('status'))}",
            f"结论：{_text(identity.get('headline'))}",
            f"查询图检测到人脸数量：{_text(it.get('query_face_count'))}",
            f"模型：{_text(it.get('model_name'))}",
            f"特征维度：{_text(it.get('embedding_dimension'))}",
            f"身份库人数：{_text(it.get('library_identity_count'))}",
            f"身份库参考图片数：{_text(it.get('library_image_count'))}",
            f"有效embedding数：{_text(it.get('valid_embedding_count'))}",
            f"人员级计算方法：{_text(it.get('aggregation_method'))}",
            f"Top1与Top2差值：{_fmt(it.get('top1_top2_margin'))}",
            "Top-5候选：",
            "排名 | 人员 | 人员级平均余弦相似度 | 有效参考图数 | 最佳单图相似度 | 最佳参考图",
        ]
    )
    if identity_candidates:
        for candidate in identity_candidates[:5]:
            item = _mapping(candidate)
            lines.append(
                f"{_text(item.get('rank'))} | {_text(item.get('identity_name') or item.get('identity_id'))} | "
                f"{_fmt(item.get('mean_similarity'))} | {_text(item.get('reference_count'))} | "
                f"{_fmt(item.get('best_reference_score'))} | {display_name(item.get('best_reference_image'))}"
            )
    else:
        lines.append("无候选")
    lines.append("为什么第一候选排在第一：")
    _append_list(lines, identity.get("basis"))
    lines.append("说明：这是当前身份库中的闭集候选排序，不能单独据此确认人物身份。")
    lines.append("警告与限制：")
    _append_list(lines, [*(identity.get("warnings") or []), *(identity.get("limitations") or [])])

    source_raw, source = _module(report, "source")
    st = _mapping(source.get("technical"))
    source_candidates = st.get("candidates") if isinstance(st.get("candidates"), list) else []
    top_source = _mapping(source_candidates[0]) if source_candidates else {}
    local_formula = _mapping(top_source.get("local_formula"))
    final_formula = _mapping(st.get("final_formula"))
    lines.extend(
        [
            "",
            "四、原始底图搜索",
            f"模块状态：{_text(source.get('status'))}",
            f"结论：{_text(source.get('headline'))}",
            f"是否检测到查询图人脸：{_yes_no(st.get('query_face_detected'))}",
            f"是否应用人脸屏蔽：{_yes_no(st.get('face_mask_applied'))}",
            f"屏蔽图：{display_name(st.get('masked_query_path'))}",
            f"底图库文件数：{_text(st.get('library_file_count'))}",
            f"实际读取图片数：{_text(st.get('library_decoded'))}",
            f"跳过图片数：{_text(st.get('library_skipped'))}",
            f"粗排候选数量：{_text(st.get('shortlist_size'))}",
            "全局粗排方式：颜色、灰度、边缘和外围区域等全局特征。",
            "局部复核方式：SIFT局部特征、Lowe比率检验和RANSAC几何验证。",
            f"第一候选全局分：{_fmt(top_source.get('global_score'))}",
            f"第一候选局部分：{_fmt(top_source.get('local_score'))}",
            f"第一候选最终分：{_fmt(top_source.get('final_score'))}",
            f"Lowe有效匹配点：{_text(top_source.get('good_match_count'))}",
            f"RANSAC内点数：{_text(top_source.get('ransac_inlier_count'))}",
            f"RANSAC内点率：{_fmt(top_source.get('ransac_inlier_ratio'))}",
            f"局部分公式代入：{_text(local_formula.get('substitution'))}",
            f"最终分公式代入：{_text(final_formula.get('substitution'))}",
            f"第一名与第二名差值：{_fmt(st.get('top1_top2_margin'))}",
            f"匹配可视化：{display_name(top_source.get('visualization_path'))}",
            "Top-5候选：",
            "排名 | 文件 | 全局分 | 局部分 | 最终分 | 有效匹配点 | RANSAC内点 | 内点率",
        ]
    )
    if source_candidates:
        for candidate in source_candidates[:5]:
            item = _mapping(candidate)
            lines.append(
                f"{_text(item.get('rank'))} | {display_name(item.get('file_path'))} | {_fmt(item.get('global_score'))} | "
                f"{_fmt(item.get('local_score'))} | {_fmt(item.get('final_score'))} | "
                f"{_text(item.get('good_match_count'))} | {_text(item.get('ransac_inlier_count'))} | "
                f"{_fmt(item.get('ransac_inlier_ratio'))}"
            )
    else:
        lines.append("无候选")
    lines.append("为什么第一候选排在第一：")
    _append_list(lines, source.get("basis"))
    lines.append("说明：结果仅表示当前底图库内最相似候选，不能单独确认原始底图。")
    lines.append("警告与限制：")
    _append_list(lines, [*(source.get("warnings") or []), *(source.get("limitations") or [])])

    summary = _mapping(report.get("summary"))
    lines.extend(
        _evidence_matrix_lines(deepfake_raw, identity_raw, source_raw)
    )
    lines.extend(
        [
            "",
            "六、并列证据总结",
            f"AIGC模块：{redact_text(summary.get('deepfake'))}",
            f"身份搜索模块：{redact_text(summary.get('identity'))}",
            f"底图搜索模块：{redact_text(summary.get('source'))}",
            "本报告不生成跨模块综合概率。",
            "",
            _text(report.get("disclaimer"), DEFAULT_DISCLAIMER),
            "",
        ]
    )
    return "\n".join(lines)


def render_html_report(report: Mapping[str, Any]) -> str:
    """Render the same report content as standalone, escaped UTF-8 HTML."""

    title = _text(_mapping(report.get("product")).get("name"), DEFAULT_PRODUCT_NAME)
    escaped_title = html.escape(title, quote=True)
    escaped_report = html.escape(render_text_report(report), quote=True)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title} - 检测结果报告</title>
  <style>
    body {{ margin: 0; background: #f4f6f8; color: #17202a; font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }}
    main {{ max-width: 1040px; margin: 28px auto; padding: 32px 40px; background: white; border: 1px solid #dfe6ec; border-radius: 12px; }}
    h1 {{ margin: 0 0 20px; font-size: 24px; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font: 14px/1.75 "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }}
    .notice {{ margin-top: 24px; padding: 12px 16px; background: #fff7e6; border-left: 4px solid #d48806; }}
    @media print {{ body {{ background: white; }} main {{ border: 0; margin: 0; max-width: none; }} }}
  </style>
</head>
<body>
<main>
  <h1>{escaped_title}</h1>
  <pre>{escaped_report}</pre>
  <div class="notice">本页面由本地确定性模板生成，未调用外部大语言模型。</div>
</main>
</body>
</html>
"""


def _atomic_write(path: Path, data: bytes) -> None:
    if len(data) > _MAX_REPORT_BYTES:
        raise ValueError(f"报告文件超过安全上限：{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return _mapping(value)


def generate_task_reports(
    task_dir: str | Path,
    *,
    task_id: str | None = None,
    task_status: Mapping[str, Any] | None = None,
    deepfake_result: Mapping[str, Any] | None = None,
    identity_result: Mapping[str, Any] | None = None,
    source_result: Mapping[str, Any] | None = None,
    unified_result: Mapping[str, Any] | None = None,
    product_name: str = DEFAULT_PRODUCT_NAME,
    product_version: str = "0.3",
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build and atomically write all required reports inside one task root.

    Missing input mappings are read from their fixed task filenames.  This lets
    the orchestrator pass in-memory results while also supporting deterministic
    regeneration during acceptance checks.
    """

    root = Path(task_dir).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    status_data = _mapping(task_status) or _read_optional_json(root / "task_status.json")
    deepfake = _mapping(deepfake_result) or _read_optional_json(root / "detection_result.json")
    identity = _mapping(identity_result) or _read_optional_json(root / "identity_result.json")
    source = _mapping(source_result) or _read_optional_json(root / "source_result.json")
    unified = _mapping(unified_result) or _read_optional_json(root / "unified_result.json")
    resolved_task_id = str(task_id or status_data.get("task_id") or root.name)
    report = build_final_report(
        resolved_task_id,
        task_status=status_data,
        deepfake_result=deepfake,
        identity_result=identity,
        source_result=source,
        unified_result=unified,
        product_name=product_name,
        product_version=product_version,
        generated_at=generated_at,
        input_metadata=_task_input_metadata(root, _query_value(
            {"deepfake": deepfake, "identity": identity, "source": source}, status_data
        )),
    )
    names = report_file_names(resolved_task_id)
    json_path = root / names["json"]
    txt_path = root / names["txt"]
    html_path = root / names["html"]
    json_bytes = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    text_bytes = render_text_report(report).encode("utf-8")
    html_bytes = render_html_report(report).encode("utf-8")
    _atomic_write(json_path, json_bytes)
    _atomic_write(txt_path, text_bytes)
    _atomic_write(html_path, html_bytes)
    return {
        "report": report,
        "json_path": str(json_path),
        "txt_path": str(txt_path),
        "html_path": str(html_path),
        "file_names": {
            "json": json_path.name,
            "txt": txt_path.name,
            "html": html_path.name,
        },
    }


# Stable descriptive alias for integration code.
build_and_write_reports = generate_task_reports


__all__ = [
    "DEFAULT_DISCLAIMER",
    "DEFAULT_PRODUCT_NAME",
    "build_and_write_reports",
    "build_final_report",
    "generate_task_reports",
    "report_file_names",
    "render_html_report",
    "render_text_report",
]
