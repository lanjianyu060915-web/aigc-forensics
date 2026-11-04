from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import streamlit as st

from components.live_progress import load_progress_events
from components.path_security import PathContainmentError, ResourcePathPolicy
from components.result_views import format_score, render_image
from components.task_history import status_label
from components.theme import verdict_card
from decision import (
    STATE_HIGH,
    STATE_INSUFFICIENT,
    STATE_NONE,
    decide_identity_open_set,
    decide_source_open_set,
)
from decision.evidence_matrix import build_evidence_matrix
from components.aigc_views import (
    overall_finding,
    provenance_finding,
    render_face_branch_section,
    render_general_aigc_section,
    render_overall_section,
    render_provenance_page,
)


PAGES = (
    "研判总览",
    "AIGC鉴伪证据",
    "文件标识",
    "人脸身份比对",
    "原始素材溯源",
    "证据链与完整报告",
)

PAGE_KEYS = {
    "deepfake": "AIGC鉴伪证据",
    "identity": "人脸身份比对",
    "source": "原始素材溯源",
}

# 视频级聚合口径的可读名；键来自 video_pipeline.aggregate 的规则常量。
_VIDEO_BAND_RULE_LABELS = {
    "band_majority": "帧级分带多数一致性（未标定口径）",
    "median_video_thresholds": "视频级中位分对照独立视频校准阈值",
}

_VIDEO_BAND_RULE_NOTES = {
    "band_majority": "帧级分数取中位数、分带按帧级一致性导出；不是概率，也不是对整段视频的鉴定结论。",
    "median_video_thresholds": (
        "帧级分数取中位数，再与独立视频校准集真人视频分布的分位点比较定带；"
        "不是概率，也不是对整段视频的鉴定结论。"
    ),
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _six(value: Any) -> str:
    number = _number(value)
    return f"{number:.6f}" if number is not None else "—"


def _signed_six(value: Any) -> str:
    number = _number(value)
    return f"{number:+.6f}" if number is not None else "—"


def _safe_text(value: Any, fallback: str = "—") -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _table_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _basename(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "—"
    return Path(value).name or value


def _module_visual(data: Mapping[str, Any], module: str) -> dict[str, Any]:
    raw = data.get("visual_evidence")
    if not isinstance(raw, Mapping):
        details = data.get("details")
        raw = details.get("visual_evidence") if isinstance(details, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    nested = raw.get(module)
    return dict(nested) if isinstance(nested, Mapping) else dict(raw)


def _walk_values(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)


def _find_value(data: Mapping[str, Any], *keys: str) -> Any:
    wanted = set(keys)
    for key in keys:
        if key in data and data[key] not in (None, "", [], {}):
            return data[key]
    for key, value in _walk_values(data):
        if key in wanted and value not in (None, "", [], {}):
            return value
    return None


def _find_list(data: Mapping[str, Any], *keys: str) -> list[Any]:
    value = _find_value(data, *keys)
    return list(value) if isinstance(value, list) else []


def _path_value(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("path", "relative_path", "artifact_path", "file"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def resolve_evidence_path(
    value: Any,
    policy: ResourcePathPolicy,
    *,
    kind: str = "visualization",
    label: str,
) -> Path | None:
    raw = _path_value(value)
    if raw is None:
        return None
    try:
        return policy.resolve(raw, kind, label=label)  # type: ignore[arg-type]
    except PathContainmentError:
        return None


def _render_evidence_image(
    value: Any,
    policy: ResourcePathPolicy,
    *,
    caption: str,
    key: str,
    kind: str = "visualization",
) -> bool:
    path = resolve_evidence_path(value, policy, kind=kind, label=caption)
    if path is None or not path.is_file():
        st.warning(f"本次任务未生成该证据：{caption}")
        return False
    render_image(path, caption, key)
    return True


def _query_path(data: Mapping[str, Any], policy: ResourcePathPolicy) -> Path | None:
    value = data.get("query_image")
    if not isinstance(value, str):
        return None
    try:
        path = policy.resolve(value, "query", label="query_image")
    except PathContainmentError:
        return None
    return path if path.is_file() else None


def collect_artifact_manifest(*results: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for result in results:
        module = str(result.get("module") or "unknown")
        raw = result.get("artifact_manifest")
        if not isinstance(raw, list):
            visual = result.get("visual_evidence")
            raw = visual.get("artifact_manifest") if isinstance(visual, Mapping) else None
        if not isinstance(raw, list):
            continue
        for index, item in enumerate(raw, 1):
            if not isinstance(item, Mapping):
                continue
            record = dict(item)
            record.setdefault("module", module)
            record.setdefault("id", f"{module}-{index}")
            path = _path_value(record.get("path") or record.get("relative_path")) or ""
            key = (str(record["module"]), path)
            if key in seen:
                continue
            seen.add(key)
            output.append(record)
    return output


def build_task_overview(
    deepfake: Mapping[str, Any],
    identity: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    task_status: Mapping[str, Any] | None = None,
    final_report: Mapping[str, Any] | None = None,
    task_dir: Path | None = None,
    product_version: str = "0.4.0",
) -> dict[str, Any]:
    status = _mapping(task_status)
    report = _mapping(final_report)
    report_input = _mapping(report.get("input"))
    report_product = _mapping(report.get("product"))
    details = _mapping(deepfake.get("details"))
    metadata = _mapping(details.get("media_metadata"))
    deepfake_runtime = _mapping(deepfake.get("runtime"))
    identity_runtime = _mapping(identity.get("runtime"))
    source_runtime = _mapping(source.get("runtime"))
    query = str(
        deepfake.get("query_image")
        or identity.get("query_image")
        or source.get("query_image")
        or status.get("query_image")
        or ""
    )
    width = metadata.get("width") or report_input.get("width")
    height = metadata.get("height") or report_input.get("height")
    dimensions = f"{width} × {height}" if width and height else "—"
    status_input = _mapping(status.get("input"))
    original_name = str(status_input.get("original_file_name") or "").strip()
    input_hash = (
        details.get("input_sha256")
        or _mapping(deepfake.get("provenance")).get("input_sha256")
        or _mapping(source.get("provenance")).get("input_sha256")
        or report_input.get("sha256")
    )
    total_runtime = _number(status.get("runtime_seconds"))
    if total_runtime is None:
        module_values = [
            _number(deepfake.get("runtime_seconds")),
            _number(identity.get("runtime_seconds")),
            _number(source.get("runtime_seconds")),
        ]
        total_runtime = sum(value or 0.0 for value in module_values)
    device = (
        deepfake_runtime.get("actual_device")
        or details.get("actual_device")
        or identity_runtime.get("device")
        or "CPU"
    )
    return {
        "system_name": report_product.get("name") or "AIGC 图像视频鉴伪与溯源系统",
        "product_version": report_product.get("version") or product_version,
        "task_id": status.get("task_id") or (task_dir.name if task_dir else "—"),
        "query_file": original_name or (Path(query).name if query else "—"),
        "original_file_name": original_name,
        "dimensions": dimensions,
        "input_sha256": input_hash or "—",
        "task_status": status.get("status") or "已载入",
        "total_runtime": total_runtime,
        "deepfake_runtime": _number(deepfake_runtime.get("module_elapsed_seconds")) or _number(deepfake.get("runtime_seconds")),
        "identity_runtime": _number(identity_runtime.get("module_elapsed_seconds")) or _number(identity.get("runtime_seconds")),
        "source_runtime": _number(source_runtime.get("module_elapsed_seconds")) or _number(source.get("runtime_seconds")),
        "device": str(device),
        "started_at": status.get("started_at") or "—",
        "finished_at": status.get("finished_at") or "—",
    }


def _friendly_time(value: Any) -> str:
    text = str(value or "").strip()
    return text.replace("T", " ")[:19] if text else "—"


_OPEN_SET_TONES = {STATE_HIGH: "ok", STATE_INSUFFICIENT: "warn", STATE_NONE: "muted"}
_BAND_TONES = {"high": "bad", "medium": "warn", "low": "muted"}


def _not_ok_finding(title: str, data: Mapping[str, Any]) -> dict[str, str] | None:
    status = _result_status(data)
    if data.get("not_applicable"):
        return {
            "title": title,
            "value": "不适用",
            "note": _safe_text(_mapping(data.get("details")).get("reason"), "这次输入不适用这个模块。"),
            "tone": "muted",
        }
    if status == "skipped" or status == "unknown":
        return {"title": title, "value": "本次未运行", "note": "这个模块没有被选中。", "tone": "muted"}
    if status != "success":
        return {
            "title": title,
            "value": "模块未完成",
            "note": _safe_text(data.get("error_message"), "请查看技术详情。"),
            "tone": "bad",
        }
    return None


def _deepfake_finding(deepfake: Mapping[str, Any]) -> dict[str, str]:
    title = "是否伪造"
    failed = _not_ok_finding(title, deepfake)
    if failed is not None:
        return failed
    overall = overall_finding(deepfake)
    if overall is not None:
        return overall
    details = _mapping(deepfake.get("details"))
    score = _six(deepfake.get("score"))
    band_label = str(details.get("band_label") or "").strip()
    if band_label:
        short_edge = details.get("short_edge_px")
        where = "视频关键帧聚合" if details.get("video_band_rule") else f"整图分支，短边 {short_edge or '—'} px"
        return {
            "title": title,
            "value": band_label,
            "note": f"{where}，风险筛查分 {score}。按分带读结果，分数的绝对大小没有概率含义。",
            "tone": _BAND_TONES.get(str(details.get("band") or ""), "muted"),
        }
    verdict = deepfake.get("is_deepfake")
    value, tone = (
        ("疑似AI换脸", "bad") if verdict is True
        else ("未见明显换脸特征", "ok") if verdict is False
        else ("证据不足", "muted")
    )
    metrics = _mapping(deepfake.get("metrics"))
    threshold = metrics.get("decision_threshold") or details.get("decision_threshold")
    return {
        "title": title,
        "value": value,
        "note": f"人脸裁剪分支，融合风险分 {score}，判定阈值 {_six(threshold)}。",
        "tone": tone,
    }


def _open_set_finding(
    title: str,
    data: Mapping[str, Any],
    decide,
    describe,
    caveat: str,
) -> tuple[dict[str, str], str | None]:
    failed = _not_ok_finding(title, data)
    if failed is not None:
        return failed, None
    decision = decide(data)
    candidates = _list(data.get("candidates"))
    top = _mapping(candidates[0]) if candidates else {}
    note = f"{describe(top)}。{caveat}" if top else caveat
    original_faces = _integer(_mapping(data.get("metrics")).get("query_face_count_original"))
    if original_faces:
        note = f"查询图有 {original_faces} 张人脸，按面积最大选了一张比对。{note}"
    state = decision.get("state")
    return (
        {
            "title": title,
            "value": _safe_text(decision.get("state_label")),
            "note": note,
            "tone": _OPEN_SET_TONES.get(state, "muted"),
        },
        str(state) if state else None,
    )


def build_key_findings(
    deepfake: Mapping[str, Any],
    identity: Mapping[str, Any],
    source: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    """把各模块的结论翻译成白话并给出下一步建议；只读结果，不产生新的分数。"""

    deepfake_card = _deepfake_finding(deepfake)
    identity_card, identity_state = _open_set_finding(
        "可能是谁",
        identity,
        decide_identity_open_set,
        lambda top: (
            f"库内第一候选 {_safe_text(top.get('identity_name') or top.get('identity_id'))}，"
            f"平均相似度 {_six(top.get('mean_similarity') or top.get('score'))}"
        ),
        "这是库内比较结果，不等于确认身份。",
    )
    source_card, source_state = _open_set_finding(
        "原图在哪",
        source,
        decide_source_open_set,
        lambda top: f"库内第一候选 {_basename(top.get('file_path'))}，最终分 {_six(top.get('final_score'))}",
        "这是库内比较结果，不等于确认来源。",
    )
    steps: list[str] = []
    overall_band = _mapping(deepfake.get("overall")).get("band")
    band = str(overall_band or _mapping(deepfake.get("details")).get("band") or "")
    if source_state == STATE_HIGH and _result_status(deepfake) == "success" and band != "low":
        steps.append("到“原始素材溯源”页把查询图和第一候选并排看：人脸以外基本一致而人脸不同，是换脸的直接线索。")
    if band == "medium":
        steps.append("鉴伪落在“需人工复核”，建议结合原始文件和传播链做人工判断。")
    if band == "low":
        steps.append("“证据不足”不代表图像真实，只说明模型没有发现足够的风险信号。")
    if identity_state == STATE_HIGH:
        steps.append("身份候选只说明和库内该人员最相似，需要结合其他线索核实。")
    if STATE_NONE in {identity_state, source_state}:
        steps.append("库中没有找到可靠候选时，可以扩充身份库或素材库后重新检索。")
    if STATE_INSUFFICIENT in {identity_state, source_state}:
        steps.append("证据不足的候选不要直接采信，建议换一张更清晰的图或帧再查。")
    if _integer(_mapping(identity.get("metrics")).get("query_face_count_original")):
        steps.append("这张图有多张人脸：如果要查的不是面积最大的那张，请裁剪后重新分析。")
    cards = [deepfake_card, identity_card, source_card]
    if provenance:
        cards.insert(1, provenance_finding(provenance))
        if provenance.get("verdict") in {"labelled_ai", "labelled_possible"}:
            steps.insert(0, "文件带有生成合成标识：到“文件标识”页查看生成服务方和传播平台代码，可作为向平台核实的线索。")
    return cards, steps


def _render_key_findings(
    deepfake: Mapping[str, Any],
    identity: Mapping[str, Any],
    source: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> None:
    cards, steps = build_key_findings(deepfake, identity, source, provenance)
    st.markdown("#### 研判要点")
    for column, card in zip(st.columns(len(cards)), cards, strict=True):
        with column:
            verdict_card(card["title"], card["value"], card["note"], card["tone"])
    if steps:
        st.markdown("**下一步建议**")
        for step in steps:
            st.markdown(f"- {step}")


def _queue_page(page: str, request_key: str) -> None:
    st.session_state[request_key] = page


def _result_status(data: Mapping[str, Any]) -> str:
    return str(data.get("status") or "unknown")


def _render_task_header(info: Mapping[str, Any], *, video: Mapping[str, Any] | None = None) -> None:
    """任务头部概览；视频任务用视频文件与视频哈希替换单图口径字段。"""

    video_info = _mapping(video)
    video_meta = _mapping(video_info.get("video"))
    is_video = str(video_info.get("input_type") or "") == "video"
    dimensions = _safe_text(info.get("dimensions"))
    if is_video and dimensions == "—":
        width = video_meta.get("width")
        height = video_meta.get("height")
        if width and height:
            dimensions = f"{width} × {height}"
    st.markdown("#### 任务概览")
    first = st.columns([1.7, 1, 1, 1, 1.5])
    first[0].metric(
        "查询视频" if is_video else "查询文件",
        _safe_text(
            (info.get("original_file_name") or video_meta.get("file_name"))
            if is_video
            else info.get("query_file")
        ),
    )
    first[1].metric(
        "视频分辨率" if is_video else "图片尺寸",
        dimensions,
    )
    first[2].metric("任务状态", status_label(info.get("task_status")))
    total = _number(info.get("total_runtime"))
    first[3].metric("任务总耗时", f"{total:.1f} 秒" if total is not None else "—")
    first[4].metric("完成时间", _friendly_time(info.get("finished_at")))
    second = st.columns(4)
    second[0].metric("鉴伪耗时", f"{_number(info.get('deepfake_runtime')) or 0.0:.1f} 秒")
    second[1].metric("身份检索耗时", f"{_number(info.get('identity_runtime')) or 0.0:.1f} 秒")
    second[2].metric("素材溯源耗时", f"{_number(info.get('source_runtime')) or 0.0:.1f} 秒")
    second[3].metric("运行设备", _safe_text(info.get("device")).upper())
    st.caption(f"任务编号：{_safe_text(info.get('task_id'))} · 系统版本 {_safe_text(info.get('product_version'))}")
    digest = _safe_text(info.get("input_sha256"))
    if is_video:
        frame_digest = digest if digest != "—" else _primary_frame_sha(video_info)
        st.caption(f"输入视频 SHA-256：{_safe_text(video_info.get('video_sha256'))}")
        st.caption(f"首要关键帧 SHA-256（鉴伪模型输入）：{frame_digest}")
    else:
        st.caption(f"输入文件 SHA-256：{digest}")


def _primary_frame_sha(video: Mapping[str, Any]) -> str:
    """视频任务的首要关键帧 SHA-256；模块结果未记录输入哈希时由关键帧清单兜底。"""

    frames = video.get("key_frames")
    if isinstance(frames, list):
        for item in frames:
            if isinstance(item, Mapping) and item.get("sha256"):
                return _safe_text(item.get("sha256"))
    return "—"


def _render_open_set_badge(decision: Mapping[str, Any]) -> None:
    """按开放集三态渲染结论标签：高可信 / 建议复核 / 库中未发现可靠候选。"""

    state = decision.get("state")
    label = _safe_text(decision.get("state_label"))
    if state == STATE_HIGH:
        st.success(f"开放集判定：{label}")
    elif state == STATE_INSUFFICIENT:
        st.warning(f"开放集判定：{label}")
    elif state == STATE_NONE:
        st.error(f"开放集判定：{label}")
    else:
        st.info(f"开放集判定：{label}")


def _render_overview_cards(
    deepfake: Mapping[str, Any],
    identity: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    request_key: str,
    key_prefix: str,
) -> None:
    columns = st.columns(3)
    details = _mapping(deepfake.get("details"))
    metrics = _mapping(deepfake.get("metrics"))
    with columns[0]:
        with st.container(border=True):
            st.markdown("#### AIGC鉴伪")
            kernel = resolve_deepfake_kernel_view(deepfake)
            overall = _mapping(deepfake.get("overall"))
            if overall:
                st.metric("总体档位", _safe_text(overall.get("band_label")))
                general = _mapping(deepfake.get("general_aigc"))
                st.caption(
                    f"通用 AIGC 分支：{_safe_text(general.get('band_label') or general.get('error_message'))}"
                    f"，融合分数 {_six(general.get('logit'))}"
                )
                st.caption("下面是整图分支的结果，只作提示，不影响总体档位。")
            if details.get("video_band_rule"):
                band_counts = _mapping(details.get("frame_band_counts"))
                st.metric("视频级分带", _safe_text(details.get("band_label") or details.get("band")))
                st.metric("视频级深伪风险分", _six(deepfake.get("score")))
                st.caption(
                    f"关键帧 {details.get('frames_scored') or '—'}/{details.get('frames_total') or '—'} · "
                    f"高风险 {band_counts.get('high') or 0} 帧 · 聚合口径 "
                    f"{_safe_text(_VIDEO_BAND_RULE_LABELS.get(str(details.get('video_band_rule')), details.get('video_band_rule')))}"
                )
                st.caption(
                    _VIDEO_BAND_RULE_NOTES.get(
                        str(details.get("video_band_rule")),
                        _VIDEO_BAND_RULE_NOTES["band_majority"],
                    )
                )
            elif kernel is not None and kernel.get("branch") == "vit_full_image":
                st.metric("证据分带", _safe_text(kernel.get("band_label")))
                st.metric("深伪风险筛查分", _six(deepfake.get("score")))
                st.caption(
                    f"分支 二代整图检测器 · 输入短边 {kernel.get('short_edge_px') or '—'} px · "
                    f"状态 {status_label(_result_status(deepfake))}"
                )
            else:
                st.metric("检测结论", _safe_text(_mapping(deepfake.get("explanation")).get("decision") or deepfake.get("label")))
                st.metric("融合风险分数", _six(metrics.get("fusion_score") or deepfake.get("score")))
                pair = st.columns(2)
                pair[0].metric("空间域", _six(metrics.get("spatial_score") or details.get("spatial_score")))
                pair[1].metric("频域", _six(metrics.get("frequency_score") or details.get("frequency_score")))
                st.caption(
                    f"阈值 {_six(metrics.get('decision_threshold') or details.get('decision_threshold'))} · "
                    f"差值 {_signed_six(metrics.get('decision_margin'))} · 状态 {status_label(_result_status(deepfake))}"
                )
                if kernel is not None:
                    st.caption(f"二代 worker 内部路由：{kernel.get('branch_label')}")
            st.button(
                "查看AIGC证据 →",
                key=f"{key_prefix}_goto_deepfake",
                width="stretch",
                on_click=_queue_page,
                args=(PAGE_KEYS["deepfake"], request_key),
            )
    candidates = _list(identity.get("candidates"))
    identity_metrics = _mapping(identity.get("metrics"))
    top_identity = _mapping(candidates[0]) if candidates else {}
    top2 = _mapping(candidates[1]) if len(candidates) > 1 else {}
    with columns[1]:
        with st.container(border=True):
            st.markdown("#### 人脸身份比对")
            st.metric("当前库内第一候选", _safe_text(top_identity.get("identity_name") or top_identity.get("identity_id")))
            st.metric("人员级平均余弦相似度", _six(top_identity.get("mean_similarity") or top_identity.get("score")))
            pair = st.columns(2)
            pair[0].metric("Top1-Top2", _six(identity_metrics.get("top1_top2_margin")))
            pair[1].metric("有效参考图", identity_metrics.get("valid_embedding_count") or "—")
            support_note = ""
            if identity_metrics.get("frames_with_faces"):
                support_note = (
                    f" · 跨帧支持 {_safe_text(top_identity.get('frame_support') or '—')}/"
                    f"{_safe_text(identity_metrics.get('frames_with_faces'))} 帧"
                )
            st.caption(
                f"第二候选 {_safe_text(top2.get('identity_name') or top2.get('identity_id'))} · "
                f"身份库 {identity_metrics.get('library_identity_count') or '—'} 人{support_note}"
            )
            _render_open_set_badge(decide_identity_open_set(identity))
            st.caption("库内候选；余弦相似度不是身份概率。")
            st.button(
                "查看身份证据 →",
                key=f"{key_prefix}_goto_identity",
                width="stretch",
                on_click=_queue_page,
                args=(PAGE_KEYS["identity"], request_key),
            )
    source_candidates = _list(source.get("candidates"))
    source_metrics = _mapping(source.get("metrics"))
    top_source = _mapping(source_candidates[0]) if source_candidates else {}
    with columns[2]:
        with st.container(border=True):
            st.markdown("#### 原始素材溯源")
            st.metric("当前库内第一候选", _basename(top_source.get("file_path")))
            st.metric("最终分", _six(top_source.get("final_score")))
            pair = st.columns(2)
            pair[0].metric("全局分", _six(top_source.get("global_score")))
            pair[1].metric("局部分", _six(top_source.get("local_score")))
            st.caption(
                f"Top1-Top2 {_six(source_metrics.get('top1_top2_margin'))} · "
                f"RANSAC {top_source.get('ransac_inlier_count') or '—'} 点 / {_six(top_source.get('ransac_inlier_ratio'))}"
            )
            _render_open_set_badge(decide_source_open_set(source))
            st.caption("库内候选；检索分数不是来源概率。")
            st.button(
                "查看素材证据 →",
                key=f"{key_prefix}_goto_source",
                width="stretch",
                on_click=_queue_page,
                args=(PAGE_KEYS["source"], request_key),
            )


def _render_threshold_chart(score: Any, threshold: Any) -> None:
    score_value = _number(score)
    threshold_value = _number(threshold)
    if score_value is None or threshold_value is None:
        st.warning("本次结果缺少风险分数或阈值，无法绘制阈值尺。")
        return
    maximum = max(1.0, score_value, threshold_value)
    spec = {
        "data": {"values": [{"start": 0.0, "score": score_value, "threshold": threshold_value}]},
        "layer": [
            {
                "mark": {"type": "bar", "height": 34, "cornerRadiusEnd": 5, "color": "#315f86"},
                "encoding": {
                    "x": {"field": "start", "type": "quantitative", "scale": {"domain": [0, maximum]}, "title": "融合风险分数"},
                    "x2": {"field": "score"},
                    "tooltip": [{"field": "score", "type": "quantitative", "format": ".6f", "title": "风险分数"}],
                },
            },
            {
                "mark": {"type": "rule", "color": "#c55a3d", "strokeWidth": 4},
                "encoding": {
                    "x": {"field": "threshold", "type": "quantitative"},
                    "tooltip": [{"field": "threshold", "type": "quantitative", "format": ".6f", "title": "判定阈值"}],
                },
            },
        ],
        "height": 70,
    }
    st.vega_lite_chart(spec, width="stretch")


def _render_signed_contributions(contributions: Sequence[Any]) -> None:
    values: list[dict[str, Any]] = []
    for index, raw in enumerate(contributions):
        if not isinstance(raw, Mapping):
            continue
        contribution_value = raw.get("contribution")
        if contribution_value is None:
            contribution_value = raw.get("signed_contribution")
        contribution = _number(contribution_value)
        if contribution is None:
            continue
        name = raw.get("feature_name") or raw.get("feature") or raw.get("name")
        if not name:
            feature_index = raw.get("feature_index", raw.get("index", index))
            name = f"feature_{int(feature_index):02d}" if isinstance(feature_index, int) else f"feature_{index:02d}"
        values.append(
            {
                "feature": str(name),
                "contribution": contribution,
                "direction": "推动风险上升" if contribution >= 0 else "推动风险下降",
                "feature_value": raw.get("model_input_value")
                if raw.get("model_input_value") is not None
                else raw.get("feature_value")
                if raw.get("feature_value") is not None
                else raw.get("standardized_value"),
                "coefficient": raw.get("coefficient"),
            }
        )
    values = sorted(values, key=lambda item: abs(item["contribution"]), reverse=True)[:10]
    if not values:
        st.info("当前模型仅支持频域整体分数解释，本次没有可严格恢复的逐特征贡献。")
        return
    spec = {
        "data": {"values": values},
        "mark": {"type": "bar", "cornerRadiusEnd": 4},
        "encoding": {
            "y": {"field": "feature", "type": "nominal", "sort": "-x", "title": "真实特征索引/名称"},
            "x": {"field": "contribution", "type": "quantitative", "title": "标准化特征值 × 回归系数"},
            "color": {
                "field": "direction",
                "type": "nominal",
                "scale": {"domain": ["推动风险上升", "推动风险下降"], "range": ["#b75d3e", "#3b6f9c"]},
                "title": "作用方向",
            },
            "tooltip": [
                {"field": "feature", "title": "特征"},
                {"field": "contribution", "type": "quantitative", "format": ".8f", "title": "有符号贡献"},
                {"field": "feature_value", "title": "模型输入值"},
                {"field": "coefficient", "title": "回归系数"},
            ],
        },
        "height": max(260, 28 * len(values)),
    }
    st.vega_lite_chart(spec, width="stretch")


KERNEL_V2_BRANCHES = {
    "vit_full_image": "二代整图检测器（ViT-S/16 深伪风险筛查分）",
    "v040_face_crop": "v0.4.0 冻结人脸裁剪分支（由二代 worker 内部代理）",
}

BAND_ORDER = ("low", "medium", "high")


def resolve_deepfake_kernel_view(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """解析二代鉴伪内核返回的路由分支与证据分带；非二代结果返回 None。

    分带表示证据质量（本次分数相对独立校准集真人图分数分布的位置），
    不是概率、准确率或司法鉴定结论。
    """

    if _result_status(data) != "success":
        return None
    details = _mapping(data.get("details"))
    method = _mapping(data.get("method"))
    branch = details.get("branch")
    if not isinstance(branch, str) or branch not in KERNEL_V2_BRANCHES:
        branch = details.get("routed_branch")
    if not isinstance(branch, str) or branch not in KERNEL_V2_BRANCHES:
        branch = method.get("branch")
    if not isinstance(branch, str) or branch not in KERNEL_V2_BRANCHES:
        return None
    band = details.get("band")
    if not isinstance(band, str) or band not in BAND_ORDER:
        band = None
    thresholds = _mapping(details.get("band_thresholds"))
    labels = _mapping(thresholds.get("labels"))
    band_label = details.get("band_label")
    if not isinstance(band_label, str) or not band_label.strip():
        band_label = labels.get(band)
    return {
        "kernel": "deepfake_detection_v2",
        "branch": branch,
        "branch_label": KERNEL_V2_BRANCHES[branch],
        "band": band,
        "band_label": _safe_text(band_label),
        "band_reason": details.get("band_reason"),
        "band_low_max": _number(thresholds.get("low_max")),
        "band_medium_max": _number(thresholds.get("medium_max")),
        "short_edge_px": _integer(details.get("short_edge_px")),
        "routing_threshold_px": _integer(details.get("routing_threshold_px")),
        "routing_reason": details.get("routing_reason"),
        "routing_matched": details.get("routing_matched"),
        "requires_review": band in ("medium", "high"),
    }


def _render_band_scale_chart(score: Any, kernel: Mapping[str, Any]) -> None:
    score_value = _number(score)
    low_max = _number(kernel.get("band_low_max"))
    medium_max = _number(kernel.get("band_medium_max"))
    if score_value is None or low_max is None or medium_max is None:
        st.warning("本次结果缺少风险分数或分带边界，无法绘制证据分带尺。")
        return
    upper = max(medium_max * 1.4, score_value * 1.05, 1e-9)
    bands = [
        {"start": 0.0, "end": low_max, "grade": "低（证据不足）", "color": "#7d8794"},
        {"start": low_max, "end": medium_max, "grade": "中（需人工复核）", "color": "#d9a441"},
        {"start": medium_max, "end": upper, "grade": "高（高风险提示）", "color": "#c55a3d"},
    ]
    spec = {
        "data": {"values": bands},
        "layer": [
            {
                "mark": {"type": "bar", "height": 34, "opacity": 0.55},
                "encoding": {
                    "x": {
                        "field": "start",
                        "type": "quantitative",
                        "scale": {"domain": [0.0, upper]},
                        "title": "深伪风险筛查分",
                    },
                    "x2": {"field": "end"},
                    "color": {
                        "field": "grade",
                        "type": "nominal",
                        "scale": {
                            "domain": [item["grade"] for item in bands],
                            "range": [item["color"] for item in bands],
                        },
                        "legend": {"title": "证据分带"},
                    },
                    "tooltip": [
                        {"field": "grade", "title": "分带"},
                        {"field": "start", "type": "quantitative", "format": ".8f", "title": "下界"},
                        {"field": "end", "type": "quantitative", "format": ".8f", "title": "上界"},
                    ],
                },
            },
            {
                "data": {"values": [{"score": score_value}]},
                "mark": {"type": "rule", "color": "#111111", "strokeWidth": 4},
                "encoding": {
                    "x": {"field": "score", "type": "quantitative"},
                    "tooltip": [
                        {"field": "score", "type": "quantitative", "format": ".8f", "title": "本次分数"}
                    ],
                },
            },
        ],
        "height": 90,
    }
    st.vega_lite_chart(spec, width="stretch")
    st.caption(
        "分带边界取独立校准集真人图分数的 p95 与 p99，"
        "不是通用置信区间；更换数据来源后必须重新标定。"
    )


def _render_deepfake_kernel_page(
    data: Mapping[str, Any],
    policy: ResourcePathPolicy,
    key_prefix: str,
    kernel: Mapping[str, Any],
) -> None:
    st.subheader("AIGC鉴伪证据（二代内核）")
    st.info("适用边界：二代内核输出深伪风险筛查分与证据分带，不是概率、准确率或司法鉴定结论。")
    details = _mapping(data.get("details"))
    st.markdown("#### 1. 输入、路由与生效分支")
    columns = st.columns([2, 3])
    with columns[0]:
        query = _query_path(data, policy)
        if query is not None:
            render_image(query, "原始查询图（二代整图分支直接使用）", f"{key_prefix}_df2_query")
        else:
            st.warning("原始查询图不可用。")
    with columns[1]:
        st.metric("本次生效分支", _safe_text(kernel.get("branch_label")))
        route_columns = st.columns(2)
        short_edge = kernel.get("short_edge_px")
        route_columns[0].metric("输入有效短边", f"{short_edge} px" if short_edge else "—")
        routing_threshold = kernel.get("routing_threshold_px")
        route_columns[1].metric("路由阈值", f"{routing_threshold} px" if routing_threshold else "—")
        reason = kernel.get("routing_reason")
        if isinstance(reason, str) and reason:
            st.caption(f"路由依据：{reason}")
        st.caption("路由规则：按输入有效短边分流；整图与高分辨率照片走二代分支，小尺寸对齐裁剪走冻结人脸裁剪分支。")
    st.markdown("#### 2. 风险分与证据分带")
    score = _number(data.get("score"))
    band_columns = st.columns(3)
    band_columns[0].metric("深伪风险筛查分", _six(score))
    band_columns[1].metric("证据分带", _safe_text(kernel.get("band_label")))
    advice = "建议人工复核" if kernel.get("requires_review") else "证据不足，不单独支撑结论"
    band_columns[2].metric("处置建议", advice)
    _render_band_scale_chart(score, kernel)
    band_reason = kernel.get("band_reason")
    if isinstance(band_reason, str) and band_reason:
        st.caption(f"分带依据：{band_reason}")
    st.markdown("#### 3. 标定与验证来源")
    calibration = _mapping(details.get("calibration_set"))
    validation = _mapping(details.get("validation_set"))
    source_columns = st.columns(2)
    with source_columns[0]:
        st.markdown("**校准集（确定分带边界）**")
        st.write(f"- 名称：{_safe_text(calibration.get('name'))}")
        st.write(f"- 真人图 {_safe_text(calibration.get('n_real'))} 张 / 换脸图 {_safe_text(calibration.get('n_fake'))} 张")
        st.write(f"- 该集 AUC：{_six(calibration.get('auc'))}")
    with source_columns[1]:
        st.markdown("**验证集（未参与定标）**")
        st.write(f"- 名称：{_safe_text(validation.get('name'))}")
        st.write(f"- AUC：{_six(validation.get('auc'))}")
        interval = _list(validation.get("auc_ci95"))
        if len(interval) == 2:
            st.write(f"- 95% 置信区间：{_six(interval[0])} ~ {_six(interval[1])}")
    st.markdown("#### 4. 溯源与局限")
    provenance = _mapping(data.get("provenance"))
    st.write(f"- 输入 SHA-256：`{_safe_text(provenance.get('input_sha256'))}`")
    st.write(f"- 模型 SHA-256：`{_safe_text(provenance.get('model_sha256'))}`")
    st.write(f"- 输入尺寸：{_safe_text(details.get('original_size'))}")
    inference_seconds = _number(details.get("inference_seconds"))
    if inference_seconds is not None:
        st.write(f"- 单张推理耗时：{inference_seconds:.4f} 秒")
    else:
        st.write("- 单张推理耗时：—")
    for warning in _list(data.get("warnings")):
        st.warning(_safe_text(warning))
    st.caption(
        "局限：分带边界来自 40 张真人图与 10 张换脸图的校准集，样本量小；"
        "换脸不等于 AIGC 生成图，跨任务迁移机理尚未解释；更换数据来源后必须重新标定。"
    )
    st.caption("本页不提供空间域/频域分解：二代分支为整图单模型输出，频域特征仅保留在冻结分支的解释性输出中。")


def _render_deepfake_page(data: Mapping[str, Any], policy: ResourcePathPolicy, key_prefix: str) -> None:
    st.subheader("AIGC鉴伪证据")
    if _result_status(data) != "success":
        st.error(_safe_text(data.get("error_message"), "AIGC模块未成功完成。"))
        return
    if data.get("overall"):
        render_overall_section(data)
        render_general_aigc_section(data)
        render_face_branch_section(
            data,
            lambda path, caption: _render_evidence_image(
                path, policy, caption=caption, key=f"{key_prefix}_{path}"
            ),
        )
        if data.get("legacy_error_message"):
            st.warning(f"整图分支没有完成：{_safe_text(data.get('legacy_error_message'))}")
            return
        st.divider()
        st.markdown("#### 整图检测分支（只作提示）")
        st.caption("这一路的边界只在高清人像上标定过，普通内容上误报很高，不影响总体档位。")
    kernel = resolve_deepfake_kernel_view(data)
    if kernel is not None and kernel.get("branch") == "vit_full_image":
        _render_deepfake_kernel_page(data, policy, key_prefix, kernel)
        return
    if _mapping(data.get("details")).get("video_band_rule"):
        _render_video_deepfake_page(data)
        return
    st.info("适用边界：模型敏感性图反映当前模型输出对输入区域的响应，不等于像素级伪造定位或司法鉴定结论。")
    if kernel is not None:
        st.caption(f"本次由二代 worker 按输入短边路由到冻结分支：{kernel.get('branch_label')}")
    visual = _module_visual(data, "deepfake")
    details = _mapping(data.get("details"))
    metrics = _mapping(data.get("metrics"))
    st.markdown("#### 1. 查询图、人脸检测与模型输入")
    columns = st.columns(3)
    query = _query_path(data, policy)
    with columns[0]:
        if query is not None:
            render_image(query, "原始查询图", f"{key_prefix}_df_query")
        else:
            st.warning("原始查询图不可用。")
    with columns[1]:
        _render_evidence_image(
            _find_value(visual, "face_detection_image", "face_annotated_image", "annotated_face_image"),
            policy,
            caption="真实人脸框、关键点与序号",
            key=f"{key_prefix}_df_face_detection",
        )
    with columns[2]:
        crop_value = _find_value(visual, "face_crop_image", "selected_face_crop", "model_input_image")
        if crop_value is None:
            keyframes = details.get("keyframe_paths")
            crop_value = keyframes[0] if isinstance(keyframes, list) and keyframes else None
        _render_evidence_image(
            crop_value,
            policy,
            caption="实际送入模型的人脸裁剪",
            key=f"{key_prefix}_df_face_crop",
        )
    face_meta = _mapping(_find_value(visual, "face_detection", "face_metadata"))
    meta_columns = st.columns(4)
    meta_columns[0].metric("检测人脸数", face_meta.get("detected_face_count") if face_meta.get("detected_face_count") is not None else face_meta.get("face_count") or metrics.get("valid_frame_count") or details.get("valid_frame_count") or "—")
    meta_columns[1].metric("采用人脸", face_meta.get("selected_face_index") or "1（若检测成功）")
    crop_expand = face_meta.get("crop_expand_ratio")
    if crop_expand is None:
        crop_expand = face_meta.get("crop_expansion") or face_meta.get("expansion_ratio")
    meta_columns[2].metric("裁剪扩展", _safe_text(crop_expand))
    meta_columns[3].metric("模型输入尺寸", _safe_text(face_meta.get("model_input_size") or "224 × 224"))

    st.markdown("#### 2. 空间域模型敏感性")
    heat_columns = st.columns(2)
    with heat_columns[0]:
        _render_evidence_image(
            _find_value(visual, "spatial_heatmap", "heatmap_image", "sensitivity_heatmap"),
            policy,
            caption="空间域模型敏感性热力图",
            key=f"{key_prefix}_df_heatmap",
        )
    with heat_columns[1]:
        _render_evidence_image(
            _find_value(visual, "spatial_overlay", "heatmap_overlay", "sensitivity_overlay", "overlay_image"),
            policy,
            caption="空间域模型关注区域（叠加图）",
            key=f"{key_prefix}_df_overlay",
        )
    spatial_meta = _mapping(_find_value(visual, "spatial_explanation", "spatial_sensitivity", "spatial"))
    spatial_columns = st.columns(3)
    spatial_columns[0].metric("解释方法", _safe_text(spatial_meta.get("method") or _find_value(visual, "spatial_explanation_method")))
    spatial_columns[1].metric("空间域分数", _six(metrics.get("spatial_score") or details.get("spatial_score")))
    elapsed = _number(
        spatial_meta.get("generation_seconds")
        if spatial_meta.get("generation_seconds") is not None
        else spatial_meta.get("elapsed_seconds")
        or _find_value(visual, "spatial_explanation_elapsed_seconds")
    )
    spatial_columns[2].metric("解释图耗时", f"{elapsed:.3f} 秒" if elapsed is not None else "—")
    st.caption("该图仅展示模型敏感性区域，不表示已经确认的换脸区域。")

    st.markdown("#### 3. 频域证据")
    frequency_columns = st.columns(3)
    with frequency_columns[0]:
        _render_evidence_image(
            _find_value(visual, "frequency_input_image", "face_crop_image", "frequency_face_crop"),
            policy,
            caption="频域模块实际输入人脸",
            key=f"{key_prefix}_df_frequency_input",
        )
    with frequency_columns[1]:
        _render_evidence_image(
            _find_value(
                visual,
                "frequency_spectrum",
                "fft_log_spectrum",
                "centered_spectrum",
                "fft_log_magnitude_image",
                "fft_centered_spectrum_image",
            ),
            policy,
            caption="二维FFT中心化对数幅度谱",
            key=f"{key_prefix}_df_fft",
        )
    with frequency_columns[2]:
        _render_evidence_image(
            _find_value(visual, "frequency_profile", "radial_frequency_profile", "band_energy_chart", "radial_profile_image"),
            policy,
            caption="径向频谱/分频段能量",
            key=f"{key_prefix}_df_frequency_profile",
        )
    st.metric("频域模型分数", _six(metrics.get("frequency_score") or details.get("frequency_score")))
    st.markdown("**对当前 LogisticRegression 输出贡献绝对值最大的真实特征**")
    contribution_value = _find_value(
        visual, "frequency_contributions", "feature_contributions", "contributions"
    )
    if isinstance(contribution_value, Mapping):
        contribution_rows = _list(contribution_value.get("top_by_absolute_contribution"))
    else:
        contribution_rows = list(contribution_value) if isinstance(contribution_value, list) else []
    _render_signed_contributions(contribution_rows)

    st.markdown("#### 4. 融合与阈值判定")
    spatial = _number(metrics.get("spatial_score") or details.get("spatial_score"))
    frequency = _number(metrics.get("frequency_score") or details.get("frequency_score"))
    sw = _number(metrics.get("spatial_weight") or details.get("spatial_weight"))
    fw = _number(metrics.get("frequency_weight") or details.get("frequency_weight"))
    fused = _number(metrics.get("fusion_score") or details.get("fusion_score") or data.get("score"))
    threshold = _number(metrics.get("decision_threshold") or details.get("decision_threshold"))
    margin = _number(metrics.get("decision_margin"))
    score_columns = st.columns(5)
    score_columns[0].metric("空间域", _six(spatial))
    score_columns[1].metric("空间权重", _six(sw))
    score_columns[2].metric("频域", _six(frequency))
    score_columns[3].metric("频域权重", _six(fw))
    score_columns[4].metric("融合风险", _six(fused), _signed_six(margin), delta_color="off")
    if None not in (spatial, frequency, sw, fw, fused):
        st.code(
            f"{sw:.6f} × {spatial:.6f} + {fw:.6f} × {frequency:.6f} = {fused:.6f}",
            language=None,
        )
    _render_threshold_chart(fused, threshold)
    conclusion = _mapping(data.get("explanation")).get("decision") or data.get("label")
    st.warning(f"当前规范结论：{_safe_text(conclusion)}；与阈值差值 {_signed_six(margin)}。")


def _candidate_reference_values(candidates: Sequence[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    points: list[dict[str, Any]] = []
    means: list[dict[str, Any]] = []
    for candidate in candidates[:5]:
        if not isinstance(candidate, Mapping):
            continue
        identity = str(candidate.get("identity_name") or candidate.get("identity_id") or "unknown")
        mean = _number(candidate.get("mean_similarity") or candidate.get("score"))
        if mean is not None:
            means.append({"identity": identity, "mean": mean})
        best_path = str(candidate.get("best_reference_image") or "")
        best_file_name = str(candidate.get("best_reference_file_name") or Path(best_path).name)
        for index, raw in enumerate(_list(candidate.get("reference_scores")), 1):
            if not isinstance(raw, Mapping):
                continue
            score = _number(raw.get("similarity"))
            if score is None:
                continue
            path = str(raw.get("reference_image") or "")
            reference_file_name = str(raw.get("reference_file_name") or Path(path).name or f"参考图{index}")
            points.append(
                {
                    "identity": identity,
                    "reference": reference_file_name,
                    "similarity": score,
                    "is_best": path == best_path or reference_file_name == best_file_name,
                }
            )
    return points, means


def _render_identity_score_chart(candidates: Sequence[Any]) -> None:
    points, means = _candidate_reference_values(candidates)
    if not points:
        st.warning("本次任务未保存逐参考图真实相似度。")
        return
    spec = {
        "layer": [
            {
                "data": {"values": points},
                "mark": {"type": "circle", "size": 100, "opacity": 0.82},
                "encoding": {
                    "x": {"field": "similarity", "type": "quantitative", "title": "逐图余弦相似度", "scale": {"domain": [-1, 1]}},
                    "y": {"field": "identity", "type": "nominal", "sort": None, "title": "人员候选"},
                    "color": {"field": "is_best", "type": "nominal", "scale": {"domain": [True, False], "range": ["#b75d3e", "#315f86"]}, "title": "最佳单图"},
                    "tooltip": [
                        {"field": "identity", "title": "人员"},
                        {"field": "reference", "title": "参考图"},
                        {"field": "similarity", "type": "quantitative", "format": ".6f", "title": "余弦相似度"},
                    ],
                },
            },
            {
                "data": {"values": means},
                "mark": {"type": "tick", "thickness": 4, "size": 28, "color": "#d39b37"},
                "encoding": {
                    "x": {"field": "mean", "type": "quantitative"},
                    "y": {"field": "identity", "type": "nominal", "sort": None},
                    "tooltip": [{"field": "mean", "type": "quantitative", "format": ".6f", "title": "人员级平均"}],
                },
            },
        ],
        "height": max(260, 54 * min(5, len(candidates))),
    }
    st.vega_lite_chart(spec, width="stretch")
    st.caption("蓝/橙圆点为逐张参考图余弦相似度，金色刻线为人员级平均值；排序依据是人员级平均，不是单张图或身份概率。")


def _render_identity_page(data: Mapping[str, Any], policy: ResourcePathPolicy, key_prefix: str) -> None:
    st.subheader("人脸身份比对证据")
    st.info("适用边界：这是当前身份库内的闭集候选排序，余弦相似度不是身份概率，第一候选尚未确认。")
    if _result_status(data) != "success":
        st.error(_safe_text(data.get("error_message"), "身份模块未成功完成。"))
        return
    visual = _module_visual(data, "identity")
    metrics = _mapping(data.get("metrics"))
    candidates = _list(data.get("candidates"))
    st.markdown("#### 1. 人脸检测、关键点与对齐")
    image_columns = st.columns(4)
    query = _query_path(data, policy)
    with image_columns[0]:
        if query is not None:
            render_image(query, "查询原图", f"{key_prefix}_id_query")
        else:
            st.warning("查询原图不可用。")
    with image_columns[1]:
        _render_evidence_image(
            _find_value(visual, "face_detection_image", "query_face_annotated"),
            policy,
            caption="真实人脸框与五点关键点",
            key=f"{key_prefix}_id_detection",
        )
    with image_columns[2]:
        _render_evidence_image(
            _find_value(visual, "unaligned_face_image", "face_crop_before_alignment", "selected_face_crop"),
            policy,
            caption="对齐前查询人脸",
            key=f"{key_prefix}_id_before",
        )
    with image_columns[3]:
        _render_evidence_image(
            _find_value(visual, "aligned_face_image", "query_aligned_face"),
            policy,
            caption="对齐后标准人脸",
            key=f"{key_prefix}_id_aligned",
        )
    stat_columns = st.columns(4)
    stat_columns[0].metric("检测人脸数", metrics.get("query_face_count") if metrics.get("query_face_count") is not None else "—")
    stat_columns[1].metric("检测置信度", _six(metrics.get("face_detection_score")))
    stat_columns[2].metric("特征维度", metrics.get("embedding_dimension") or "—")
    stat_columns[3].metric("特征状态", "已提取" if candidates else "未生成候选")

    st.markdown("#### 2. Top-5 当前库内候选")
    top_score = _number(_mapping(candidates[0]).get("mean_similarity")) if candidates else None
    cards = st.columns(min(5, max(1, len(candidates))))
    for index, raw in enumerate(candidates[:5]):
        candidate = _mapping(raw)
        with cards[index]:
            with st.container(border=True):
                st.markdown(f"**第 {candidate.get('rank') or index + 1} 名 · {_safe_text(candidate.get('identity_name') or candidate.get('identity_id'))}**")
                image_value = (
                    candidate.get("candidate_card_image")
                    or candidate.get("best_reference_image")
                    or candidate.get("reference_image")
                )
                image_path = resolve_evidence_path(image_value, policy, kind="identity", label="身份候选参考图")
                if image_path is not None and image_path.is_file():
                    st.image(str(image_path), width="stretch")
                else:
                    st.caption("参考图不可用")
                mean = _number(candidate.get("mean_similarity") or candidate.get("score"))
                st.metric("人员级平均", _six(mean))
                st.metric("最佳单图", _six(candidate.get("best_reference_score")))
                valid_count = candidate.get("valid_reference_count")
                if valid_count is None:
                    valid_count = candidate.get("reference_count")
                st.caption(f"有效参考图 {valid_count if valid_count is not None else '—'} 张")
                st.caption(f"距第一候选 {_signed_six((mean - top_score) if mean is not None and top_score is not None else None)}")
                st.caption(f"最佳文件：{_safe_text(candidate.get('best_reference_file_name'), _basename(candidate.get('best_reference_image') or candidate.get('reference_image')))}")

    st.markdown("#### 3. 人员级分数形成过程")
    _render_identity_score_chart(candidates)
    skipped: list[Any] = []
    for candidate in candidates[:5]:
        if isinstance(candidate, Mapping):
            skipped.extend(_list(candidate.get("skipped_reference_images")))
    if not skipped:
        skipped = _find_list(visual, "skipped_reference_images", "skipped_references")
    if skipped:
        with st.expander("被跳过参考图及原因"):
            st.dataframe(skipped, width="stretch", hide_index=True)
    else:
        skipped_count = metrics.get("skipped_library_image_count")
        st.caption(f"跳过参考图数量：{skipped_count if skipped_count is not None else '本次未记录详细清单'}")

    st.markdown("#### 4. 查询人脸与第一候选最佳参考图")
    compare_columns = st.columns(2)
    with compare_columns[0]:
        aligned = _find_value(visual, "aligned_face_image", "query_aligned_face")
        _render_evidence_image(aligned, policy, caption="查询人脸（对齐后）", key=f"{key_prefix}_id_compare_query")
    top = _mapping(candidates[0]) if candidates else {}
    with compare_columns[1]:
        best_reference_visual = _mapping(visual.get("best_reference"))
        top_path = resolve_evidence_path(
            best_reference_visual.get("face_detection_image")
            or top.get("candidate_card_image")
            or top.get("best_reference_image")
            or top.get("reference_image"),
            policy,
            kind="identity",
            label="最佳参考图",
        )
        if top_path is not None and top_path.is_file():
            render_image(top_path, "第一候选最佳参考图", f"{key_prefix}_id_compare_reference")
        else:
            st.warning("第一候选最佳参考图不可用。")
    compare_stats = st.columns(3)
    compare_stats[0].metric("最佳单图余弦相似度", _six(top.get("best_reference_score")))
    compare_stats[1].metric("人员级平均余弦相似度", _six(top.get("mean_similarity") or top.get("score")))
    compare_stats[2].metric("Top1与Top2差值", _six(metrics.get("top1_top2_margin")))
    comparison_image = _find_value(visual, "best_reference_comparison_image")
    if comparison_image is not None:
        _render_evidence_image(
            comparison_image,
            policy,
            caption="查询人脸与第一候选最佳参考图证据对照",
            key=f"{key_prefix}_id_best_comparison",
            kind="identity",
        )


def _render_source_score_chart(candidates: Sequence[Any]) -> None:
    values: list[dict[str, Any]] = []
    for raw in candidates[:5]:
        candidate = _mapping(raw)
        name = f"#{candidate.get('rank')} {_basename(candidate.get('file_path'))}"
        for label, key in (("全局分", "global_score"), ("局部分", "local_score"), ("最终分", "final_score")):
            value = _number(candidate.get(key))
            if value is not None:
                values.append({"candidate": name, "score_type": label, "score": value})
    if not values:
        st.warning("本次任务没有可绘制的素材候选分数。")
        return
    spec = {
        "data": {"values": values},
        "mark": {"type": "bar", "cornerRadiusEnd": 3},
        "encoding": {
            "x": {"field": "candidate", "type": "nominal", "title": "候选"},
            "y": {"field": "score", "type": "quantitative", "title": "检索分数"},
            "xOffset": {"field": "score_type"},
            "color": {"field": "score_type", "type": "nominal", "scale": {"range": ["#617a8a", "#b0834c", "#315f86"]}},
            "tooltip": [
                {"field": "candidate", "title": "候选"},
                {"field": "score_type", "title": "分数类型"},
                {"field": "score", "type": "quantitative", "format": ".6f", "title": "数值"},
            ],
        },
        "height": 320,
    }
    st.vega_lite_chart(spec, width="stretch")


def _render_global_components(candidates: Sequence[Any], visual: Mapping[str, Any]) -> None:
    values: list[dict[str, Any]] = []
    visual_candidates = _find_list(visual, "top_candidates", "candidates")
    global_scoring = _mapping(visual.get("global_scoring"))
    scoring_candidates = _list(global_scoring.get("candidates"))
    by_file: dict[str, Mapping[str, Any]] = {}
    for raw in visual_candidates:
        if isinstance(raw, Mapping):
            by_file[_basename(raw.get("file_path") or raw.get("candidate_path") or raw.get("file_name"))] = raw
    for raw in scoring_candidates:
        if isinstance(raw, Mapping):
            by_file[_basename(raw.get("file_path") or raw.get("candidate_path") or raw.get("file_name"))] = raw
    for raw in candidates[:5]:
        candidate = _mapping(raw)
        name = _basename(candidate.get("file_path"))
        components = candidate.get("global_score_components") or candidate.get("global_components")
        if not isinstance(components, Mapping):
            visual_candidate = by_file.get(name, {})
            components = (
                visual_candidate.get("components") or visual_candidate.get("global_components")
                if isinstance(visual_candidate, Mapping)
                else None
            )
        if not isinstance(components, Mapping):
            continue
        for component, value in components.items():
            if component in {"weights", "weighted_terms", "formula", "score", "border_common_pixel_count"}:
                continue
            score = _number(value)
            if score is not None:
                values.append({"candidate": name, "component": str(component), "score": score})
    if not values:
        st.info("现有任务只保存了最终全局分，本次未生成可严格追溯的全局子分量；不会为了绘图凭空拆分。")
        return
    spec = {
        "data": {"values": values},
        "mark": {"type": "bar"},
        "encoding": {
            "x": {"field": "candidate", "type": "nominal", "title": "候选文件"},
            "y": {"field": "score", "type": "quantitative", "title": "实际子分量"},
            "xOffset": {"field": "component"},
            "color": {"field": "component", "type": "nominal", "title": "全局特征分量"},
            "tooltip": [
                {"field": "candidate", "title": "候选"},
                {"field": "component", "title": "分量"},
                {"field": "score", "type": "quantitative", "format": ".6f", "title": "分数"},
            ],
        },
        "height": 300,
    }
    st.vega_lite_chart(spec, width="stretch")


def _render_source_page(data: Mapping[str, Any], policy: ResourcePathPolicy, key_prefix: str) -> None:
    st.subheader("原始素材溯源证据")
    st.info("适用边界：结果是当前素材库内的候选排序，检索相似度不是来源概率，不能单独确认原始素材。")
    if _result_status(data) != "success":
        st.error(_safe_text(data.get("error_message"), "素材模块未成功完成。"))
        return
    metrics = _mapping(data.get("metrics"))
    visual = _module_visual(data, "source")
    candidates = _list(data.get("candidates"))
    st.markdown("#### 1. 查询图与真实检索输入")
    query_columns = st.columns(3)
    query = _query_path(data, policy)
    with query_columns[0]:
        if query is not None:
            render_image(query, "原始查询图", f"{key_prefix}_src_query")
        else:
            st.warning("查询原图不可用。")
    with query_columns[1]:
        _render_evidence_image(
            _find_value(visual, "face_detection_image", "query_face_annotated"),
            policy,
            caption="查询图真实人脸检测框",
            key=f"{key_prefix}_src_face_detection",
        )
    with query_columns[2]:
        masked = _find_value(visual, "face_masked_image", "masked_query_image") or metrics.get("masked_query_path")
        _render_evidence_image(masked, policy, caption="后端实际使用的人脸屏蔽图", key=f"{key_prefix}_src_masked")
    st.caption(
        f"是否应用屏蔽：{'是' if metrics.get('face_mask_applied') is True else '否' if metrics.get('face_mask_applied') is False else '未记录'}。"
        "屏蔽用于降低查询人脸身份对背景/素材检索的干扰，不改变素材库候选内容。"
    )

    st.markdown("#### 2. 全库粗排漏斗")
    total = _integer(metrics.get("library_file_count"))
    decoded = _integer(metrics.get("library_decoded"))
    skipped = _integer(metrics.get("library_skipped_count"))
    shortlist = _integer(metrics.get("shortlist_size"))
    displayed = min(5, len(candidates))
    funnel = st.columns(5)
    labels = [
        ("素材库文件", total),
        ("成功读取", decoded),
        ("跳过文件", skipped),
        ("进入局部复核", shortlist),
        ("最终展示", displayed),
    ]
    for index, (label, value) in enumerate(labels):
        funnel[index].metric(label, value if value is not None else "—")
        if index < 4:
            funnel[index].caption("→ 下一真实处理阶段")

    st.markdown("#### 3. 全局特征组成与候选对比")
    _render_global_components(candidates, visual)
    _render_source_score_chart(candidates)

    st.markdown("#### 4. Top-5 当前库内候选")
    cards = st.columns(min(5, max(1, len(candidates))))
    top_final = _number(_mapping(candidates[0]).get("final_score")) if candidates else None
    visual_top_candidates = {
        int(item.get("rank")): item
        for item in _find_list(visual, "top_candidates")
        if isinstance(item, Mapping) and isinstance(item.get("rank"), int)
    }
    for index, raw in enumerate(candidates[:5]):
        candidate = _mapping(raw)
        with cards[index]:
            with st.container(border=True):
                st.markdown(f"**第 {candidate.get('rank') or index + 1} 名 · {_basename(candidate.get('file_path'))}**")
                visual_candidate = _mapping(visual_top_candidates.get(int(candidate.get("rank") or index + 1)))
                candidate_image = (
                    candidate.get("candidate_image_artifact")
                    or visual_candidate.get("candidate_image")
                    or candidate.get("file_path")
                )
                path = resolve_evidence_path(candidate_image, policy, kind="source", label="素材候选")
                if path is not None and path.is_file():
                    st.image(str(path), width="stretch")
                else:
                    st.caption("候选图不可用")
                st.metric("最终分", _six(candidate.get("final_score")))
                st.caption(f"全局 {_six(candidate.get('global_score'))} · 局部 {_six(candidate.get('local_score'))}")
                st.caption(f"Lowe有效匹配 {candidate.get('good_match_count') or 0} 点")
                st.caption(f"RANSAC {candidate.get('ransac_inlier_count') or 0} 点 / {_six(candidate.get('ransac_inlier_ratio'))}")
                final = _number(candidate.get("final_score"))
                st.caption(f"距第一候选 {_signed_six((final - top_final) if final is not None and top_final is not None else None)}")

    st.markdown("#### 5. SIFT、Lowe与RANSAC几何证据（第一候选）")
    match_columns = st.columns(2)
    top = _mapping(candidates[0]) if candidates else {}
    with match_columns[0]:
        _render_evidence_image(
            _find_value(visual, "query_keypoints", "query_keypoints_image"),
            policy,
            caption="查询图SIFT关键点",
            key=f"{key_prefix}_src_query_keypoints",
        )
    with match_columns[1]:
        _render_evidence_image(
            _find_value(visual, "candidate_keypoints", "candidate_keypoints_image"),
            policy,
            caption="第一候选SIFT关键点",
            key=f"{key_prefix}_src_candidate_keypoints",
        )
    match_columns = st.columns(2)
    with match_columns[0]:
        all_matches = _find_value(visual, "matches_all", "matches_lowe", "lowe_matches_image") or top.get("visualization_path")
        _render_evidence_image(all_matches, policy, caption="Lowe筛选后的全部有效匹配线", key=f"{key_prefix}_src_matches_all")
    with match_columns[1]:
        inlier_matches = _find_value(
            visual,
            "matches_inliers_outliers",
            "ransac_inliers_outliers_image",
            "matches_inliers",
            "ransac_inliers_image",
            "match_inliers",
        ) or top.get("visualization_path")
        _render_evidence_image(inlier_matches, policy, caption="RANSAC内点（绿）与外点（红/灰）", key=f"{key_prefix}_src_matches_inliers")
    homography = _find_value(visual, "homography_overlay", "homography_projection")
    if homography is not None:
        _render_evidence_image(homography, policy, caption="单应性投影/几何对齐叠加", key=f"{key_prefix}_src_homography")
    local_components = _mapping(top.get("local_score_components"))
    local_stats = st.columns(4)
    local_stats[0].metric("Lowe有效匹配", top.get("good_match_count") or "—")
    local_stats[1].metric("RANSAC内点", top.get("ransac_inlier_count") or "—")
    local_stats[2].metric("RANSAC内点率", _six(top.get("ransac_inlier_ratio")))
    local_stats[3].metric("内点数归一化", _six(local_components.get("normalized_inlier_count")))
    st.code(_safe_text(local_components.get("formula"), "本次未记录局部分公式"), language=None)

    st.markdown("#### 6. 最终分数形成")
    global_score = _number(top.get("global_score"))
    local_score = _number(top.get("local_score"))
    gw = _number(top.get("global_weight") or metrics.get("global_weight"))
    lw = _number(top.get("local_weight") or metrics.get("local_weight"))
    final_score = _number(top.get("final_score"))
    final_columns = st.columns(5)
    final_columns[0].metric("全局分", _six(global_score))
    final_columns[1].metric("全局权重", _six(gw))
    final_columns[2].metric("局部分", _six(local_score))
    final_columns[3].metric("局部权重", _six(lw))
    final_columns[4].metric("最终分", _six(final_score))
    if None not in (global_score, local_score, gw, lw, final_score):
        st.code(f"{gw:.6f} × {global_score:.6f} + {lw:.6f} × {local_score:.6f} = {final_score:.6f}", language=None)
    second = _mapping(candidates[1]) if len(candidates) > 1 else {}
    st.caption(f"第二候选最终分 {_six(second.get('final_score'))}；第一名与第二名差值 {_six(metrics.get('top1_top2_margin'))}。")


def _render_evidence_chain_page(
    deepfake: Mapping[str, Any],
    identity: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    task_dir: Path,
    task_status: Mapping[str, Any] | None,
    policy: ResourcePathPolicy,
    task_artifact_manifest: Mapping[str, Any] | None = None,
) -> None:
    st.subheader("证据链与完整报告")
    st.info("本页只汇总当前任务实际结果、配置、模型/输入哈希、进度事件和已落盘证据；不生成跨模块综合概率。")
    st.markdown("#### 证据矩阵（研判卡）")
    st.caption("逐模块证据质量分级；模块之间不做数值相加，也不生成综合可信概率。")
    matrix = build_evidence_matrix(deepfake, identity, source)
    module_labels = {"deepfake": "AIGC鉴伪", "identity": "人脸身份比对", "source": "原始素材溯源"}
    matrix_rows: list[dict[str, Any]] = []
    for module_key, module_label in module_labels.items():
        block = _mapping(_mapping(matrix.get("modules")).get(module_key))
        decision = _mapping(block.get("decision"))
        signal = _mapping(decision.get("primary_signal"))
        matrix_rows.append(
            {
                "模块": module_label,
                "证据质量": _safe_text(block.get("grade_label")),
                "主要数值": _six(block.get("primary_value")),
                "数值含义": _safe_text(block.get("primary_label")),
                "判定判据": _safe_text(signal.get("name")) if signal else "—",
            }
        )
    st.dataframe(matrix_rows, width="stretch", hide_index=True)
    quality = _mapping(matrix.get("input_quality"))
    coverage = _mapping(matrix.get("library_coverage"))
    geometry = _mapping(matrix.get("geometry"))
    open_set = _mapping(matrix.get("open_set_risk"))
    detail_columns = st.columns(4)
    detail_columns[0].metric("查询图像质量", _safe_text(quality.get("grade")))
    detail_columns[1].metric("身份库覆盖", "已确认" if coverage.get("complete") else "覆盖不足")
    detail_columns[2].metric("几何一致性", _safe_text(geometry.get("grade")))
    detail_columns[3].metric("开放集风险", _safe_text(open_set.get("level")))
    for note in _list(open_set.get("notes")):
        st.warning(_safe_text(note))
    for note in _list(matrix.get("warnings")):
        st.warning(_safe_text(note))
    with st.expander("证据矩阵完整结构（可逐项复核）"):
        st.json(matrix)
    authoritative = _mapping(task_artifact_manifest)
    authoritative_records = authoritative.get("artifacts")
    manifest = (
        [dict(item) for item in authoritative_records if isinstance(item, Mapping)]
        if isinstance(authoritative_records, list)
        else collect_artifact_manifest(deepfake, identity, source)
    )
    st.markdown("#### 任务级 artifact manifest")
    if manifest:
        rows: list[dict[str, Any]] = []
        for item in manifest:
            path_value = _path_value(item.get("path") or item.get("relative_path")) or ""
            path = resolve_evidence_path(path_value, policy, label=f"artifact {item.get('id')}") if path_value else None
            rows.append(
                {
                    "模块": item.get("module"),
                    "证据类型": item.get("kind") or item.get("type"),
                    "相对路径": path_value,
                    "状态": item.get("status") or ("present" if path and path.is_file() else "missing"),
                    "大小(字节)": item.get("size_bytes") or (path.stat().st_size if path and path.is_file() else None),
                    "SHA-256": item.get("sha256"),
                    "说明": item.get("description"),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.warning("该任务是旧结构或本次未生成 artifact manifest；原结果仍可读取，不会因此崩溃。")

    st.markdown("#### 输入、模型与配置溯源")
    provenance_rows: list[dict[str, Any]] = []
    for module, result in (("AIGC", deepfake), ("身份", identity), ("素材", source)):
        provenance = _mapping(result.get("provenance"))
        for key, value in provenance.items():
            if "path" in key.lower() and isinstance(value, str):
                value = _basename(value)
            provenance_rows.append({"模块": module, "字段": key, "值": _table_value(value)})
    if provenance_rows:
        st.dataframe(provenance_rows, width="stretch", hide_index=True)

    st.markdown("#### 实际运行事件时间线")
    events = load_progress_events(task_dir)
    if events:
        rows = []
        for event in events[-300:]:
            rows.append(
                {
                    "时间": event.get("timestamp"),
                    "模块": event.get("module"),
                    "阶段": event.get("stage"),
                    "状态": event.get("status"),
                    "真实进度": event.get("progress_percent"),
                    "耗时(秒)": event.get("elapsed_seconds"),
                    "说明": event.get("message"),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True, height=420)
    else:
        st.caption("该任务没有保存可读进度事件。")
    status = _mapping(task_status)
    if status:
        with st.expander("任务状态与各模块原始状态"):
            st.json(status)


def _read_task_json(task_dir: Path, name: str) -> dict[str, Any]:
    """读取任务目录下的 JSON；缺失或损坏时返回空字典，不影响页面其余部分。"""

    try:
        payload = json.loads((task_dir / name).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _video_frame_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """把视频级鉴伪结果里的逐帧明细转成可复核的表格行。"""

    details = _mapping(data.get("details"))
    frames = details.get("per_frame")
    rows: list[dict[str, Any]] = []
    if not isinstance(frames, list):
        return rows
    for item in frames:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            {
                "时间点（秒）": item.get("timestamp_seconds"),
                "关键帧": item.get("file_name"),
                "深伪风险分": format_score(item.get("score")),
                "帧级分带": _safe_text(item.get("band_label") or item.get("band")),
                "分带来源": _safe_text(item.get("band_source")),
                "检测分支": _safe_text(item.get("branch")),
            }
        )
    return rows


def _render_video_deepfake_page(data: Mapping[str, Any]) -> None:
    """视频任务的 AIGC 鉴伪页：只呈现帧级证据与聚合口径，不冒充单图结论。"""

    details = _mapping(data.get("details"))
    st.info(
        "视频输入：鉴伪在多个关键帧上逐帧运行；视频级分数取帧级中位数，视频级分带由帧级分带"
        "的一致性规则导出。该分数不是概率，也不是对整段视频的鉴定结论。"
    )
    scored_frames = details.get("frames_scored")
    total_frames = details.get("frames_total")
    columns = st.columns(4)
    columns[0].metric("视频级分带", _safe_text(details.get("band_label") or details.get("band")))
    columns[1].metric("视频级分数", format_score(data.get("score")))
    columns[2].metric("成功鉴伪帧", f"{scored_frames}/{total_frames}")
    columns[3].metric("分带口径", _safe_text(details.get("band_rule")))
    statement = _safe_text(details.get("consistency_statement"), "")
    if statement:
        st.caption(statement)
    branches = _mapping(details.get("frame_branches"))
    if branches:
        summary = "；".join(f"{key} × {value}" for key, value in branches.items())
        st.caption(f"本次帧级检测分支：{summary}")
    rows = _video_frame_rows(data)
    if rows:
        st.markdown("#### 逐帧鉴伪结果")
        st.table(rows)
    for warning in data.get("warnings") or []:
        if isinstance(warning, str) and warning.strip():
            st.warning(warning)
    st.markdown("#### 结论边界")
    for item in data.get("limitations") or []:
        if isinstance(item, str) and item.strip():
            st.write(f"- {item}")


def render_video_input_panel(
    task_dir: Path,
    policy: ResourcePathPolicy,
    *,
    deepfake: Mapping[str, Any],
    key_prefix: str,
    max_frames: int = 5,
) -> bool:
    """视频任务的关键帧时间轴与视频级模块输出；非视频任务返回 False。"""

    summary = _read_task_json(task_dir, "video_summary.json")
    if not summary:
        return False
    video = _mapping(summary.get("video"))
    modules = _mapping(summary.get("modules"))
    key_frames = [
        item for item in (summary.get("key_frames") or []) if isinstance(item, Mapping)
    ]
    st.markdown("### 视频输入：关键帧与逐帧聚合")
    columns = st.columns(4)
    duration = video.get("duration_seconds")
    if isinstance(duration, (int, float)):
        duration_label = f"{float(duration):.2f} 秒"
    else:
        duration_label = "—"
    columns[0].metric("视频时长", duration_label)
    fps = video.get("fps")
    fps_label = f"{float(fps):g} fps" if isinstance(fps, (int, float)) else "—"
    columns[1].metric("帧率", fps_label)
    columns[2].metric("关键帧", f"{len(key_frames)} 帧")
    digest = str(summary.get("video_sha256") or "")
    columns[3].metric("视频 SHA-256", f"{digest[:12]}…" if digest else "—")
    st.caption(
        "流程：抽帧 → 逐帧人脸检测 → 按清晰度/亮度/人脸数选关键帧 → 逐帧运行模块 → 视频级聚合。"
        "鉴伪取帧级分数中位数与分带一致性，身份取跨帧平均余弦相似度，素材只检索首要帧；三个模块的数值不相加。"
    )
    deepfake_module = _mapping(modules.get("deepfake"))
    if deepfake_module.get("band_rule") == "frozen_threshold":
        st.warning(
            "本次帧级鉴伪由冻结内核给出判定（没有校准分带），视频级分带退化为阈值二值判定，只能视为占位结果；"
            "启用 AIGC 持久模型服务后关键帧会走二代 ViT 分支。"
        )
    visible_frames = key_frames[:max_frames]
    if visible_frames:
        frame_columns = st.columns(len(visible_frames))
        for index, frame in enumerate(visible_frames):
            frame_name = str(frame.get("file_name") or "")
            timestamp = frame.get("timestamp_seconds")
            with frame_columns[index]:
                path = resolve_evidence_path(
                    f"key_frames/{frame_name}",
                    policy,
                    kind="query",
                    label="关键帧",
                )
                if path is not None and path.is_file():
                    render_image(
                        path,
                        f"{timestamp}s 关键帧",
                        f"{key_prefix}_video_frame_{index}",
                    )
                else:
                    st.warning(f"关键帧不可用：{_safe_text(frame_name)}")
                quality = format_score(frame.get("quality_score"))
                st.caption(f"{_safe_text(frame_name)}｜{timestamp}s｜质量分 {quality}")
                reasons = frame.get("selection_reasons")
                if isinstance(reasons, list):
                    reason_text = "；".join(
                        str(item).strip() for item in reasons if str(item).strip()
                    )
                    if reason_text:
                        st.caption(f"选帧理由：{reason_text}")
    rows: list[dict[str, Any]] = []
    if deepfake_module:
        deepfake_frames = deepfake_module.get("frames_scored")
        rows.append(
            {
                "模块": "AIGC鉴伪",
                "视频级输出": _safe_text(
                    deepfake_module.get("band_label") or deepfake_module.get("band")
                ),
                "数值": format_score(deepfake_module.get("score")),
                "聚合规则": _safe_text(deepfake_module.get("rule")),
                "证据范围": f"{deepfake_frames} 帧",
            }
        )
    identity_module = _mapping(modules.get("identity"))
    if identity_module:
        faces = identity_module.get("frames_with_faces")
        support = identity_module.get("top1_frame_support")
        rows.append(
            {
                "模块": "人脸身份比对",
                "视频级输出": _safe_text(identity_module.get("top1_identity_name")),
                "数值": format_score(identity_module.get("top1_mean_similarity")),
                "聚合规则": _safe_text(identity_module.get("rule")),
                "证据范围": f"有脸帧 {faces}，top1 支持帧 {support}",
            }
        )
    source_module = _mapping(modules.get("source"))
    if source_module:
        rows.append(
            {
                "模块": "原始素材检索",
                "视频级输出": _safe_text(source_module.get("top1_candidate")),
                "数值": format_score(source_module.get("top1_score")),
                "聚合规则": _safe_text(source_module.get("rule")),
                "证据范围": "首要帧",
            }
        )
    if rows:
        st.markdown("#### 视频级模块输出")
        st.table(rows)
    frame_rows = _video_frame_rows(deepfake)
    if frame_rows:
        with st.expander("逐帧鉴伪明细（帧级分数、分带来源与路由分支）"):
            st.table(frame_rows)
    return True


def render_evidence_workspace(
    *,
    deepfake: Mapping[str, Any],
    identity: Mapping[str, Any],
    source: Mapping[str, Any],
    policy: ResourcePathPolicy,
    task_dir: Path,
    task_status: Mapping[str, Any] | None = None,
    final_report: Mapping[str, Any] | None = None,
    task_artifact_manifest: Mapping[str, Any] | None = None,
    product_version: str = "0.4.0",
    key_prefix: str = "evidence",
) -> str:
    request_key = f"{key_prefix}_page_request"
    navigation_key = f"{key_prefix}_page"
    requested = st.session_state.pop(request_key, None)
    if requested in PAGES:
        st.session_state[navigation_key] = requested
    if navigation_key not in st.session_state:
        st.session_state[navigation_key] = PAGES[0]
    selected = st.radio(
        "研判页面",
        PAGES,
        key=navigation_key,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()
    provenance = _read_task_json(task_dir, "file_provenance_result.json")
    if selected == "研判总览":
        overview = build_task_overview(
            deepfake,
            identity,
            source,
            task_status=task_status,
            final_report=final_report,
            task_dir=task_dir,
            product_version=product_version,
        )
        _render_key_findings(deepfake, identity, source, provenance)
        _render_task_header(overview, video=_read_task_json(task_dir, "video_summary.json"))
        st.markdown("#### 各模块的证据")
        st.caption("各模块解决不同问题；本页并列呈现证据，不相加、不生成所谓综合可信概率。")
        _render_overview_cards(
            deepfake,
            identity,
            source,
            request_key=request_key,
            key_prefix=key_prefix,
        )
        if render_video_input_panel(
            task_dir,
            policy,
            deepfake=deepfake,
            key_prefix=key_prefix,
        ):
            st.info(
                "视频任务的关键帧时间轴与逐帧聚合口径见上；逐帧明细与帧级证据文件都在任务目录内，可逐项复核。"
            )
    elif selected == "AIGC鉴伪证据":
        _render_deepfake_page(deepfake, policy, key_prefix)
    elif selected == "文件标识":
        render_provenance_page(provenance)
    elif selected == "人脸身份比对":
        _render_identity_page(identity, policy, key_prefix)
    elif selected == "原始素材溯源":
        _render_source_page(source, policy, key_prefix)
    else:
        _render_evidence_chain_page(
            deepfake,
            identity,
            source,
            task_dir=task_dir,
            task_status=task_status,
            policy=policy,
            task_artifact_manifest=task_artifact_manifest,
        )
    return str(selected)


__all__ = [
    "PAGES",
    "build_key_findings",
    "build_task_overview",
    "collect_artifact_manifest",
    "render_evidence_workspace",
    "resolve_evidence_path",
]
