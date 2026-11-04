from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import streamlit as st

from components.explainability import (
    build_deepfake_explanation,
    build_identity_explanation,
    build_source_explanation,
    display_name,
    redact_text,
)
from components.path_security import PathContainmentError, ResourcePathPolicy


UNCERTAIN_DEEPFAKE_LABELS = {
    "low_confidence",
    "insufficient",
    "insufficient_evidence",
    "no_face",
    "undetermined",
    "unknown",
}


def format_score(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def _format_six(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "—"


def _optional_score(value: Any) -> str:
    return format_score(value) if value is not None else "—"


def _skip_reason(data: Mapping[str, Any]) -> str:
    details = data.get("details")
    if isinstance(details, Mapping) and details.get("reason"):
        return redact_text(details["reason"])
    return redact_text(data.get("error_message") or "本次任务未选择该模块。")


def _deepfake_evidence(data: dict[str, Any]) -> dict[str, str]:
    explanation = build_deepfake_explanation(data)
    status = explanation["status"]
    if status == "skipped":
        return {"title": "AI换脸/深伪风险甄别", "state": "info", "headline": "未运行", "detail": explanation["summary"]}
    if status != "success":
        return {"title": "AI换脸/深伪风险甄别", "state": "error", "headline": "执行失败", "detail": explanation["summary"]}
    uncertain = data.get("is_deepfake") is None or str(data.get("label") or "").lower() in UNCERTAIN_DEEPFAKE_LABELS
    state = "warning" if data.get("is_deepfake") is not False or uncertain else "success"
    detail = explanation["summary"]
    if str(data.get("label") or "").lower() == "no_face":
        detail += " 这是输入人脸证据不足，不是模型加载失败。"
    elif uncertain:
        detail += " 证据不足，不能解释为 REAL。"
    return {
        "title": "AI换脸/深伪风险甄别",
        "state": state,
        "headline": explanation["headline"].replace("检测结论：", ""),
        "detail": detail,
    }


def _identity_evidence(data: dict[str, Any]) -> dict[str, str]:
    explanation = build_identity_explanation(data)
    state = "info" if explanation["status"] == "skipped" else "error" if explanation["status"] == "error" else "warning" if not data.get("candidates") else "success"
    return {"title": "人脸身份搜索", "state": state, "headline": explanation["headline"], "detail": explanation["summary"]}


def _source_evidence(data: dict[str, Any]) -> dict[str, str]:
    explanation = build_source_explanation(data)
    state = "info" if explanation["status"] == "skipped" else "error" if explanation["status"] == "error" else "warning" if not data.get("candidates") else "success"
    return {"title": "原始底图搜索", "state": state, "headline": explanation["headline"], "detail": explanation["summary"]}


def build_evidence_summary(
    deepfake: dict[str, Any], identity: dict[str, Any], source: dict[str, Any]
) -> dict[str, dict[str, str]]:
    """Build three independent evidence cards without a fused probability."""

    return {
        "deepfake": _deepfake_evidence(deepfake),
        "identity": _identity_evidence(identity),
        "source": _source_evidence(source),
    }


def render_evidence_summary(
    deepfake: dict[str, Any], identity: dict[str, Any], source: dict[str, Any]
) -> None:
    st.subheader("最终证据汇总")
    st.info("三个模块解决不同问题，本区只并列汇总证据，不计算、不暗示任何跨模块综合概率。")
    summary = build_evidence_summary(deepfake, identity, source)
    columns = st.columns(3)
    for column, key in zip(columns, ("deepfake", "identity", "source"), strict=True):
        item = summary[key]
        with column:
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
                getattr(st, item["state"], st.info)(item["headline"])
                st.caption(item["detail"])


def identity_view_items(
    data: dict[str, Any], path_policy: ResourcePathPolicy
) -> list[dict[str, Any]]:
    output = []
    for candidate in data["candidates"]:
        best_path = candidate.get("best_reference_image") or candidate["reference_image"]
        output.append(
            {
                **candidate,
                "resolved_reference_image": path_policy.resolve(
                    candidate["reference_image"], "identity", label=f"identity rank {candidate['rank']} reference_image"
                ),
                "resolved_best_reference_image": path_policy.resolve(
                    best_path, "identity", label=f"identity rank {candidate['rank']} best_reference_image"
                ),
            }
        )
    return output


def source_view_items(
    data: dict[str, Any], path_policy: ResourcePathPolicy
) -> list[dict[str, Any]]:
    return [
        {
            **candidate,
            "resolved_file_path": path_policy.resolve(
                candidate["file_path"], "source", label=f"source rank {candidate['rank']} file_path"
            ),
            "resolved_visualization_path": path_policy.resolve(
                candidate["visualization_path"], "visualization", label=f"source rank {candidate['rank']} visualization_path"
            ),
        }
        for candidate in data["candidates"]
    ]


@st.dialog("图片预览", width="large")
def show_image_dialog(image_path: str, caption: str) -> None:
    path = Path(image_path)
    if not path.is_file():
        st.error("图片已不存在，无法预览。")
        return
    st.image(str(path), caption=caption, width="stretch")


def render_image(image_path: Path, caption: str, key: str) -> None:
    if not image_path.is_file():
        st.warning(f"{caption}不存在或暂不可用（{display_name(image_path)}）。")
        return
    st.image(str(image_path), caption=caption, width="stretch")
    if st.button("点击放大", key=f"zoom_{key}"):
        show_image_dialog(str(image_path), caption)


def render_result_header(data: dict[str, Any]) -> None:
    left, right = st.columns(2)
    left.write(f"**查询图片：** `{display_name(data.get('query_image'))}`")
    right.write(f"**模块耗时：** `{format_score(data.get('runtime_seconds'))} 秒`")


def _render_basis(explanation: Mapping[str, Any]) -> None:
    st.markdown("**结论依据**")
    basis = explanation.get("basis")
    if not isinstance(basis, list) or not basis:
        st.caption("当前结果没有足够的结构化中间指标可生成进一步说明。")
        return
    for item in basis:
        st.markdown(f"- {redact_text(item)}")


def _render_warnings(explanation: Mapping[str, Any]) -> None:
    warnings = explanation.get("warnings")
    if isinstance(warnings, list):
        for item in warnings:
            st.warning(redact_text(item))


def _render_limitations(explanation: Mapping[str, Any]) -> None:
    limitations = explanation.get("limitations")
    if isinstance(limitations, list) and limitations:
        st.markdown("**适用边界**")
        for item in limitations:
            st.caption(redact_text(item))


def render_deepfake_result(data: dict[str, Any]) -> None:
    st.subheader("AI换脸/深伪风险甄别依据")
    render_result_header(data)
    explanation = build_deepfake_explanation(data)
    if data["status"] == "skipped":
        st.info(f"本次未运行AI换脸/深伪风险甄别：{_skip_reason(data)}")
        return
    if data["status"] != "success":
        st.error(explanation["summary"])
        with st.expander("查看错误技术详情"):
            st.json(explanation["technical"])
        return

    evidence = _deepfake_evidence(data)
    getattr(st, evidence["state"], st.warning)(explanation["headline"])
    technical = explanation["technical"]
    score_column, threshold_column, margin_column, device_column = st.columns(4)
    score_column.metric("风险分数", _optional_score(technical.get("fusion_score")))
    threshold_column.metric("判定阈值", _optional_score(technical.get("decision_threshold")))
    margin_value = technical.get("threshold_margin")
    margin_column.metric("与阈值差值", f"{float(margin_value):+.4f}" if margin_value is not None else "—")
    device_column.metric("运行设备", str(technical.get("actual_device") or "—"))
    _render_basis(explanation)
    _render_warnings(explanation)
    _render_limitations(explanation)
    query_path, keyframe_paths = _resolve_deepfake_images(data, explanation)
    if query_path is not None or keyframe_paths:
        visual_columns = st.columns(2)
        with visual_columns[0]:
            if query_path is not None:
                render_image(query_path, "查询图", "deepfake_query")
        with visual_columns[1]:
            if keyframe_paths:
                render_image(
                    keyframe_paths[0],
                    "原生模型实际使用的有效关键帧",
                    "deepfake_keyframe_1",
                )
    with st.expander("展开查看AIGC技术指标"):
        formula = technical.get("formula") if isinstance(technical.get("formula"), Mapping) else {}
        st.write(f"**检测方法：** `{technical.get('method_name') or '未记录'}`")
        st.write(f"**空间域模型：** `{technical.get('spatial_model_name') or '未记录'}`")
        st.write(f"**频域模型：** `{technical.get('frequency_model_name') or '未记录'}`")
        st.write(f"**模型名称：** `{technical.get('model_name') or '未记录'}`")
        st.write("**模型版本与预处理：**")
        st.json(technical.get("model_version") or {})
        st.write(f"**空间域模型输出分数：** `{_format_six(technical.get('spatial_score'))}`")
        st.write(f"**频域模型输出分数：** `{_format_six(technical.get('frequency_score'))}`")
        st.write(f"**融合公式代入：** `{formula.get('substitution') or '未记录足够指标'}`")
        st.write(f"**输入文件SHA-256：** `{technical.get('input_sha256') or '未记录'}`")
        st.write(f"**融合配置SHA-256：** `{technical.get('fusion_config_sha256') or '未记录'}`")
        st.write("**模型文件SHA-256：**")
        st.json(technical.get("model_hashes") or {})
        st.write(f"**有效人脸或有效帧数量：** `{technical.get('valid_face_or_frame_count') if technical.get('valid_face_or_frame_count') is not None else '未记录'}`")
        st.json(technical)


def _resolve_deepfake_images(
    data: Mapping[str, Any], explanation: Mapping[str, Any]
) -> tuple[Path | None, list[Path]]:
    """Resolve only query/task-contained native keyframes for ordinary display."""

    query_value = data.get("query_image")
    if not isinstance(query_value, str) or not query_value:
        return None, []
    query_path = Path(query_value).expanduser().resolve(strict=False)
    if not query_path.is_file():
        return None, []
    task_root = query_path.parent.resolve(strict=False)
    visualization_root = (task_root / "visualizations").resolve(strict=False)
    technical = explanation.get("technical")
    technical = technical if isinstance(technical, Mapping) else {}
    native_run_value = technical.get("native_run_directory")
    if not isinstance(native_run_value, str) or not native_run_value:
        return query_path, []
    native_run = Path(native_run_value).expanduser().resolve(strict=False)
    try:
        native_run.relative_to(visualization_root)
    except ValueError:
        return query_path, []
    raw_keyframes = technical.get("keyframe_paths")
    if not isinstance(raw_keyframes, list):
        return query_path, []
    keyframes: list[Path] = []
    for raw_value in raw_keyframes:
        if not isinstance(raw_value, str) or not raw_value:
            continue
        supplied = Path(raw_value).expanduser()
        candidate = (
            supplied if supplied.is_absolute() else native_run / supplied
        ).resolve(strict=False)
        try:
            candidate.relative_to(native_run)
        except ValueError:
            continue
        if candidate.is_file():
            keyframes.append(candidate)
    return query_path, keyframes


def render_identity_results(data: dict[str, Any], path_policy: ResourcePathPolicy) -> None:
    st.subheader("人脸身份搜索依据")
    render_result_header(data)
    explanation = build_identity_explanation(data)
    if data["status"] == "skipped":
        st.info(f"本次任务未运行人脸身份搜索：{_skip_reason(data)}")
        return
    if data["status"] != "success":
        st.error(explanation["headline"])
        st.caption(explanation["summary"])
        with st.expander("查看错误技术详情"):
            st.json(explanation["technical"])
        return
    if not data["candidates"]:
        st.warning(explanation["headline"])
        st.caption(explanation["summary"])
        _render_limitations(explanation)
        return

    st.success(explanation["headline"])
    st.info("这是当前身份库中的候选排序，不是最终身份确认。余弦相似度不表示身份概率。")
    items = identity_view_items(data, path_policy)
    technical = explanation["technical"]
    candidates = technical.get("candidates") if isinstance(technical.get("candidates"), list) else []
    top = items[0]
    top_metrics = candidates[0] if candidates else {}
    image_column, detail_column = st.columns([1, 2])
    with image_column:
        render_image(top["resolved_best_reference_image"], "第一候选最佳参考照片", "identity_top1")
    with detail_column:
        first, second = st.columns(2)
        first.metric("人员级平均余弦相似度", _optional_score(top_metrics.get("mean_similarity")))
        second.metric("最佳单张参考图相似度", _optional_score(top_metrics.get("best_reference_score")))
        st.write(f"**人员：** `{top.get('identity_name') or top.get('identity_id')}`")
        st.write(f"**有效参考图数量：** `{top_metrics.get('reference_count') if top_metrics.get('reference_count') is not None else '未记录'}`")
        st.write(f"**最佳参考照片：** `{display_name(top_metrics.get('best_reference_image') or top.get('reference_image'))}`")
    top2_score = candidates[1].get("mean_similarity") if len(candidates) > 1 else None
    cards = st.columns(5)
    cards[0].metric("查询图人脸数", technical.get("query_face_count") if technical.get("query_face_count") is not None else "—")
    cards[1].metric("身份库人数", technical.get("library_identity_count") if technical.get("library_identity_count") is not None else "—")
    cards[2].metric("有效参考图总数", technical.get("valid_embedding_count") if technical.get("valid_embedding_count") is not None else "—")
    cards[3].metric("第二名相似度", _optional_score(top2_score))
    cards[4].metric("Top1与Top2差值", _optional_score(technical.get("top1_top2_margin")))
    _render_basis(explanation)
    _render_warnings(explanation)
    _render_limitations(explanation)
    with st.expander("展开查看全部身份候选和技术指标"):
        for item, candidate_metrics in zip(items, candidates, strict=False):
            with st.container(border=True):
                columns = st.columns([1, 2])
                with columns[0]:
                    render_image(item["resolved_best_reference_image"], f"第{item['rank']}名参考照片", f"identity_{item['rank']}")
                with columns[1]:
                    st.markdown(f"### 第{item['rank']}名：{item.get('identity_name') or item.get('identity_id')}")
                    st.metric("人员级平均余弦相似度", _optional_score(candidate_metrics.get("mean_similarity")))
                    st.write(f"有效参考图：{candidate_metrics.get('reference_count') if candidate_metrics.get('reference_count') is not None else '未记录'} 张")
                    st.write(f"最佳参考图：`{display_name(candidate_metrics.get('best_reference_image'))}`")
        st.json(technical)


def _resolve_query_image(data: Mapping[str, Any], policy: ResourcePathPolicy) -> Path | None:
    value = data.get("query_image")
    if not isinstance(value, str) or not value:
        return None
    try:
        return policy.resolve(value, "query", label="query_image")
    except PathContainmentError:
        return None


def _resolve_masked_image(explanation: Mapping[str, Any], policy: ResourcePathPolicy) -> Path | None:
    technical = explanation.get("technical")
    technical = technical if isinstance(technical, Mapping) else {}
    value = technical.get("masked_query_path")
    if not isinstance(value, str) or not value:
        return None
    try:
        return policy.resolve(value, "visualization", label="masked_query_path")
    except PathContainmentError:
        return None


def render_source_results(data: dict[str, Any], path_policy: ResourcePathPolicy) -> None:
    st.subheader("原始底图搜索依据")
    render_result_header(data)
    explanation = build_source_explanation(data)
    if data["status"] == "skipped":
        st.info(f"本次任务未运行原始底图搜索：{_skip_reason(data)}")
        return
    if data["status"] != "success":
        st.error(explanation["headline"])
        st.caption(explanation["summary"])
        with st.expander("查看错误技术详情"):
            st.json(explanation["technical"])
        return
    if not data["candidates"]:
        st.warning(explanation["headline"])
        st.caption(explanation["summary"])
        _render_limitations(explanation)
        return

    st.success(explanation["headline"])
    st.info("结果表示当前底图库内最相似候选，不是来源概率，也不等于已经确认原始底图。")
    technical = explanation["technical"]
    candidates = technical.get("candidates") if isinstance(technical.get("candidates"), list) else []
    top_metrics = candidates[0] if candidates else {}
    items = source_view_items(data, path_policy)
    top = items[0]
    score_columns = st.columns(4)
    score_columns[0].metric("全局相似度", _optional_score(top_metrics.get("global_score")))
    score_columns[1].metric("局部相似度", _optional_score(top_metrics.get("local_score")))
    score_columns[2].metric("最终相似度", _optional_score(top_metrics.get("final_score")))
    score_columns[3].metric("Top1与Top2差值", _optional_score(technical.get("top1_top2_margin")))
    match_columns = st.columns(5)
    match_columns[0].metric("有效匹配点", top_metrics.get("good_match_count") if top_metrics.get("good_match_count") is not None else "—")
    match_columns[1].metric("RANSAC内点", top_metrics.get("ransac_inlier_count") if top_metrics.get("ransac_inlier_count") is not None else "—")
    match_columns[2].metric("RANSAC内点率", _optional_score(top_metrics.get("ransac_inlier_ratio")))
    match_columns[3].metric("实际读取图库", technical.get("library_decoded") if technical.get("library_decoded") is not None else "—")
    match_columns[4].metric("粗排候选数", technical.get("shortlist_size") if technical.get("shortlist_size") is not None else "—")
    mask_state = technical.get("face_mask_applied")
    st.caption(f"查询图人脸屏蔽：{'已应用' if mask_state is True else '未应用' if mask_state is False else '未记录'}")
    query_path = _resolve_query_image(data, path_policy)
    masked_path = _resolve_masked_image(explanation, path_policy)
    visual_columns = st.columns(2)
    with visual_columns[0]:
        if query_path is not None:
            render_image(query_path, "查询图", "source_query")
        else:
            st.warning("查询图路径未通过任务目录安全校验。")
    with visual_columns[1]:
        if masked_path is not None:
            render_image(masked_path, "屏蔽人脸后的查询图", "source_masked_query")
        elif technical.get("query_face_detected") is False or technical.get("face_mask_applied") is False:
            st.info("查询图未检测到可屏蔽的人脸区域，本次按完整图像提取底图特征。")
        else:
            st.warning("结果未提供可安全展示的查询图人脸屏蔽图。")
    result_columns = st.columns(2)
    with result_columns[0]:
        render_image(top["resolved_file_path"], f"Top-1候选图：{display_name(top.get('file_path'))}", "source_top1")
    with result_columns[1]:
        render_image(top["resolved_visualization_path"], "SIFT/RANSAC匹配连线图", "source_top1_visualization")
    _render_basis(explanation)
    _render_warnings(explanation)
    _render_limitations(explanation)
    with st.expander("展开查看全部底图候选和技术指标"):
        for item, candidate_metrics in zip(items, candidates, strict=False):
            st.markdown(
                f"**第{item['rank']}名 `{display_name(item.get('file_path'))}`**　"
                f"全局={_format_six(candidate_metrics.get('global_score'))}　局部={_format_six(candidate_metrics.get('local_score'))}　"
                f"最终={_format_six(candidate_metrics.get('final_score'))}　"
                f"匹配点={candidate_metrics.get('good_match_count') if candidate_metrics.get('good_match_count') is not None else '未记录'}　"
                f"内点={candidate_metrics.get('ransac_inlier_count') if candidate_metrics.get('ransac_inlier_count') is not None else '未记录'}"
            )
        st.json(technical)


__all__ = [
    "build_evidence_summary",
    "format_score",
    "identity_view_items",
    "render_deepfake_result",
    "render_evidence_summary",
    "render_identity_results",
    "render_source_results",
    "source_view_items",
]
