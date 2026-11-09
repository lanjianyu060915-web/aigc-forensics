"""通用 AIGC 分支和文件标识两块页面；只读任务结果，不产生新的分数。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from typing import Any

import streamlit as st

from components.theme import key_values, verdict_card
from decision.aigc_evidence import DISCLAIMER, LOW_NOTE, evidence_table


BAND_TONES = {"high": "bad", "medium": "warn", "low": "muted"}
VERDICT_TONES = {
    "labelled_ai": "bad",
    "labelled_possible": "warn",
    "generator_hint": "warn",
    "no_label": "muted",
}
BRANCH_COLORS = {"CLIP 分支": "#1F4E79", "CommunityForensics 分支": "#C55A3D"}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _logit_text(value: Any) -> str:
    number = _number(value)
    return f"{number:.3f}" if number is not None else "—"


def _scored_face_count(face: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    count = _mapping(audit.get("coverage")).get("scored_count")
    if type(count) is int and count >= 0:
        return f"{count} 张"
    rows = _list(face.get("per_face")) or _list(face.get("crops"))
    scores = [_number(_mapping(row).get("score", _mapping(row).get("logit"))) for row in rows]
    measured = sum(value is not None and math.isfinite(value) for value in scores)
    return f"{measured} 张" if measured else "未记录"


def _text(value: Any, fallback: str = "—") -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def overall_finding(deepfake: Mapping[str, Any]) -> dict[str, str] | None:
    """研判要点里“是否生成或伪造”这张卡；没有总体结论时返回 None，走原来的逻辑。"""
    overall = _mapping(deepfake.get("overall"))
    if not overall:
        return None
    return {
        "title": "是否生成或伪造",
        "value": _text(overall.get("band_label"), "未标定"),
        "note": f"{_text(overall.get('reason'))} 按档位读结果，分数没有概率含义。",
        "tone": BAND_TONES.get(str(overall.get("band")), "muted"),
    }


def provenance_finding(provenance: Mapping[str, Any]) -> dict[str, str]:
    if provenance.get("status") != "success":
        return {
            "title": "文件自带标识",
            "value": "未完成",
            "note": _text(provenance.get("error_message"), "文件标识解析没有完成。"),
            "tone": "muted",
        }
    verdict = str(provenance.get("verdict") or "no_label")
    notes = {
        "labelled_ai": "文件元数据声明这是 AI 生成内容，标识可能被伪造，要结合其他证据。",
        "labelled_possible": "文件元数据标注可能或疑似 AI 生成，要结合其他证据。",
        "generator_hint": "元数据里有生成工具的痕迹，但没有正式标识。",
        "no_label": "没有发现标识。重新编码、截图、转发都会丢失标识，这不代表不是生成内容。",
    }
    return {
        "title": "文件自带标识",
        "value": _text(provenance.get("verdict_label"), "未发现标识"),
        "note": notes.get(verdict, ""),
        "tone": VERDICT_TONES.get(verdict, "muted"),
    }


def _render_logit_scale(
    logit: float,
    thresholds: Mapping[str, Any],
    title: str = "融合分数（logit，越高越倾向生成）",
) -> None:
    low_max = _number(thresholds.get("low_max"))
    medium_max = _number(thresholds.get("medium_max"))
    if low_max is None or medium_max is None:
        return
    lower = min(logit - 1.0, low_max - 4.0)
    upper = max(logit + 1.0, medium_max + 3.0)
    bands = [
        {"start": lower, "end": low_max, "grade": "证据不足", "color": "#7d8794"},
        {"start": low_max, "end": medium_max, "grade": "需人工复核", "color": "#d9a441"},
        {"start": medium_max, "end": upper, "grade": "高风险提示", "color": "#c55a3d"},
    ]
    st.vega_lite_chart(
        {
            "data": {"values": bands},
            "layer": [
                {
                    "mark": {"type": "bar", "height": 30, "opacity": 0.55},
                    "encoding": {
                        "x": {
                            "field": "start",
                            "type": "quantitative",
                            "scale": {"domain": [lower, upper]},
                            "title": title,
                        },
                        "x2": {"field": "end"},
                        "color": {
                            "field": "grade",
                            "type": "nominal",
                            "scale": {
                                "domain": [item["grade"] for item in bands],
                                "range": [item["color"] for item in bands],
                            },
                            "legend": {"title": "档位", "orient": "top", "direction": "horizontal"},
                        },
                        "tooltip": [
                            {"field": "grade", "title": "档位"},
                            {"field": "start", "type": "quantitative", "format": ".3f", "title": "下界"},
                            {"field": "end", "type": "quantitative", "format": ".3f", "title": "上界"},
                        ],
                    },
                },
                {
                    "data": {"values": [{"score": logit}]},
                    "mark": {"type": "rule", "color": "#111111", "strokeWidth": 3},
                    "encoding": {
                        "x": {"field": "score", "type": "quantitative"},
                        "tooltip": [
                            {"field": "score", "type": "quantitative", "format": ".3f", "title": "本次分数"}
                        ],
                    },
                },
            ],
            "height": 70,
        },
        width="stretch",
    )


def _render_frame_chart(general: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for item in _list(general.get("frames")):
        frame = _mapping(item)
        rows.append(
            {
                "time": _number(frame.get("timestamp_seconds")),
                "logit": _number(frame.get("clip_logit")),
                "branch": "CLIP 分支",
            }
        )
    for item in _list(general.get("community_frames")):
        frame = _mapping(item)
        rows.append(
            {
                "time": _number(frame.get("timestamp_seconds")),
                "logit": _number(frame.get("logit")),
                "branch": "CommunityForensics 分支",
            }
        )
    if not rows:
        return
    st.vega_lite_chart(
        {
            "data": {"values": rows},
            "mark": {"type": "line", "point": {"size": 60, "filled": True}, "strokeWidth": 2},
            "encoding": {
                "x": {"field": "time", "type": "quantitative", "title": "时间（秒）"},
                "y": {"field": "logit", "type": "quantitative", "title": "逐帧分数（logit）"},
                "color": {
                    "field": "branch",
                    "type": "nominal",
                    "scale": {
                        "domain": list(BRANCH_COLORS),
                        "range": list(BRANCH_COLORS.values()),
                    },
                    "legend": {"title": "分支", "orient": "top"},
                },
                "tooltip": [
                    {"field": "branch", "title": "分支"},
                    {"field": "time", "type": "quantitative", "format": ".2f", "title": "时间（秒）"},
                    {"field": "logit", "type": "quantitative", "format": ".3f", "title": "分数"},
                ],
            },
            "height": 220,
        },
        width="stretch",
    )
    st.caption("逐帧分数只说明模型在这些时刻的反应，不是伪造时段定位。")


def _formula_text(general: Mapping[str, Any]) -> str:
    branches = _mapping(general.get("branches"))
    values = (
        f"重训头 {_logit_text(branches.get('clip_refit_logit'))}，"
        f"CommunityForensics {_logit_text(branches.get('community_logit'))}"
    )
    if general.get("input_kind") != "image":
        values = f"CLIP 原头 {_logit_text(branches.get('clip_original_logit'))}，{values}"
    return f"融合公式：{_text(branches.get('formula'))}。本次分数：{values}。"


def render_overall_section(deepfake: Mapping[str, Any]) -> None:
    """总体结论和各路分档并排；没有通用分支的旧任务不显示。"""
    overall = _mapping(deepfake.get("overall"))
    if not overall:
        return
    st.markdown("#### 总体结论")
    verdict_card(
        "由各路分档推导的总体档位",
        _text(overall.get("band_label"), "未标定"),
        _text(overall.get("reason")),
        BAND_TONES.get(str(overall.get("band")), "muted"),
    )
    branches = _list(overall.get("branches"))
    columns = st.columns(max(len(branches), 1))
    for column, item in zip(columns, branches, strict=False):
        branch = _mapping(item)
        with column:
            if branch.get("status") == "success":
                verdict_card(
                    _text(branch.get("label")),
                    _text(branch.get("band_label"), "未标定"),
                    f"分数 {_logit_text(branch.get('score'))}",
                    BAND_TONES.get(str(branch.get("band")), "muted"),
                )
            elif branch.get("status") == "no_face":
                general = _mapping(deepfake.get("general_aigc"))
                face = _mapping(general.get("face_branch"))
                audit = _mapping(deepfake.get("face_coverage_evidence") or face.get("face_audit") or general.get("face_preprocessing"))
                no_face_note = ("本次检测未检出人脸，这一路不参与总体结论。"
                                if _mapping(audit.get("coverage")).get("detected_total") == 0 else
                                "没有可参与当前定档路径的人脸；检出、裁剪失败和未覆盖情况见逐脸记录。")
                verdict_card(
                    _text(branch.get("label")),
                    "不适用",
                    no_face_note,
                    "muted",
                )
            else:
                verdict_card(
                    _text(branch.get("label")),
                    "未完成",
                    _text(branch.get("error_message")),
                    "muted",
                )
    st.caption("各路分数性质不同，不相加。通用 AIGC 分支定主结论，人脸分支最多抬一级，整图分支只作提示。")
    for hint in _list(overall.get("hints")):
        message = _mapping(hint).get("message")
        if message:
            st.warning(str(message))
    evidence = _mapping(overall.get("band_evidence"))
    rows = evidence_table(evidence)
    if rows:
        st.markdown("##### 档位证据强度")
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(DISCLAIMER)
        if evidence.get("low_note"):
            st.info(LOW_NOTE)


def render_general_aigc_section(deepfake: Mapping[str, Any]) -> None:
    """通用 AIGC 分支的证据明细。"""
    general = _mapping(deepfake.get("general_aigc"))
    if not general:
        return
    st.markdown("#### 通用 AIGC 检测分支")
    if general.get("status") != "success":
        st.warning(f"这一路没有完成：{_text(general.get('error_message'))}")
        return
    branches = _mapping(general.get("branches"))
    media = _mapping(general.get("media"))
    runtime = _mapping(general.get("runtime"))
    thresholds = _mapping(general.get("band_thresholds"))
    logit = _number(general.get("logit"))
    columns = st.columns(4)
    columns[0].metric("档位", _text(general.get("band_label"), "未标定"))
    columns[1].metric("融合分数", _logit_text(logit))
    if general.get("input_kind") == "image":
        columns[2].metric("重训头", _logit_text(branches.get("clip_refit_logit")))
    else:
        columns[2].metric("CLIP 分支", _logit_text(branches.get("clip_logit")))
    columns[3].metric("CommunityForensics", _logit_text(branches.get("community_logit")))
    if logit is not None and thresholds:
        _render_logit_scale(logit, thresholds)
    if general.get("band_reason"):
        st.caption(f"分档依据：{general['band_reason']}")
    st.caption(_formula_text(general))
    if general.get("input_kind") == "video":
        _render_frame_chart(general)
    calibration = _mapping(thresholds.get("calibration_set"))
    with st.expander("运行详情和局限"):
        key_values(
            [
                ("模型", _text(_mapping(general.get("model")).get("model_id"))),
                ("校准版本", _text(_mapping(general.get("model")).get("calibration_version"))),
                ("校准样本", _text(calibration.get("description"))),
                ("运算设备", _text(runtime.get("device_name"))),
                ("分辨率", f"{_text(media.get('width'))} × {_text(media.get('height'))}"),
                ("解码方式", {"strict": "严格", "tolerant": "容错", "seek": "按时间点取帧", "image": "图片"}.get(str(media.get("decode_mode")), "—")),
                ("推理耗时", f"{_logit_text(runtime.get('inference_seconds'))} 秒"),
            ]
        )
        for text in _list(general.get("limitations")):
            st.caption(f"· {text}")
    for text in _list(general.get("warnings")):
        st.warning(_text(text))


def render_face_branch_section(
    deepfake: Mapping[str, Any],
    render_crop: Callable[[str, str], Any] | None = None,
) -> None:
    """人脸分支的证据明细：逐张裁剪的分数和所用的边界。render_crop 负责显示任务目录里的裁剪图。"""
    general = _mapping(deepfake.get("general_aigc"))
    face = _mapping(general.get("face_branch"))
    audit = _mapping(deepfake.get("face_coverage_evidence") or face.get("face_audit") or general.get("face_preprocessing"))
    if audit:
        from reporting.coverage_summary_v4 import face_binding_coverage_rows, face_coverage_note, module_face_coverage_rows
        st.markdown("#### 分模块人脸覆盖")
        st.caption(face_coverage_note(audit))
        st.dataframe(module_face_coverage_rows(audit), width="stretch", hide_index=True)
        st.caption("各模块按自身实际记录统计，检出数可能不同；身份尝试数包含失败，配对处理数只统计能明确对应本表查询脸的完成测量。")
        st.dataframe(face_binding_coverage_rows(audit), width="stretch", hide_index=True)
        st.caption("对应数以鉴伪查询脸观察为范围，视频逐帧计数。已处理与失败是明确对应记录的子集；"
                   "配对部分完成已计入已处理，表示至少一个候选完成测量，其余候选仍有未完成记录。"
                   "旧任务缺少记录时显示未记录，未知和歧义不计为已处理。")
        checks = [_mapping(value) for value in _list(audit.get("face_checks"))]
        if checks:
            states = {"processed": "已处理", "partial": "部分完成", "failed": "失败", "ambiguous": "歧义",
                      "unknown": "未知", "unmatched": "未对应", "matched": "明确对应", "skipped": "跳过"}
            st.dataframe([{"画面": row.get("source_id"), "人脸记录": row.get("face_uid"),
                           "裁剪状态": row.get("crop_status"), "评分状态": row.get("score_status"),
                           "分数": row.get("score"), "参与定档": row.get("used_for_band"),
                           "身份对应": states.get(row.get("identity_binding_status"), "未记录"),
                           "身份检查": states.get(row.get("identity_processing_status"), "未知"),
                           "配对检查": states.get(row.get("pair_processing_status"), "未知"),
                           "配对逐候选": "；".join(f"候选 {entry.get('candidate_rank') or index}: {states.get(entry.get('processing_status'), '未知')}"
                                                 for index, entry in enumerate((_mapping(v) for v in _list(row.get("pair_checks"))), 1)) or "未记录",
                           "未覆盖原因": row.get("exclusion_reason") or row.get("score_binding_reason")}
                          for row in checks], width="stretch", hide_index=True)
    if not face:
        return
    st.markdown("#### 人脸分支")
    per_face = _list(face.get("per_face"))
    if face.get("status") == "no_face" and not per_face:
        if audit and _mapping(audit.get("coverage")).get("detected_total") == 0:
            st.caption("本次检测未检出人脸，这一路不参与总体结论。")
        else:
            st.caption("本次没有可参与旧最大脸定档路径的人脸；检出数量、裁剪失败及小脸情况以检查记录为准。")
        return
    if face.get("status") not in {"success", "no_face", "no_eligible_face"}:
        st.warning(f"这一路没有完成：{_text(face.get('error_message'))}")
        return
    logit = _number(face.get("logit"))
    thresholds = _mapping(face.get("band_thresholds"))
    crops = _list(face.get("crops"))
    columns = st.columns(3)
    display_label = "无可参与定档人脸" if per_face and face.get("status") == "no_face" else _text(face.get("band_label"), "未标定")
    columns[0].metric("档位", display_label)
    columns[1].metric("参与定档的分数", _logit_text(logit))
    columns[2].metric("已评分人脸", _scored_face_count(face, audit))
    if logit is not None and thresholds:
        score_caption = "人脸模型分数（非材料真伪概率）" if face.get("backend") == "gend" else "人脸裁剪图分数（logit，越高越倾向生成）"
        _render_logit_scale(logit, thresholds, score_caption)
    if face.get("band_reason"):
        st.caption(f"分档依据：{face['band_reason']}")
    if per_face:
        strategy = "每帧最大脸" if face.get("band_aggregation") == "largest" else "每帧可用人脸中的最高分"
        st.caption(f"定档取值：{strategy}；视频对各帧所选分数取算术平均。小脸分数仅作不可靠线索。")
        highest = _mapping(face.get("highest_score_face"))
        if highest:
            st.caption(f"最高分来自画面 {_text(highest.get('source'))} 的第 {highest.get('face_id')} 张脸。")
        st.dataframe([
            {"画面": row.get("source"), "人脸序号": row.get("face_id"),
             "短边像素": row.get("short_edge_px"), "分数": row.get("score"),
             "参与定档": "是" if row.get("used_for_band") else "否",
             "可靠性": "人脸过小，结果不可靠" if row.get("too_small") else "未触发小脸限制"}
            for value in per_face if (row := _mapping(value))
        ], use_container_width=True, hide_index=True)
        if face.get("compatibility_note"):
            st.caption(_text(face["compatibility_note"]))
    if render_crop is not None:
        for figure in _list(face.get("annotated_frames")):
            figure = _mapping(figure)
            render_crop(f"face_crops/{_text(figure.get('file'))}", _text(figure.get("note")))
        display = per_face or crops
        for start in range(0, len(display), 3):
            for column, item in zip(st.columns(3), display[start:start + 3], strict=False):
                crop = _mapping(item)
                with column:
                    value = crop.get("score", crop.get("logit"))
                    render_crop(f"face_crops/{_text(crop.get('file'))}", f"人脸 {crop.get('face_id', start + 1)}：分数 {_logit_text(value)}")
    st.caption(
        ("用 GenD 打人脸裁剪图，模型分数不是鉴伪结论的概率。" if face.get("backend") == "gend"
         else "用 CommunityForensics 打人脸裁剪图，分数只和真实人脸的分布比较。") +
        "换脸检测跨数据集不稳，高分是复核线索，低分不能排除换脸。"
    )


def render_provenance_page(provenance: Mapping[str, Any], visible_label: Mapping[str, Any] | None = None) -> None:
    """文件标识页：只陈列文件里实际存在的内容，不下真假结论。"""
    st.subheader("文件标识")
    st.info("文件元数据和画面文字分别展示。有标识是线索，没有标识不说明任何事；画面文字由离线文字识别读取。")
    from modules.visible_label import visible_label_lines
    for index, line in enumerate(visible_label_lines(_mapping(visible_label))):
        if index == 0:
            st.markdown("#### " + line)
        else:
            st.write(line)
    if not provenance:
        st.warning("这个任务没有文件标识解析结果。")
        return
    finding = provenance_finding(provenance)
    verdict_card(finding["title"], finding["value"], finding["note"], finding["tone"])
    if provenance.get("status") != "success":
        return
    key_values(
        [
            ("文件", _text(provenance.get("input_file"))),
            ("容器格式", _text(provenance.get("container"))),
            ("大小", f"{_text(provenance.get('size_bytes'))} 字节"),
            ("SHA-256", _text(provenance.get("sha256"))),
        ]
    )
    labels = _list(provenance.get("aigc_labels"))
    st.markdown("#### 隐式标识")
    if labels:
        for item in labels:
            label = _mapping(item)
            with st.container(border=True):
                key_values(
                    [
                        ("依据标准", _text(label.get("standard"))),
                        ("所在位置", _text(label.get("location"))),
                        ("标注含义", f"{_text(label.get('label'))}：{_text(label.get('label_meaning'))}"),
                        ("生成服务方", _text(label.get("content_producer"), "未填写")),
                        ("生成编号", _text(label.get("produce_id"), "未填写")),
                        ("传播平台", _text(label.get("content_propagator"), "未填写")),
                        ("传播编号", _text(label.get("propagate_id"), "未填写")),
                        ("疑似信用代码", "、".join(str(code) for code in _list(label.get("credit_codes"))) or "无"),
                    ]
                )
    else:
        st.write("没有发现 GB 45438-2025 隐式标识。")
    c2pa = _mapping(provenance.get("c2pa"))
    st.markdown("#### 内容凭证（C2PA）")
    if c2pa.get("present"):
        st.write(f"存在内容凭证结构，位置：{'、'.join(str(item) for item in _list(c2pa.get('locations')))}。")
    else:
        st.write("没有发现内容凭证结构。")
    st.caption(_text(c2pa.get("note"), "只做结构检测，未验证签名。"))
    source_type = provenance.get("digital_source_type")
    if source_type:
        st.write(f"XMP 来源类型：`{source_type}`")
    hints = _list(provenance.get("generator_hints"))
    if hints:
        st.markdown("#### 生成工具痕迹")
        for item in hints:
            hint = _mapping(item)
            st.write(f"- **{_text(hint.get('source'))}**：{_text(hint.get('value'))[:300]}")
    with st.expander("全部元数据"):
        st.json(_mapping(provenance.get("metadata")))
    for text in _list(provenance.get("warnings")):
        st.warning(_text(text))
    for text in _list(provenance.get("limitations")):
        st.caption(f"· {text}")


__all__ = [
    "overall_finding",
    "provenance_finding",
    "render_general_aigc_section",
    "render_overall_section",
    "render_provenance_page",
]
