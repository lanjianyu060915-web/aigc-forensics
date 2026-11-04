"""通用 AIGC 分支和文件标识两块页面；只读任务结果，不产生新的分数。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import streamlit as st

from components.theme import key_values, verdict_card


BAND_TONES = {"high": "bad", "medium": "warn", "low": "muted"}
VERDICT_TONES = {
    "labelled_ai": "bad",
    "labelled_possible": "warn",
    "generator_hint": "warn",
    "no_label": "muted",
}
CROP_LABELS = ("中心", "左上", "右上", "左下", "右下")
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


def _render_crop_table(general: Mapping[str, Any]) -> None:
    frames = _list(general.get("frames"))
    crops = _list(_mapping(frames[0]).get("crop_logits")) if frames else []
    if not crops:
        return
    st.table(
        [
            {"裁剪位置": CROP_LABELS[index] if index < len(CROP_LABELS) else str(index), "分数": f"{value:.3f}"}
            for index, value in enumerate(crops)
            if isinstance(value, (int, float))
        ]
    )
    st.caption("图片按原尺寸取 5 个 224 像素裁剪，分数取均值。")


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
                verdict_card(
                    _text(branch.get("label")),
                    "不适用",
                    "画面里没有可用的人脸，这一路不参与总体结论。",
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
    columns[2].metric("CLIP 分支", _logit_text(branches.get("clip_logit")))
    columns[3].metric("CommunityForensics", _logit_text(branches.get("community_logit")))
    if logit is not None and thresholds:
        _render_logit_scale(logit, thresholds)
    if general.get("band_reason"):
        st.caption(f"分档依据：{general['band_reason']}")
    st.caption(
        f"融合公式：0.8 × [0.5 × CLIP 原头 {_logit_text(branches.get('clip_original_logit'))} + "
        f"0.5 × 重训头 {_logit_text(branches.get('clip_refit_logit'))}] + "
        f"0.2 × CommunityForensics {_logit_text(branches.get('community_logit'))}"
    )
    if general.get("input_kind") == "video":
        _render_frame_chart(general)
    else:
        _render_crop_table(general)
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
    face = _mapping(_mapping(deepfake.get("general_aigc")).get("face_branch"))
    if not face:
        return
    st.markdown("#### 人脸分支")
    if face.get("status") == "no_face":
        st.caption("没有检出可用的人脸（人脸框短边不足 40 像素也算没有），这一路不参与总体结论。")
        return
    if face.get("status") != "success":
        st.warning(f"这一路没有完成：{_text(face.get('error_message'))}")
        return
    logit = _number(face.get("logit"))
    thresholds = _mapping(face.get("band_thresholds"))
    crops = _list(face.get("crops"))
    columns = st.columns(3)
    columns[0].metric("档位", _text(face.get("band_label"), "未标定"))
    columns[1].metric("裁剪图均值分数", _logit_text(logit))
    columns[2].metric("人脸裁剪图", f"{len(crops)} 张")
    if logit is not None and thresholds:
        _render_logit_scale(logit, thresholds, "人脸裁剪图分数（logit，越高越倾向生成）")
    if face.get("band_reason"):
        st.caption(f"分档依据：{face['band_reason']}")
    if render_crop is not None and crops:
        for column, item in zip(st.columns(max(len(crops), 3)), crops, strict=False):
            crop = _mapping(item)
            with column:
                render_crop(f"face_crops/{_text(crop.get('file'))}", f"分数 {_logit_text(crop.get('logit'))}")
    st.caption(
        "用 CommunityForensics 打人脸裁剪图，分数只和真实人脸的分布比较。"
        "换脸检测跨数据集不稳，高分是复核线索，低分不能排除换脸。"
    )


def render_provenance_page(provenance: Mapping[str, Any]) -> None:
    """文件标识页：只陈列文件里实际存在的内容，不下真假结论。"""
    st.subheader("文件标识")
    st.info("这一页只读文件本身，不跑模型。有标识是线索，没有标识不说明任何事。")
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
