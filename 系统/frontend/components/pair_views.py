"""配对核验的人工对比展示。"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import streamlit as st

from components.path_security import PathContainmentError, ResourcePathPolicy


def _artifact(value: Any, policy: ResourcePathPolicy) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        path = policy.resolve(value, "visualization", label="配对证据图")
        return path if path.is_file() else None
    except (PathContainmentError, OSError):
        return None


def _show_image(path: Path | None, caption: str) -> None:
    if path is not None:
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption(f"{caption}：未生成")


def _slider_compare(left: Path, right: Path, key: str) -> None:
    with Image.open(left) as first, Image.open(right) as second:
        first.load()
        second.load()
        a, b = first.convert("RGB"), second.convert("RGB")
    if a.size != b.size:
        st.caption("两张图的坐标范围不同，暂不显示滑动对比。")
        return
    value = st.slider("拖动分界线比较：左侧候选图，右侧对齐后的查询图", 0, 100, 50, key=key)
    boundary = round(a.width * value / 100)
    composed = b.copy()
    composed.paste(a.crop((0, 0, boundary, a.height)), (0, 0))
    if 0 < boundary < a.width:
        ImageDraw.Draw(composed).line((boundary, 0, boundary, a.height), fill="#00b6d8", width=2)
    st.image(composed, width="stretch")


def render_pair_comparison(pair: Any, policy: ResourcePathPolicy, key_prefix: str) -> None:
    if not isinstance(pair, Mapping):
        return
    st.subheader("候选底图配对对比")
    st.info(str(pair.get("notice") or "配对结果供人工对比。"))
    # 与报告复用脱敏状态文案，覆盖尚未形成候选时的初始化故障。
    # 缺少状态字段的旧结果仍按原有布局展示。
    from pair_integration import pair_report_lines
    for line in pair_report_lines(pair):
        if line.startswith("配对比对未完成："):
            st.warning(line)
            st.caption("其他模块的结果不受影响。")
        elif line.startswith("配对比对未运行："):
            st.info(line)
    if pair.get("scope") == "frame":
        timestamp = pair.get("timestamp_seconds")
        stamp = f"{timestamp:g} 秒" if isinstance(timestamp, (int, float)) else "未记录"
        st.warning(f"核验时间点：{stamp}。这里只展示首要帧，不是整段视频的结论。")
    st.caption("只核验检索排名前三的候选，真实底图可能排在后面。差异也可能由压缩、裁剪或颜色变化引起。")
    candidates = pair.get("candidates") or []
    if not candidates:
        st.caption("本次没有可展示的候选配对。")
        return
    tabs = st.tabs([f"候选 {row.get('rank', index)}" for index, row in enumerate(candidates, 1)])
    for index, (tab, row) in enumerate(zip(tabs, candidates), 1):
        with tab:
            st.markdown(str(row.get("candidate_name") or Path(str(row.get("candidate_path") or "未记录候选")).name))
            if row.get("status") != "success":
                st.warning("该候选的配对没有完成，其他模块的结果不受影响。")
                continue
            comparison = row.get("comparison") or {}
            artifacts = row.get("artifacts") or {}
            left, right = st.columns(2)
            with left:
                _show_image(_artifact(artifacts.get("query"), policy), "查询图")
            with right:
                _show_image(_artifact(artifacts.get("reference"), policy), "候选底图")
            aligned = _artifact(artifacts.get("aligned"), policy)
            reference = _artifact(artifacts.get("reference"), policy)
            if aligned and reference:
                _slider_compare(reference, aligned, f"{key_prefix}_pair_{index}_slider")
            else:
                st.caption("未生成可靠的对齐图，暂不能通过滑动对比判断同源。")
            difference = _artifact(artifacts.get("difference"), policy)
            if difference:
                _show_image(difference, "差异热图：仅提示像素变化位置，不能单凭颜色判断修改性质")
            alignment = comparison.get("alignment") or {}
            st.caption(f"对齐内点数：{alignment.get('inliers', '未记录')}；"
                       f"配对计算耗时：{comparison.get('comparison_seconds', '未记录')} 秒")
            faces = comparison.get("faces") or []
            for face_index, face in enumerate(faces):
                st.markdown(f"**人脸对比 {face_index + 1}**")
                cols = st.columns(2)
                with cols[0]:
                    _show_image(_artifact(artifacts.get(f"face_{face_index}_query"), policy), "查询图中的脸")
                with cols[1]:
                    _show_image(_artifact(artifacts.get(f"face_{face_index}_reference"), policy), "候选图中对应位置的脸")
                score = face.get("cosine_similarity")
                text = f"{score:.4f}" if isinstance(score, (int, float)) else "无法比对"
                st.caption(f"人脸余弦相似度：{text}。数值不是身份概率，需要结合图像质量人工核对。")
            if not faces:
                st.caption("本候选没有可展示的对应人脸；不据此判断是否发生换脸。")
