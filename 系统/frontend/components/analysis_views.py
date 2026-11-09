"""研判摘要与独立报告导出；页面刷新不会触发模型请求。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from collections.abc import Mapping

import streamlit as st

from reporting.evidence_figures import local_file
from reporting.llm_client import canonical_bytes, DEFAULT_LLM_CONFIG
from reporting.review_records import review_report_mismatch
from .llm_views import llm_context, upload_notice


def load_analysis(task_dir: Path, report: Mapping | None = None) -> dict:
    path = local_file(task_dir, "analysis_report.json")
    if path is not None:
        if path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("研判记录超过大小限制")
        value = json.loads(path.read_text("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("研判记录不是对象")
        from reporting.analysis_pipeline import verify_analysis_artifacts
        if verify_analysis_artifacts(task_dir):
            raise ValueError("研判证据清单不完整或文件摘要不一致")
        if report is not None:
            source = {k: v for k, v in report.items() if k not in ("analysis", "evidence_package")}
            if hashlib.sha256(canonical_bytes(source)).hexdigest() != value.get("source_report_sha256"):
                raise ValueError("研判记录与原任务结果不一致")
        return value
    value = (report or {}).get("analysis")
    # 有档位的研判必须以独立文件和完整性清单为准；不能退回未经核验的嵌入副本。
    if isinstance(value, Mapping) and value.get("final_decision"):
        raise ValueError("研判原件或完整性清单缺失")
    return dict(value) if isinstance(value, Mapping) else {}


def render_analysis_summary(task_dir: Path, report: Mapping | None) -> None:
    try:
        analysis = load_analysis(task_dir, report)
    except (OSError, ValueError):
        st.warning("研判摘要与当前任务不一致或无法读取，请检查证据文件。")
        return
    final = analysis.get("final_decision") or {}
    if not final:
        if analysis.get("status") == "error":
            st.warning(str(analysis.get("reason") or "研判报告未完成"))
        return
    labels = {**DEFAULT_LLM_CONFIG["band_labels"], **analysis.get("settings", {}).get("band_labels", {})}
    columns = st.columns([2, 1, 1])
    columns[0].metric("最终档位", labels.get(final.get("final_band"), "无法完成判断"))
    columns[1].metric("AIGC 可疑度评分", str(final["score"]) if final.get("score") is not None else "未评分")
    columns[2].metric("研判方式", {"rule": "规则", "rule_fallback": "规则回退", "llm": "大模型"}.get(final.get("source"), "未记录"))
    st.caption("AIGC 可疑度评分范围为 0–100，是本次研判的分档表达，不是伪造概率；各检测分支分数不相加。")
    st.write(analysis.get("summary") or analysis.get("content", {}).get("summary") or final.get("reason", ""))
    if final.get("source") == "rule_fallback":
        st.caption("回退原因：" + str(final.get("reason") or "接口或校验未通过"))
    package = analysis.get("evidence_package") or (report or {}).get("evidence_package") or {}
    face = next((v.get("data", {}) for v in package.get("items", []) if v.get("item_id") == "E04"), {})
    highest = face.get("highest_face") or {}
    if highest:
        label = "人脸 " + str(highest.get("face_id", highest.get("face_index", "未编号")))
        if highest.get("timestamp_seconds") is not None:
            label += f"，时间 {highest['timestamp_seconds']} 秒"
        st.caption("人脸分支最高分来自" + label + "；是否参与定档及小脸限制见鉴伪证据页。")


def render_analysis_downloads(bundle, *, reviews: list[dict], reviews_ok: bool, key_prefix: str) -> None:
    root = bundle.task_dir
    try:
        analysis = load_analysis(root, bundle.report)
        pdf_path = local_file(root, analysis.get("pdf_file")) if analysis else None
        if pdf_path:
            if pdf_path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("报告超过导出大小限制")
            raw = pdf_path.read_bytes()
            if not raw.startswith(b"%PDF-"):
                raise ValueError("PDF格式无效")
            st.download_button("导出研判分析报告（PDF）", data=raw, file_name=pdf_path.name,
                               mime="application/pdf", key=f"{key_prefix}_analysis_pdf", type="primary", width="stretch")
            mismatched = any(review_report_mismatch(root, record) for record in reviews)
            if reviews and reviews_ok and not mismatched:
                signature = hashlib.sha256(raw + canonical_bytes(reviews)).hexdigest()
                cache_key = f"{key_prefix}_analysis_review"
                cached = st.session_state.get(cache_key)
                if not cached or cached[0] != signature:
                    if st.button("生成研判分析报告复核版", key=f"{key_prefix}_analysis_review_prepare"):
                        from reporting.analysis_pipeline import generate_review_analysis
                        with st.spinner("正在追加人工复核章节"):
                            reviewed = generate_review_analysis(root, reviews=reviews)
                        cached = (signature, reviewed)
                        st.session_state[cache_key] = cached
                    else:
                        cached = None
                if cached:
                    st.download_button("导出研判分析报告复核版（PDF）", data=cached[1], file_name="研判分析报告（复核版）.pdf",
                                       mime="application/pdf", key=f"{key_prefix}_analysis_review_pdf")
            elif reviews and mismatched:
                st.warning("复核记录所依据的原报告已变化，暂不能导出研判复核版。")
        elif not analysis or analysis.get("status") == "error":
            config_path = Path(os.environ.get("FSTS_CONFIG_PATH", str(Path(__file__).resolve().parents[2] / "config.json")))
            config, profile, settings = llm_context(config_path)
            st.caption(upload_notice(settings))
            if st.button("生成研判分析报告", key=f"{key_prefix}_analysis_generate", type="primary"):
                from reporting.analysis_pipeline import generate_analysis_report
                with st.spinner("正在整理已有证据并生成研判分析报告"):
                    generate_analysis_report(root, bundle.report, config=config, settings=settings, profile_path=profile)
                st.rerun()
        else:
            st.warning("研判报告文件缺失，请核对任务证据包。")
    except Exception as error:
        st.warning(f"研判分析报告暂不可用（{type(error).__name__}）。原技术报告仍可导出。")
