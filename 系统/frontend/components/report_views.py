"""Safe Streamlit views and downloads for fixed task report artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import streamlit as st


PRODUCT_ROOT = Path(__file__).resolve().parents[2]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))

from reporting.report_builder import report_file_names  # noqa: E402
from reporting.review_records import (  # noqa: E402
    REVIEW_CONCLUSIONS, REVIEW_NOTICE, MANUAL_JUDGMENTS, ReviewRecordError,
    append_review_record, load_review_records, review_report_mismatch, review_context,
)


MAX_REPORT_BYTES = 20 * 1024 * 1024


class ReportViewError(ValueError):
    """A task report is missing, malformed, oversized, or outside its task."""


@dataclass(frozen=True)
class ReportBundle:
    task_dir: Path
    report: dict[str, Any]
    json_path: Path
    txt_path: Path
    html_path: Path
    json_bytes: bytes
    txt_bytes: bytes
    html_bytes: bytes


def _safe_fixed_file(task_root: Path, file_name: str) -> Path:
    if Path(file_name).name != file_name:
        raise ReportViewError("报告文件名必须是任务目录内的固定文件名。")
    root = task_root.resolve(strict=True)
    candidate = (root / file_name).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ReportViewError("报告路径越过任务目录，已拒绝。") from exc
    if not candidate.is_file():
        raise ReportViewError(f"报告文件不存在：{file_name}")
    try:
        size = candidate.stat().st_size
    except OSError as exc:
        raise ReportViewError(f"无法读取报告文件状态：{file_name}") from exc
    if size <= 0 or size > MAX_REPORT_BYTES:
        raise ReportViewError(f"报告文件大小不合法：{file_name}")
    return candidate


def load_report_bundle(task_dir: str | Path, task_id: str | None = None) -> ReportBundle:
    """Load only fixed report files contained by the supplied task directory."""

    root = Path(task_dir).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise ReportViewError("任务结果目录不存在或不是目录。")
    json_path = _safe_fixed_file(root, "final_report.json")
    try:
        json_bytes = json_path.read_bytes()
        report = json.loads(json_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportViewError("final_report.json 不是有效UTF-8 JSON。") from exc
    if not isinstance(report, dict):
        raise ReportViewError("final_report.json 顶层必须是对象。")
    if str(report.get("schema_version") or "") != "3.0":
        raise ReportViewError("final_report.json schema_version 必须为3.0。")
    report_task = report.get("task")
    report_task = report_task if isinstance(report_task, dict) else {}
    embedded_task_id = str(report_task.get("task_id") or "").strip()
    if not embedded_task_id:
        raise ReportViewError("final_report.json 缺少任务编号。")
    resolved_task_id = str(task_id or embedded_task_id or root.name)
    if task_id is not None and embedded_task_id and str(task_id) != embedded_task_id:
        raise ReportViewError("页面任务编号与报告任务编号不一致。")
    names = report_file_names(resolved_task_id)
    txt_path = _safe_fixed_file(root, names["txt"])
    html_path = _safe_fixed_file(root, names["html"])
    try:
        txt_bytes = txt_path.read_bytes()
        html_bytes = html_path.read_bytes()
        txt_bytes.decode("utf-8")
        html_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReportViewError("TXT或HTML报告不是有效UTF-8文件。") from exc
    return ReportBundle(
        task_dir=root,
        report=report,
        json_path=json_path,
        txt_path=txt_path,
        html_path=html_path,
        json_bytes=json_bytes,
        txt_bytes=txt_bytes,
        html_bytes=html_bytes,
    )


def prepare_pdf_download(bundle: ReportBundle, review_records: list[dict] | None = None) -> tuple[str, bytes]:
    """旧任务和复核版在内存生成，不重写任务的任何原始报告。"""
    task = bundle.report.get("task") or {}
    name = report_file_names(task.get("task_id"))["pdf"]
    analysis = bundle.report.get("analysis") or {}
    if isinstance(analysis, dict) and isinstance(analysis.get("technical_pdf_file"), str):
        name = analysis["technical_pdf_file"]
    if not review_records:
        try:
            path = _safe_fixed_file(bundle.task_dir, name)
        except ReportViewError:
            path = None
        if path is not None:
            raw = path.read_bytes()
            if not raw.startswith(b"%PDF-"):
                raise ReportViewError("已保存的 PDF 格式异常。")
            return name, raw
    else:
        name = name.removesuffix(".pdf") + "_复核版.pdf"
    from reporting.pdf_report import render_pdf_report

    return name, render_pdf_report(bundle.report, bundle.task_dir, review_records=review_records)


def render_report_view(bundle: ReportBundle, *, key_prefix: str = "task_report") -> None:
    """Render an ordinary-user report preview without exposing raw provenance."""

    report = bundle.report
    task = report.get("task") if isinstance(report.get("task"), dict) else {}
    input_data = report.get("input") if isinstance(report.get("input"), dict) else {}
    st.success("检测与检索结果已生成")
    columns = st.columns(3)
    columns[0].metric("任务编号", str(task.get("task_id") or "未记录"))
    columns[1].metric("任务状态", str(task.get("status") or "未记录"))
    columns[2].metric("查询文件", str(input_data.get("file_name") or "未记录"))

    state_key = f"{key_prefix}_expanded"
    if st.button("查看完整报告", key=f"{key_prefix}_view", width="stretch"):
        st.session_state[state_key] = not bool(st.session_state.get(state_key, False))
    if st.session_state.get(state_key, False):
        st.text(bundle.txt_bytes.decode("utf-8"))

    left, right = st.columns(2)
    left.download_button(
        "导出TXT报告",
        data=bundle.txt_bytes,
        file_name=bundle.txt_path.name,
        mime="text/plain; charset=utf-8",
        key=f"{key_prefix}_txt",
        width="stretch",
    )
    right.download_button(
        "导出HTML报告",
        data=bundle.html_bytes,
        file_name=bundle.html_path.name,
        mime="text/html; charset=utf-8",
        key=f"{key_prefix}_html",
        width="stretch",
    )
    st.markdown("#### PDF 研判报告和人工复核")
    records = []
    records_ok = True
    try:
        records = load_review_records(bundle.task_dir)
    except (OSError, ReviewRecordError) as error:
        st.warning(str(error))
        records_ok = False
    st.caption(REVIEW_NOTICE)
    try:
        pair, identity = review_context(bundle.task_dir, report)
    except (OSError, ValueError) as error:
        pair, identity = {}, {}
        st.warning("复核依据读取失败，请核对任务文件。")
        records_ok = False
    candidates = {row.get("rank"): row for row in pair.get("candidates", []) if row.get("rank") in (1, 2, 3)}
    faces = {row.get("face_index"): row for row in identity.get("faces", []) if type(row.get("face_index")) is int}
    if st.session_state.pop(f"{key_prefix}_review_saved", False):
        st.session_state[f"{key_prefix}_reviewer"] = ""
        st.session_state[f"{key_prefix}_opinion"] = ""
        st.success("复核意见已追加保存，原检测结果和原 PDF 保持不变。")
    with st.form(f"{key_prefix}_review_form", clear_on_submit=False):
        reviewer = st.text_input("复核人（必填）", key=f"{key_prefix}_reviewer")
        conclusion = st.selectbox("对检测档位的意见", REVIEW_CONCLUSIONS)
        manual_judgment = st.selectbox("人工判断", MANUAL_JUDGMENTS)
        candidate_rank = st.selectbox("依据候选", [None, *sorted(candidates)], format_func=lambda rank: "未指定" if rank is None else f"第{rank}候选：{candidates[rank].get('candidate_name') or '未记录名称'}") if candidates else None
        target_face_index = st.selectbox("人工选定的目标脸", [None, *sorted(faces)], format_func=lambda index: "未指定" if index is None else f"人脸{index}") if len(faces) > 1 else None
        opinion = st.text_area("复核意见（必填）", key=f"{key_prefix}_opinion")
        submitted = st.form_submit_button("追加保存复核意见", disabled=not records_ok)
    if submitted:
        try:
            append_review_record(bundle.task_dir, reviewer, conclusion, opinion,
                                 manual_judgment=manual_judgment, basis_candidate_rank=candidate_rank,
                                 target_face_index=target_face_index)
            records = load_review_records(bundle.task_dir)
            st.session_state[f"{key_prefix}_review_saved"] = True
            st.rerun()
        except (OSError, ReviewRecordError) as error:
            st.warning(str(error))
    if records:
        st.caption(f"已保存 {len(records)} 条复核记录。导出时生成复核版，原报告保持不变。")
        with st.expander("查看复核记录"):
            for index, record in enumerate(records, start=1):
                if review_report_mismatch(bundle.task_dir, record):
                    st.warning("该条复核所依据的报告与当前报告不一致")
                st.text(f"{index}. {record['reviewer']} · {record['reviewed_at']} · {record['conclusion']}\n{record['opinion']}")
                if record.get("manual_judgment"):
                    st.text("人工判断，非系统结论：" + record["manual_judgment"])
    if not records_ok:
        st.caption("复核记录不可读，已禁用复核版和追加；原版报告仍可导出。")
    from components.analysis_views import render_analysis_downloads
    render_analysis_downloads(bundle, reviews=records, reviews_ok=records_ok, key_prefix=key_prefix)
    try:
        pdf_name = (report.get("analysis") or {}).get("technical_pdf_file") or report_file_names(task.get("task_id"))["pdf"]
        if records or not (bundle.task_dir / pdf_name).is_file():
            from reporting.pdf_report import check_pdf_ready

            check_pdf_ready()
            signature = hashlib.sha256(bundle.json_bytes + json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            cache_key = f"{key_prefix}_pdf_memory"
            cached = st.session_state.get(cache_key)
            if not cached or cached[0] != signature:
                st.session_state.pop(cache_key, None)
                if st.button("生成技术明细报告（PDF）", key=f"{key_prefix}_prepare_pdf", width="stretch"):
                    with st.spinner("正在生成技术明细报告"):
                        name, raw = prepare_pdf_download(bundle, records)
                    cached = (signature, name, raw)
                    st.session_state[cache_key] = cached
                else:
                    cached = None
            if cached:
                st.download_button("导出技术明细报告（PDF）", data=cached[2], file_name=cached[1], mime="application/pdf", key=f"{key_prefix}_pdf", width="stretch")
        else:
            name, raw = prepare_pdf_download(bundle)
            st.download_button("导出技术明细报告（PDF）", data=raw, file_name=name, mime="application/pdf", key=f"{key_prefix}_pdf", width="stretch")
    except Exception as error:
        st.warning(f"PDF 报告暂不可用（{type(error).__name__}），仍可导出 TXT、HTML 和 JSON。")
    with st.expander("报告技术详情"):
        st.caption("JSON报告保留结构化来源信息；普通TXT/HTML仅展示文件名。")
        st.download_button(
            "导出JSON报告",
            data=bundle.json_bytes,
            file_name=bundle.json_path.name,
            mime="application/json; charset=utf-8",
            key=f"{key_prefix}_json",
        )
        st.write(f"任务结果目录：`{bundle.task_dir}`")


def render_task_reports(
    task_dir: str | Path,
    task_id: str | None = None,
    *,
    key_prefix: str = "task_report",
) -> ReportBundle | None:
    """Load and render reports, turning controlled file errors into a warning."""

    try:
        bundle = load_report_bundle(task_dir, task_id)
    except ReportViewError as exc:
        st.warning(f"检测报告暂不可用：{exc}")
        return None
    render_report_view(bundle, key_prefix=key_prefix)
    return bundle


__all__ = [
    "MAX_REPORT_BYTES",
    "ReportBundle",
    "ReportViewError",
    "load_report_bundle",
    "render_report_view",
    "render_task_reports",
]
