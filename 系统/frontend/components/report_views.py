"""Safe Streamlit views and downloads for fixed task report artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import streamlit as st


PRODUCT_ROOT = Path(__file__).resolve().parents[2]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))

from reporting.report_builder import report_file_names  # noqa: E402


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
