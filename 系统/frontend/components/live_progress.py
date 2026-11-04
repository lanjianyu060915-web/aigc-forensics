from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import streamlit as st


MODULE_ORDER = ("deepfake", "identity", "source")
MODULE_LABELS = {
    "overall": "总体",
    "video": "视频抽帧与选帧",
    "deepfake": "AI换脸/深伪风险甄别",
    "identity": "人脸身份搜索",
    "source": "原始底图搜索",
}
STATUS_LABELS = {
    "pending": "等待",
    "running": "运行中",
    "success": "完成",
    "warning": "警告",
    "error": "失败",
    "skipped": "已跳过",
}
STATUS_ICONS = {
    "pending": "⚪",
    "running": "🔵",
    "success": "🟢",
    "warning": "🟠",
    "error": "🔴",
    "skipped": "⚫",
}


def _read_json(path: Path) -> Any:
    try:
        if not path.is_file() or path.stat().st_size > 10 * 1024 * 1024:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def load_progress_events(task_dir: str | Path) -> list[dict[str, Any]]:
    """Read complete JSONL records only; tolerate an interrupted final append."""

    root = Path(task_dir).expanduser().resolve(strict=False)
    path = root / "logs" / "progress_events.jsonl"
    events: list[dict[str, Any]] = []
    try:
        if path.is_file() and path.stat().st_size <= 20 * 1024 * 1024:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    value = line.strip()
                    if not value:
                        continue
                    try:
                        event = json.loads(value)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        events.append(event)
    except (OSError, UnicodeError):
        pass
    if events:
        return events

    # Compatibility with v0.2 task archives.
    legacy = _read_json(root / "logs" / "progress.json")
    if isinstance(legacy, list):
        for index, item in enumerate(legacy, 1):
            if not isinstance(item, Mapping):
                continue
            events.append(
                {
                    "event_id": f"legacy-{index}",
                    "timestamp": item.get("timestamp"),
                    "task_id": root.name,
                    "module": item.get("module") or "overall",
                    "stage": item.get("stage") or "legacy",
                    "status": "running",
                    "message": item.get("message") or "旧版进度事件",
                    "progress_current": None,
                    "progress_total": None,
                    "progress_percent": None,
                    "elapsed_seconds": None,
                    "details": {"compatibility_mode": "v0.2"},
                }
            )
    return events


def load_progress_latest(task_dir: str | Path) -> dict[str, Any] | None:
    root = Path(task_dir).expanduser().resolve(strict=False)
    value = _read_json(root / "logs" / "progress_latest.json")
    return dict(value) if isinstance(value, Mapping) else None


def _percent(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return min(100.0, max(0.0, number))


def _module_card_event(event: Mapping[str, Any], module: str) -> dict[str, Any]:
    """Keep stage success distinct from the module's terminal success state."""

    display = dict(event)
    status = str(display.get("status") or "")
    stage = str(display.get("stage") or "")
    if status == "success" and stage != f"{module}_finished":
        details = display.get("details")
        normalized_details = dict(details) if isinstance(details, Mapping) else {}
        normalized_details["stage_event_status"] = "success"
        normalized_details["module_card_status"] = "running_until_module_finished"
        display["details"] = normalized_details
        display["status"] = "running"
    return display


@dataclass(frozen=True)
class ProgressSnapshot:
    overall_status: str
    overall_percent: float | None
    current_event: dict[str, Any] | None
    module_events: dict[str, dict[str, Any]]
    events: list[dict[str, Any]]


def build_progress_snapshot(
    events: Iterable[Mapping[str, Any]],
    *,
    selected_modules: Iterable[str] = MODULE_ORDER,
    job_status: str = "running",
) -> ProgressSnapshot:
    normalized = [dict(item) for item in events if isinstance(item, Mapping)]
    selected = {str(item) for item in selected_modules}
    module_events: dict[str, dict[str, Any]] = {
        module: {
            "module": module,
            "stage": "pending",
            "status": "pending" if module in selected else "skipped",
            "message": "等待执行" if module in selected else "本次未选择",
            "progress_percent": 0.0 if module in selected else 100.0,
        }
        for module in MODULE_ORDER
    }
    overall_event: dict[str, Any] | None = None
    for event in normalized:
        module = str(event.get("module") or "overall")
        if module in module_events:
            module_events[module] = _module_card_event(event, module)
        if module == "overall":
            overall_event = event

    current = normalized[-1] if normalized else None
    finished = sum(
        str(module_events[module].get("status")) in {"success", "warning", "error", "skipped"}
        for module in MODULE_ORDER
        if module in selected
    )
    total = len(selected & set(MODULE_ORDER))
    measured_overall = (100.0 * finished / total) if total else 100.0
    overall_percent = (
        _percent(overall_event.get("progress_percent"))
        if overall_event is not None
        else measured_overall
    )
    if overall_percent is None:
        overall_percent = measured_overall
    overall_status = str(
        (overall_event or {}).get("status")
        or ("success" if job_status == "success" else job_status)
        or "running"
    )
    return ProgressSnapshot(
        overall_status=overall_status,
        overall_percent=overall_percent,
        current_event=current,
        module_events=module_events,
        events=normalized,
    )


def render_live_progress(
    *,
    task_dir: str | Path | None,
    selected_modules: Iterable[str],
    job_status: str,
    queued_message: str | None = None,
) -> ProgressSnapshot | None:
    st.markdown("#### 分析进度")
    if task_dir is None:
        st.info(queued_message or "任务已进入执行队列，正在创建独立任务目录……")
        return None

    events = load_progress_events(task_dir)
    snapshot = build_progress_snapshot(
        events,
        selected_modules=selected_modules,
        job_status=job_status,
    )
    current = snapshot.current_event or {}
    current_module = str(current.get("module") or "overall")
    current_label = MODULE_LABELS.get(current_module, current_module)
    status_message = str(current.get("message") or current.get("stage") or "等待真实进度事件")
    elapsed = current.get("elapsed_seconds")
    elapsed_text = f" · 已运行 {float(elapsed):.0f} 秒" if isinstance(elapsed, (int, float)) else ""
    st.progress(
        (snapshot.overall_percent or 0.0) / 100.0,
        text=f"总进度 {snapshot.overall_percent or 0.0:.0f}%{elapsed_text}",
    )
    st.write(f"**{current_label}：** {status_message}")
    current_count = current.get("progress_current")
    total_count = current.get("progress_total")
    if isinstance(current_count, (int, float)) and isinstance(total_count, (int, float)):
        st.caption(f"实际处理数量：{current_count:g} / {total_count:g}")

    columns = st.columns(3)
    for column, module in zip(columns, MODULE_ORDER, strict=True):
        event = snapshot.module_events[module]
        status = str(event.get("status") or "pending")
        with column:
            with st.container(border=True):
                st.markdown(f"**{STATUS_ICONS.get(status, '⚪')} {MODULE_LABELS[module]}**")
                st.write(STATUS_LABELS.get(status, status))
                st.caption(str(event.get("message") or "等待执行"))
                percent = _percent(event.get("progress_percent"))
                if percent is not None:
                    st.progress(percent / 100.0, text=f"{percent:.1f}%")
                current_count = event.get("progress_current")
                total_count = event.get("progress_total")
                if isinstance(current_count, (int, float)) and isinstance(total_count, (int, float)):
                    st.caption(f"实际处理数量：{current_count:g} / {total_count:g}")

    with st.expander("技术运行日志", expanded=job_status in {"error", "partial_success"}):
        if not snapshot.events:
            st.caption("尚未产生进度事件。")
        for event in snapshot.events[-200:]:
            timestamp = str(event.get("timestamp") or "")
            module = str(event.get("module") or "overall")
            stage = str(event.get("stage") or "")
            message = str(event.get("message") or "")
            counts = ""
            if event.get("progress_current") is not None and event.get("progress_total") is not None:
                counts = f" [{event['progress_current']}/{event['progress_total']}]"
            st.code(f"{timestamp} | {module}/{stage} | {message}{counts}", language=None)
    return snapshot
