"""Present live startup metadata; no model/service initialization on render."""
from __future__ import annotations

from collections.abc import Mapping
import math

SERVICE_LABELS = {"deepfake": "鉴伪服务", "identity": "人脸身份服务", "general_aigc": "通用鉴伪服务"}
STAGE_LABELS = {"not_started": "尚未启动", "pending": "等待启动", "creating_process": "创建进程",
                "process_created": "进程已创建", "importing": "导入运行环境", "checking_model": "校验模型",
                "loading_model": "加载模型", "checking_library": "检查参考库及缓存", "rebuilding_index": "重建参考索引",
                "warming_up": "预热", "ready": "已就绪", "failed": "失败", "error": "失败",
                "stopped": "已停止", "disabled": "未启用", "restarting": "重新启动", "unavailable": "不可用"}


def _duration(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return "耗时未记录"
    return f"{value:.1f} 秒"


def _resource_text(resource):
    if not isinstance(resource, Mapping):
        return "资源占用未记录"
    values = []
    for key, label in (("memory", "内存"), ("gpu_memory", "显存")):
        item = resource.get(key) or {}
        peak = item.get("sampled_peak_bytes") if isinstance(item, Mapping) else None
        if type(peak) is int and peak >= 0:
            suffix = "（部分采样缺失）" if item.get("status") == "partial" else ""
            values.append(f"{label}采样峰值 {peak / (1024 ** 2):.1f} MiB{suffix}")
        else:
            values.append(f"{label}未能测量")
    return "；".join(values)


def runtime_status_rows(snapshot: Mapping, *, source_cache: Mapping | None = None) -> list[dict]:
    rows = []
    for name, label in SERVICE_LABELS.items():
        raw = snapshot.get(name)
        raw = raw if isinstance(raw, Mapping) else {}
        stage = str(raw.get("stage") or raw.get("state") or "not_started")
        status = str(raw.get("stage_status") or "pending")
        stage_label = STAGE_LABELS.get(stage, "阶段未记录")
        if stage == "warming_up" and status == "skipped":
            stage_label = "未单独执行预热"
        rows.append({"service": name, "label": label, "kind": "model_worker", "pid": raw.get("pid"),
                     "stage": stage, "stage_label": stage_label, "stage_status": status,
                     "message": str(raw.get("stage_message") or raw.get("error") or "等待服务状态更新。"),
                     "elapsed": _duration(raw.get("startup_elapsed_seconds")),
                     "error": raw.get("error"), "warmup_status": raw.get("warmup_status", "not_reported"),
                     "resources": _resource_text(raw.get("startup_resources"))})
    if source_cache is not None:
        # Source retrieval has no persistent model worker. Do not infer a PID,
        # model readiness or index rebuild from a generic cache-check result.
        rows.append({"service": "source", "label": "素材参考缓存", "kind": "reference_cache", "pid": None,
                     "stage": str(source_cache.get("state") or source_cache.get("cache_status") or "not_recorded"),
                     "stage_label": "素材缓存检查", "stage_status": str(source_cache.get("status") or "unknown"),
                     "message": str(source_cache.get("message") or "仅表示素材缓存检查状态，不是常驻模型服务。"),
                     "elapsed": _duration(source_cache.get("elapsed_seconds")), "error": source_cache.get("error"),
                     "warmup_status": "not_applicable", "resources": "不作为常驻模型资源统计"})
    return rows


def startup_needs_refresh(snapshot: Mapping) -> bool:
    if snapshot.get("closed") is True:
        return False
    return snapshot.get("health") == "starting" or any(
        isinstance(snapshot.get(name), Mapping) and snapshot[name].get("state") in {"starting", "pending"}
        for name in SERVICE_LABELS
    )


def render_runtime_status(snapshot: Mapping, *, source_cache: Mapping | None = None) -> None:
    """Render within the caller's existing automatic-refresh fragment."""
    import streamlit as st
    for row in runtime_status_rows(snapshot, source_cache=source_cache):
        st.markdown(f"**{row['label']}：{row['stage_label']}**")
        st.caption(f"{row['message']} 启动/检查耗时：{row['elapsed']}。")
        if row["error"]:
            st.error(str(row["error"]))
        if row["kind"] == "model_worker":
            st.caption(row["resources"])
