from __future__ import annotations

from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
import json
from html import escape
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
from typing import Any


FRONTEND_ROOT = Path(__file__).resolve().parent
PRODUCT_ROOT = FRONTEND_ROOT.parent
for import_root in (FRONTEND_ROOT, PRODUCT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import streamlit as st

from product_info import product_names

from orchestrator import (
    OrchestrationError,
    load_config,
    resolve_path,
    run_task,
)
from components.data_loader import (
    ResultDataError,
    TaskResults,
    load_detection_result,
    load_identity_result,
    load_source_result,
    load_task_results,
)
from components.input_preview import (
    ExampleQuery,
    InputPreview,
    describe_input,
    routing_edge_from_config,
)
from components.path_security import ResourcePathPolicy
from components.case_library_views import render_case_library_settings
from components.live_progress import render_live_progress
from components.report_views import render_task_reports
from components.llm_views import llm_context, upload_notice, render_llm_settings, render_llm_status
from components.evidence_dashboard import render_evidence_workspace
from components.status_adapter import (
    ModuleAvailability,
    device_label,
    module_availability,
)
from components.task_history import (
    build_evidence_bundle,
    display_cell,
    list_tasks,
    status_label,
    summaries_to_csv,
    summaries_to_table,
    summarize_task,
    verify_task_artifacts,
)
from components.theme import (
    inject_theme,
    key_values,
    pill,
    render_html,
    step_title,
)


PROJECT_ROOT = FRONTEND_ROOT
APP_CONFIG = Path(
    os.environ.get("FSTS_CONFIG_PATH", str(PRODUCT_ROOT / "config.json"))
).expanduser().resolve(strict=False)
SAMPLE_ROOT = FRONTEND_ROOT / "sample_data"
MOCK_RESULTS_ROOT = FRONTEND_ROOT / "mock_results"
IDENTITY_LIBRARY_ROOT = SAMPLE_ROOT / "identity_libraries"
SOURCE_LIBRARY_ROOT = SAMPLE_ROOT / "source_libraries"
EXAMPLES_ROOT = PRODUCT_ROOT / "examples"
MODULE_KEYS = ("deepfake", "identity", "source")
MODULE_LABELS = {
    "deepfake": "AIGC 鉴伪",
    "identity": "人脸身份比对",
    "source": "原始素材检索",
}
MODULE_SHORT_LABELS = {"deepfake": "鉴伪", "identity": "身份", "source": "素材"}
SERVICE_LABELS = {**MODULE_LABELS, "general_aigc": "通用 AIGC 模型"}
PAGE_LABELS = {
    "Live Analysis": "新建研判",
    "Batch Analysis": "批量研判",
    "Task Result": "历史任务",
    "System Status": "系统状态",
    "LLM Settings": "大模型设置",
    "Guide": "使用说明",
    "Mock Demo": "开发模拟数据",
}
RESULT_MODE_LABELS = {
    "Live Analysis": "本次研判",
    "Task Result": "归档任务",
    "Mock Demo": "模拟数据",
}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_VIDEO_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_BATCH_FILES = 100
EXAMPLE_LABELS = {
    "integrated_query_001.jpg": "换脸样例 1（高分辨率）",
    "integrated_query_002.jpg": "换脸样例 2",
    "integrated_query_003.jpg": "换脸样例 3",
    "aigc_fake_image.jpg": "小尺寸人脸裁剪 1",
    "aigc_real_image.jpg": "小尺寸人脸裁剪 2",
    "no_face_image.jpg": "无人脸图片",
    "labelled_demo_video.mp4": "带隐式标识的视频（标识为人工写入）",
}


@st.cache_resource(show_spinner=False)
def _live_job_runtime() -> dict[str, Any]:
    """One serial background executor plus a process-local job registry."""

    return {
        "executor": ThreadPoolExecutor(max_workers=1, thread_name_prefix="fsts-analysis"),
        "lock": threading.RLock(),
        "jobs": {},
    }


def _job_update(runtime: Mapping[str, Any], job_id: str, **values: Any) -> None:
    lock = runtime["lock"]
    with lock:
        job = runtime["jobs"].get(job_id)
        if isinstance(job, dict):
            job.update(values)


def _job_snapshot(job_id: str | None) -> dict[str, Any] | None:
    if not job_id:
        return None
    runtime = _live_job_runtime()
    with runtime["lock"]:
        job = runtime["jobs"].get(job_id)
        return dict(job) if isinstance(job, dict) else None


def _execute_live_job(
    runtime: Mapping[str, Any],
    job_id: str,
    *,
    inbox_path: Path,
    identity_root: Path | None,
    source_root: Path | None,
    modules: frozenset[str],
    services: Any | None,
    original_file_name: str | None = None,
) -> None:
    """Run forensic work without calling any Streamlit API from this thread."""

    _job_update(runtime, job_id, status="running")

    def on_progress(event: dict[str, Any]) -> None:
        values: dict[str, Any] = {"latest_event": dict(event)}
        details = event.get("details")
        if isinstance(details, Mapping) and details.get("task_dir"):
            values["task_dir"] = str(details["task_dir"])
            values["task_id"] = str(event.get("task_id") or "")
        _job_update(runtime, job_id, **values)

    try:
        result = run_task(
            inbox_path,
            identity_library=identity_root,
            source_library=source_root,
            selected_modules=modules,
            services=services,
            config_path=APP_CONFIG,
            progress_callback=on_progress,
            original_file_name=original_file_name,
        )
        _job_update(
            runtime,
            job_id,
            status=str(result.get("status") or "error"),
            task_dir=str(result.get("task_dir") or ""),
            task_id=str(result.get("task_id") or ""),
            result=result,
            error=None,
        )
    except Exception as exc:
        _job_update(
            runtime,
            job_id,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        try:
            inbox_path.unlink(missing_ok=True)
        except OSError:
            pass


def _execute_batch_job(
    runtime: Mapping[str, Any],
    job_id: str,
    *,
    entries: list[tuple[str, Path]],
    identity_root: Path | None,
    source_root: Path | None,
    modules: frozenset[str],
    services: Any | None,
) -> None:
    """逐个文件串行分析；单个文件失败只记录在该行，不中断其余文件。"""

    _job_update(runtime, job_id, status="running")
    for index, (name, inbox_path) in enumerate(entries):
        with runtime["lock"]:
            job = runtime["jobs"].get(job_id)
            if not isinstance(job, dict):
                return
            job["current_index"] = index
            job["items"][index]["status"] = "running"
        try:
            result = run_task(
                inbox_path,
                identity_library=identity_root,
                source_library=source_root,
                selected_modules=modules,
                services=services,
                config_path=APP_CONFIG,
                original_file_name=name,
            )
            task_dir = str(result.get("task_dir") or "")
            values = {
                "status": str(result.get("status") or "error"),
                "task_dir": task_dir,
                "summary": summarize_task(task_dir) if task_dir else None,
                "error": None,
            }
        except Exception as exc:
            values = {
                "status": "error",
                "task_dir": "",
                "summary": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            try:
                inbox_path.unlink(missing_ok=True)
            except OSError:
                pass
        with runtime["lock"]:
            job = runtime["jobs"].get(job_id)
            if isinstance(job, dict):
                job["items"][index].update(values)
    _job_update(runtime, job_id, status="finished", current_index=len(entries))


def _execute_warm_job(
    runtime: Mapping[str, Any],
    job_id: str,
    *,
    library: Path,
    cache_dir: Path,
) -> None:
    """建立素材特征缓存；放在同一串行执行器里，避免和分析任务抢资源。"""

    _job_update(runtime, job_id, status="running")
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(PRODUCT_ROOT / "tools" / "build_source_cache.py"),
                "--library",
                str(library),
                "--cache-dir",
                str(cache_dir),
            ],
            cwd=str(PRODUCT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip()[-400:] or f"退出码 {completed.returncode}")
        _job_update(
            runtime,
            job_id,
            status="success",
            result=json.loads(completed.stdout.strip().splitlines()[-1]),
        )
    except Exception as exc:
        _job_update(runtime, job_id, status="error", error=f"{type(exc).__name__}: {exc}")


def discover_libraries(library_root: Path) -> list[Path]:
    if not library_root.is_dir():
        return []
    return sorted(
        (path.resolve() for path in library_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )


def discover_examples() -> list[Path]:
    if not EXAMPLES_ROOT.is_dir():
        return []
    order = list(EXAMPLE_LABELS)
    return sorted(
        (path for path in EXAMPLES_ROOT.iterdir() if path.name in EXAMPLE_LABELS),
        key=lambda path: order.index(path.name),
    )


def _parse_optional_root(path_value: str, label: str) -> tuple[Path, ...]:
    value = path_value.strip()
    if not value:
        return ()
    path = Path(value).expanduser().resolve(strict=False)
    if not path.is_dir():
        raise ResultDataError(f"{label}不存在或不是目录：{path}")
    return (path,)


@st.cache_data(ttl=5, show_spinner=False)
def _cached_system_status(config_path_text: str) -> dict[str, Any]:
    """Inspect real resources while turning import/inspection failures into data."""

    try:
        from system_status import inspect_system_status

        snapshot = inspect_system_status(config_path=config_path_text)
        if not isinstance(snapshot, dict):
            raise TypeError("系统状态检查结果不是 JSON 对象")
        return {"snapshot": snapshot, "error": None}
    except Exception as exc:
        return {
            "snapshot": {
                "overall_status": "unavailable",
                "device": "unknown",
                "deepfake": {
                    "available": False,
                    "message": "无法完成 AIGC 资源检查。",
                },
                "identity": {
                    "available": False,
                    "message": "无法完成身份库检查。",
                },
                "source": {
                    "available": False,
                    "message": "无法完成素材库检查。",
                },
            },
            "error": f"{type(exc).__name__}: {exc}",
        }


@st.cache_resource(show_spinner=False)
def _cached_runtime_services(config_path_text: str) -> dict[str, Any]:
    """后台启动常驻模型服务，页面不等待；进度写在返回的字典里。"""

    bundle: dict[str, Any] = {
        "manager": None,
        "status": {"started": False, "ready": False, "health": "starting"},
        "error": None,
        "state": "starting",
    }

    def start() -> None:
        manager: Any = None
        try:
            from runtime_services import RuntimeServiceManager

            manager = RuntimeServiceManager(config_path_text)
            bundle["manager"] = manager
            startup_status = manager.start()
            if not isinstance(startup_status, dict):
                startup_status = manager.get_status()
            bundle.update(status=startup_status, state="ready")
        except Exception as exc:
            if manager is not None:
                try:
                    manager.close()
                except Exception:
                    pass
            bundle.update(
                manager=None,
                status={"started": False, "ready": False, "health": "error"},
                error=f"{type(exc).__name__}: {exc}",
                state="failed",
            )

    threading.Thread(target=start, name="fsts-service-start", daemon=True).start()
    return bundle


def _runtime_service_snapshot(bundle: Mapping[str, Any]) -> dict[str, Any]:
    manager = bundle.get("manager")
    if manager is not None:
        try:
            status = manager.get_status()
            if isinstance(status, dict):
                return status
        except Exception as exc:
            return {
                "started": False,
                "ready": False,
                "health": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    status = bundle.get("status")
    return dict(status) if isinstance(status, Mapping) else {}


def _services_starting(bundle: Mapping[str, Any]) -> bool:
    return bundle.get("state") == "starting"


def _store_results(
    *,
    mode: str,
    deepfake_result: dict[str, Any],
    identity_result: dict[str, Any],
    source_result: dict[str, Any],
    path_policy: ResourcePathPolicy,
    source_label: str,
    legacy_dual_module: bool = False,
    task_status: dict[str, Any] | None = None,
    unified_result: dict[str, Any] | None = None,
    final_report: dict[str, Any] | None = None,
    artifact_manifest: dict[str, Any] | None = None,
) -> None:
    st.session_state["loaded_results"] = {
        "mode": mode,
        "deepfake": deepfake_result,
        "identity": identity_result,
        "source": source_result,
        "policy": path_policy,
        "source_label": source_label,
        "legacy_dual_module": legacy_dual_module,
        "task_status": task_status,
        "unified_result": unified_result,
        "final_report": final_report,
        "artifact_manifest": artifact_manifest,
    }


def _load_mock(identity_library: Path, source_library: Path) -> None:
    policy = ResourcePathPolicy(
        identity_roots=(identity_library,),
        source_roots=(source_library,),
        sample_roots=(SAMPLE_ROOT,),
    )
    deepfake = load_detection_result(
        MOCK_RESULTS_ROOT / "detection_result.json", policy
    )
    identity = load_identity_result(
        MOCK_RESULTS_ROOT / "identity_result.json", policy
    )
    source = load_source_result(MOCK_RESULTS_ROOT / "source_result.json", policy)
    _store_results(
        mode="Mock Demo",
        deepfake_result=deepfake,
        identity_result=identity,
        source_result=source,
        path_policy=policy,
        source_label="内置、非算法生成的布局测试素材",
    )


def _load_task(
    task_dir_text: str,
    identity_root_text: str,
    source_root_text: str,
    resource_root_text: str,
    *,
    mode: str = "Task Result",
) -> None:
    task_dir = Path(task_dir_text.strip()).expanduser().resolve(strict=False)
    results: TaskResults = load_task_results(
        task_dir,
        identity_roots=_parse_optional_root(identity_root_text, "身份库根"),
        source_roots=_parse_optional_root(source_root_text, "素材库根"),
        sample_roots=_parse_optional_root(resource_root_text, "额外批准资源根"),
    )
    _store_results(
        mode=mode,
        deepfake_result=results.deepfake_result,
        identity_result=results.identity_result,
        source_result=results.source_result,
        path_policy=results.path_policy,
        source_label=str(results.task_dir),
        legacy_dual_module=results.legacy_dual_module,
        task_status=results.task_status,
        unified_result=results.unified_result,
        final_report=results.final_report,
        artifact_manifest=results.artifact_manifest,
    )


def _live_defaults() -> tuple[str, str, Path]:
    config, config_path = load_config(APP_CONFIG)
    identity_root = resolve_path(config["identity"]["library"], config_path)
    source_root = resolve_path(config["source"]["library"], config_path)
    inbox_root = resolve_path(config["runtime"]["inbox_root"], config_path)
    return str(identity_root), str(source_root), inbox_root


def _archived_task_defaults() -> tuple[str, str, str]:
    """Return formal config roots and the newest readable task, if present."""

    config, config_path = load_config(APP_CONFIG)
    tasks_root = resolve_path(config["runtime"]["tasks_root"], config_path)
    identity_root = resolve_path(config["identity"]["library"], config_path)
    source_root = resolve_path(config["source"]["library"], config_path)
    newest_task: Path | None = None
    try:
        candidates = [
            path
            for path in tasks_root.iterdir()
            if path.is_dir() and (path / "task_status.json").is_file()
        ]
        if candidates:
            newest_task = max(
                candidates,
                key=lambda path: (path / "task_status.json").stat().st_mtime_ns,
            )
    except OSError:
        newest_task = None
    return (
        str(newest_task or tasks_root),
        str(identity_root),
        str(source_root),
    )


def _validate_upload(uploaded_query: Any) -> str:
    if uploaded_query is None:
        raise ResultDataError("请先上传查询图片或视频。")
    suffix = Path(str(uploaded_query.name)).suffix.lower()
    size = getattr(uploaded_query, "size", 0)
    if suffix in ALLOWED_VIDEO_SUFFIXES:
        if isinstance(size, (int, float)) and size > MAX_VIDEO_UPLOAD_BYTES:
            raise ResultDataError("查询视频超过 200 MiB 上限。")
        return suffix
    if isinstance(size, (int, float)) and size > MAX_UPLOAD_BYTES:
        raise ResultDataError("查询图片超过 20 MiB 上限。")
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise ResultDataError(
            "查询文件扩展名只允许 jpg/jpeg/png/webp（图片）或 mp4/mov/avi/mkv（视频）。"
        )
    return suffix


def _humanize_progress(event: Any) -> str:
    if isinstance(event, Mapping):
        message = event.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
        module = str(event.get("module") or event.get("service") or "").lower()
        stage = str(event.get("stage") or event.get("state") or "").lower()
        if module == "deepfake":
            return "正在加载 AIGC 模型并检测图像……"
        if module == "identity":
            return "正在检测人脸、提取身份特征并搜索数据库……"
        if module == "source":
            return "正在提取素材特征并搜索数据库……"
        if stage:
            return f"任务进度：{stage}"
        return "正在执行分析任务……"
    message = str(event).strip()
    replacements = {
        "正在进行人脸身份比对": "正在检测人脸、提取身份特征并搜索身份数据库",
        "正在进行原始素材检索": "正在提取素材特征并搜索素材数据库",
        "正在进行换脸鉴伪": "正在加载 AIGC 模型并检测图像",
    }
    for original, friendly in replacements.items():
        if original in message:
            return message.replace(original, friendly)
    return message or "正在执行分析任务……"


def _start_live_analysis(
    uploaded_query: Any,
    identity_root_text: str,
    source_root_text: str,
    selected_modules: Collection[str],
    services: Any | None,
) -> str:
    modules = frozenset(selected_modules)
    if not modules or not modules <= set(MODULE_KEYS):
        raise ResultDataError(f"未知模块选择：{sorted(modules)}")
    suffix = _validate_upload(uploaded_query)

    identity_root = (
        _parse_optional_root(identity_root_text, "身份库")
        if "identity" in modules
        else ()
    )
    source_root = (
        _parse_optional_root(source_root_text, "素材库")
        if "source" in modules
        else ()
    )
    if "identity" in modules and not identity_root:
        raise ResultDataError("人脸身份比对必须指定身份库。")
    if "source" in modules and not source_root:
        raise ResultDataError("原始素材检索必须指定素材库。")

    _, _, inbox_root = _live_defaults()
    inbox_root.mkdir(parents=True, exist_ok=True)
    inbox_path = inbox_root / f"upload_{secrets.token_hex(8)}{suffix}"
    inbox_path.write_bytes(uploaded_query.getvalue())
    job_id = f"job_{secrets.token_hex(8)}"
    runtime = _live_job_runtime()
    job = {
        "job_id": job_id,
        "status": "queued",
        "task_id": None,
        "task_dir": None,
        "selected_modules": tuple(key for key in MODULE_KEYS if key in modules),
        "identity_root": str(identity_root[0]) if identity_root else "",
        "source_root": str(source_root[0]) if source_root else "",
        "query_name": str(uploaded_query.name),
        "latest_event": None,
        "result": None,
        "error": None,
    }
    with runtime["lock"]:
        runtime["jobs"][job_id] = job
    try:
        future = runtime["executor"].submit(
            _execute_live_job,
            runtime,
            job_id,
            inbox_path=inbox_path,
            identity_root=identity_root[0] if identity_root else None,
            source_root=source_root[0] if source_root else None,
            modules=modules,
            services=services,
            original_file_name=str(uploaded_query.name),
        )
    except Exception:
        try:
            inbox_path.unlink(missing_ok=True)
        except OSError:
            pass
        with runtime["lock"]:
            runtime["jobs"].pop(job_id, None)
        raise
    _job_update(runtime, job_id, future=future)
    return job_id


def _start_batch_analysis(
    uploads: list[Any],
    identity_root_text: str,
    source_root_text: str,
    selected_modules: Collection[str],
    services: Any | None,
) -> str:
    modules = frozenset(selected_modules)
    if not modules or not modules <= set(MODULE_KEYS):
        raise ResultDataError("请至少选择一个分析模块。")
    if not uploads:
        raise ResultDataError("请先选择要分析的图片。")
    if len(uploads) > MAX_BATCH_FILES:
        raise ResultDataError(f"一次最多分析 {MAX_BATCH_FILES} 张图片。")
    identity_root = (
        _parse_optional_root(identity_root_text, "身份库") if "identity" in modules else ()
    )
    source_root = (
        _parse_optional_root(source_root_text, "素材库") if "source" in modules else ()
    )
    if "identity" in modules and not identity_root:
        raise ResultDataError("人脸身份比对必须指定身份库。")
    if "source" in modules and not source_root:
        raise ResultDataError("原始素材检索必须指定素材库。")

    _, _, inbox_root = _live_defaults()
    inbox_root.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, Path]] = []
    items: list[dict[str, Any]] = []
    for upload in uploads:
        name = str(upload.name)
        try:
            suffix = _validate_upload(upload)
            if suffix in ALLOWED_VIDEO_SUFFIXES:
                raise ResultDataError("批量研判只支持图片。")
        except ResultDataError as exc:
            items.append({"name": name, "status": "error", "error": str(exc), "summary": None})
            continue
        path = inbox_root / f"upload_{secrets.token_hex(8)}{suffix}"
        path.write_bytes(upload.getvalue())
        entries.append((name, path))
        items.append({"name": name, "status": "queued", "error": None, "summary": None})
    # 校验失败的文件排到最后展示，分析顺序和上传顺序一致。
    runnable = [item for item in items if item["status"] == "queued"]
    rejected = [item for item in items if item["status"] != "queued"]
    job_id = f"batch_{secrets.token_hex(8)}"
    runtime = _live_job_runtime()
    with runtime["lock"]:
        runtime["jobs"][job_id] = {
            "job_id": job_id,
            "status": "queued",
            "items": runnable + rejected,
            "current_index": 0,
            "selected_modules": tuple(key for key in MODULE_KEYS if key in modules),
        }
    runtime["executor"].submit(
        _execute_batch_job,
        runtime,
        job_id,
        entries=entries,
        identity_root=identity_root[0] if identity_root else None,
        source_root=source_root[0] if source_root else None,
        modules=modules,
        services=services,
    )
    return job_id


def _product_version() -> str:
    try:
        config, _ = load_config(APP_CONFIG)
    except Exception:
        return "0.4.0"
    return str(config.get("display_version") or config.get("product_version") or "0.4.0")


def _render_task_tools(task_root: Path, key_prefix: str) -> None:
    """证据完整性校验和证据包导出；只读任务目录。"""

    with st.container(border=True):
        st.markdown("#### 证据固定")
        st.caption(
            "校验：按任务自带清单重算每个证据文件的 SHA-256。"
            "证据包：把任务目录打成 zip，并附文件校验清单，便于移交和归档。"
        )
        left, right = st.columns(2)
        if left.button("校验证据完整性", key=f"{key_prefix}_verify", width="stretch"):
            st.session_state[f"{key_prefix}_verify_result"] = verify_task_artifacts(task_root)
        bundle_key = f"{key_prefix}_bundle"
        if right.button("生成证据包", key=f"{key_prefix}_bundle_build", width="stretch"):
            try:
                st.session_state[bundle_key] = build_evidence_bundle(task_root)
                st.session_state.pop(f"{bundle_key}_error", None)
            except (OSError, ValueError) as exc:
                st.session_state.pop(bundle_key, None)
                st.session_state[f"{bundle_key}_error"] = str(exc)
        verify_result = st.session_state.get(f"{key_prefix}_verify_result")
        if isinstance(verify_result, Mapping):
            if not verify_result.get("available"):
                st.warning(str(verify_result.get("message")))
            elif verify_result.get("problems"):
                st.error(str(verify_result.get("message")))
                st.dataframe(verify_result["problems"], hide_index=True, width="stretch")
            else:
                st.success(str(verify_result.get("message")))
        if st.session_state.get(f"{bundle_key}_error"):
            st.error(str(st.session_state[f"{bundle_key}_error"]))
        bundle = st.session_state.get(bundle_key)
        if isinstance(bundle, bytes):
            st.download_button(
                f"下载证据包（{len(bundle) / 1048576:.1f} MB）",
                data=bundle,
                file_name=f"证据包_{task_root.name}.zip",
                mime="application/zip",
                key=f"{key_prefix}_bundle_download",
                width="stretch",
            )


def _render_loaded_results(mode: str) -> None:
    loaded = st.session_state.get("loaded_results")
    if not loaded or loaded["mode"] != mode:
        return

    st.success(f"{RESULT_MODE_LABELS.get(mode, mode)}结果文件格式与路径检查正常。")
    if loaded.get("legacy_dual_module"):
        st.warning(
            "检测到旧版双模块 task：缺少 detection_result.json，"
            "AIGC 页签明确标记为“未运行”，不伪造鉴伪结果。"
        )

    task_root = (
        MOCK_RESULTS_ROOT.resolve()
        if mode == "Mock Demo"
        else Path(str(loaded["source_label"])).expanduser().resolve(strict=False)
    )
    key_prefix = f"{mode.replace(' ', '_').lower()}_{task_root.name}"
    selected_page = render_evidence_workspace(
        deepfake=loaded["deepfake"],
        identity=loaded["identity"],
        source=loaded["source"],
        policy=loaded["policy"],
        task_dir=task_root,
        task_status=loaded.get("task_status"),
        final_report=loaded.get("final_report"),
        task_artifact_manifest=loaded.get("artifact_manifest"),
        product_version=_product_version(),
        key_prefix=key_prefix,
    )

    if mode != "Mock Demo" and selected_page == "证据链与完整报告":
        st.markdown("### 完整报告与导出")
        render_task_reports(
            task_root,
            task_id=task_root.name,
            key_prefix=key_prefix,
        )
    if mode != "Mock Demo":
        _render_task_tools(task_root, key_prefix)

    with st.expander("原始 JSON 与批准资源根"):
        st.markdown("#### 完整任务路径（技术详情）")
        st.code(str(loaded["source_label"]))
        st.markdown("#### AIGC 鉴伪 JSON")
        st.json(loaded["deepfake"])
        st.markdown("#### 人脸身份比对 JSON")
        st.json(loaded["identity"])
        st.markdown("#### 原始素材检索 JSON")
        st.json(loaded["source"])
        st.markdown("#### 本次批准的资源根")
        for root in loaded["policy"].display_roots():
            st.code(root)


def _module_pill(item: ModuleAvailability) -> str:
    if item.runnable:
        return pill("就绪", "ok")
    if item.empty:
        return pill("未初始化", "warn")
    return pill("不可用", "bad")


def _service_pill(bundle: Mapping[str, Any]) -> str:
    if _services_starting(bundle):
        return pill("启动中", "warn")
    if bundle.get("manager") is None:
        return pill("子进程模式", "warn")
    status = _runtime_service_snapshot(bundle)
    ready = [
        bool((status.get(key) or {}).get("ready"))
        for key in SERVICE_LABELS
        if isinstance(status.get(key), Mapping) and status[key].get("state") != "disabled"
    ]
    if ready and all(ready):
        return pill("运行中", "ok")
    return pill("部分可用", "warn")


def _render_sidebar_readiness(
    modules: Mapping[str, ModuleAvailability],
    service_bundle: Mapping[str, Any],
) -> None:
    rows = [
        ("鉴伪模型", _module_pill(modules["deepfake"])),
        ("通用 AIGC 模型", _module_pill(modules["general_aigc"])),
        ("身份库", _module_pill(modules["identity"])),
        ("素材库", _module_pill(modules["source"])),
        ("模型服务", _service_pill(service_bundle)),
    ]
    body = "".join(
        f'<div class="fsts-side-row"><span>{label}</span>{badge}</div>' for label, badge in rows
    )
    render_html(f'<div class="fsts-side-title">系统状态</div>{body}')


def _render_status_strip(
    snapshot: Mapping[str, Any],
) -> tuple[dict[str, ModuleAvailability], tuple[str, str]]:
    modules = {key: module_availability(snapshot, key) for key in (*MODULE_KEYS, "general_aigc")}
    device = device_label(snapshot)
    return modules, device


def _render_system_page(
    *,
    modules: Mapping[str, ModuleAvailability],
    device: tuple[str, str],
    service_bundle: Mapping[str, Any],
    inspection_error: str | None,
) -> None:
    st.subheader("系统状态")
    st.caption("这里列出每个模块实际用到的模型、数据库和运行环境。")
    render_llm_status(APP_CONFIG)
    columns = st.columns(5)
    for column, key in zip(columns[:4], ("deepfake", "general_aigc", "identity", "source"), strict=True):
        item = modules[key]
        column.metric(item.title, item.value)
        column.caption(item.caption)
    columns[4].metric("运算设备", device[0])
    columns[4].caption(device[1])
    if inspection_error:
        st.error(f"系统状态检查失败：{inspection_error}")

    deepfake_raw = modules["deepfake"].raw
    components = deepfake_raw.get("components")
    readiness = deepfake_raw.get("readiness")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### 模型服务")
            service_status = _runtime_service_snapshot(service_bundle)
            service_ready = {
                key: bool((service_status.get(key) or {}).get("ready"))
                for key in SERVICE_LABELS
                if isinstance(service_status.get(key), Mapping)
                and service_status[key].get("state") != "disabled"
            }
            if _services_starting(service_bundle):
                st.info("常驻模型服务正在后台启动；身份库首次使用时要先建索引，约需 2 分钟。")
            elif service_bundle.get("manager") is None:
                st.warning("常驻模型服务未启动；分析会改用隔离子进程，结果相同，只是每次要重新加载模型。")
            elif service_ready and all(service_ready.values()):
                st.success("模型已常驻内存，后续分析可直接复用。")
            else:
                ready_names = [SERVICE_LABELS[key] for key, ready in service_ready.items() if ready]
                ready_text = "、".join(ready_names) if ready_names else "无"
                st.warning(
                    f"常驻服务部分可用（已就绪：{ready_text}）；其余模块自动改用隔离子进程。"
                )
            if service_bundle.get("error"):
                st.caption(f"服务降级原因：{service_bundle['error']}")
            if st.button("重新检测 / 重试服务", width="stretch"):
                manager = service_bundle.get("manager")
                if manager is not None:
                    try:
                        manager.close()
                    except Exception:
                        pass
                _cached_runtime_services.clear()
                _cached_system_status.clear()
                st.rerun()
            with st.expander("服务运行详情"):
                st.json(service_status)
    with right:
        with st.container(border=True):
            st.markdown("#### 素材特征缓存")
            source_index = modules["source"].raw.get("index")
            source_index = source_index if isinstance(source_index, Mapping) else {}
            st.write(str(source_index.get("message") or "未配置特征缓存"))
            size_bytes = source_index.get("size_bytes")
            if isinstance(size_bytes, int) and size_bytes:
                st.caption(f"缓存索引大小：{size_bytes / 1048576:.1f} MB")
            _render_warm_controls()

    if isinstance(components, Mapping):
        with st.expander("鉴伪模型资源检查", expanded=not modules["deepfake"].runnable):
            labels = {
                "spatial_model": "空间域模型",
                "frequency_model": "频域模型",
                "fusion_config": "融合配置",
                "decision_threshold": "判定阈值配置",
                "python": "Python解释器",
                "torch": "torch",
                "torchvision": "torchvision",
                "sklearn": "sklearn",
                "opencv": "opencv",
                "cuda": "CUDA",
            }
            for key, label in labels.items():
                record = components.get(key)
                if not isinstance(record, Mapping):
                    st.error(f"{label}：未返回检查结果")
                    continue
                status = str(record.get("status") or "unknown")
                available = record.get("available")
                detail = record.get("message") or record.get("version") or record.get("value")
                icon = "✅" if status == "ok" else "⚠️" if key == "cuda" and available is False else "❌"
                st.write(f"{icon} **{label}**：{status}{f'（{detail}）' if detail is not None else ''}")
            ready_status = readiness.get("status") if isinstance(readiness, Mapping) else "unknown"
            if ready_status == "ready":
                st.success("鉴伪模块：就绪")
            else:
                missing = readiness.get("missing_resources", []) if isinstance(readiness, Mapping) else []
                st.error(f"鉴伪模块：不可用；缺少或异常资源：{'、'.join(str(item) for item in missing) or '未知'}")

    try:
        config, config_path = load_config(APP_CONFIG)
        upgrade = config.get("upgrade") if isinstance(config.get("upgrade"), Mapping) else {}
        kernels = upgrade.get("upgraded_kernels") if isinstance(upgrade.get("upgraded_kernels"), Mapping) else {}
        with st.expander("版本与目录"):
            key_values(
                [
                    ("系统版本", _product_version()),
                    ("基线契约", str(config.get("product_version") or "—")),
                    ("内核", "、".join(str(value) for value in kernels.values()) or "—"),
                    ("配置文件", str(config_path)),
                    ("任务目录", str(resolve_path(config["runtime"]["tasks_root"], config_path))),
                ]
            )
    except (OrchestrationError, KeyError, TypeError):
        pass


def _render_warm_controls() -> None:
    job_id = st.session_state.get("warm_job_id")
    job = _job_snapshot(str(job_id) if job_id else None)
    running = bool(job and str(job.get("status")) in {"queued", "running"})
    if st.button("建立 / 更新特征缓存", disabled=running, width="stretch", key="warm_source_cache"):
        try:
            config, config_path = load_config(APP_CONFIG)
            cache_value = config["source"].get("feature_cache")
            if not cache_value:
                raise ResultDataError("当前配置没有启用素材特征缓存。")
            library = resolve_path(config["source"]["library"], config_path)
            cache_dir = resolve_path(cache_value, config_path)
            new_id = f"warm_{secrets.token_hex(8)}"
            runtime = _live_job_runtime()
            with runtime["lock"]:
                runtime["jobs"][new_id] = {"job_id": new_id, "status": "queued"}
            runtime["executor"].submit(
                _execute_warm_job, runtime, new_id, library=library, cache_dir=cache_dir
            )
            st.session_state["warm_job_id"] = new_id
            st.rerun()
        except (ResultDataError, OrchestrationError, KeyError, OSError) as exc:
            st.error(f"无法建立缓存：{exc}")
    _render_warm_status()


@st.fragment(run_every=2.0)
def _render_warm_status() -> None:
    job_id = st.session_state.get("warm_job_id")
    job = _job_snapshot(str(job_id) if job_id else None)
    if not job:
        return
    status = str(job.get("status"))
    if status in {"queued", "running"}:
        st.info("正在建立素材特征缓存，图片较多时需要几分钟。")
    elif status == "success":
        result = job.get("result") if isinstance(job.get("result"), Mapping) else {}
        st.success(
            f"缓存已更新：{result.get('image_count')} 张图片，"
            f"耗时 {result.get('runtime_seconds')} 秒。"
        )
    else:
        st.error(f"建立缓存失败：{job.get('error')}")


def _module_meta(task_status: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = task_status.get(key)
    return dict(value) if isinstance(value, Mapping) else {"status": "skipped"}


def _render_live_summary() -> None:
    task_status = st.session_state.get("live_task_status")
    loaded = st.session_state.get("loaded_results")
    if not isinstance(task_status, Mapping) or not loaded:
        return
    if loaded.get("mode") != "Live Analysis":
        return

    badges = [
        pill(
            f"任务{status_label(task_status.get('status'))}",
            "ok" if task_status.get("status") == "success" else "warn",
        )
    ]
    total_runtime = task_status.get("runtime_seconds")
    if isinstance(total_runtime, (int, float)):
        badges.append(pill(f"总耗时 {float(total_runtime):.1f} 秒", "muted"))
    notes: list[str] = []
    for key in MODULE_KEYS:
        meta = _module_meta(task_status, key)
        status = str(meta.get("status") or meta.get("state") or "skipped")
        runtime = meta.get("runtime_seconds")
        label = "不适用" if meta.get("not_applicable") else status_label(status)
        text = f"{MODULE_SHORT_LABELS[key]} {label}"
        if isinstance(runtime, (int, float)) and status != "skipped":
            text += f" {float(runtime):.1f} 秒"
        tone = {"success": "ok", "skipped": "muted", "error": "bad"}.get(status, "warn")
        badges.append(pill(text, tone))
        if meta.get("fallback_reason"):
            notes.append(f"{MODULE_SHORT_LABELS[key]}：常驻服务不可用，已改用隔离子进程（{meta['fallback_reason']}）")
        error = meta.get("error") or meta.get("error_message")
        if status == "error" and error:
            notes.append(f"{MODULE_SHORT_LABELS[key]}：{error}")
    render_html(f'<div class="fsts-header">{"".join(badges)}</div>')
    for note in notes:
        st.caption(note)
    errors = task_status.get("errors")
    if isinstance(errors, list):
        for error in errors:
            st.error(str(error))


@st.fragment(run_every=0.5)
def _render_active_job() -> None:
    job_id = st.session_state.get("live_job_id")
    job = _job_snapshot(str(job_id) if job_id else None)
    if not job:
        return
    status = str(job.get("status") or "queued")
    selected = tuple(job.get("selected_modules") or MODULE_KEYS)
    task_dir = str(job.get("task_dir") or "").strip() or None
    if status in {"queued", "running"}:
        render_live_progress(
            task_dir=task_dir,
            selected_modules=selected,
            job_status=status,
            queued_message=(
                "当前任务正在等待前一个任务结束；系统按提交顺序串行执行，避免模型与素材库争用。"
                if status == "queued"
                else "正在创建独立任务目录……"
            ),
        )
        return
    error = job.get("error")
    if error and not task_dir:
        st.error(f"分析任务未能创建结果目录：{error}")
        return
    result = job.get("result")
    if not task_dir or not isinstance(result, Mapping):
        st.error(f"分析任务结束但没有返回可读结果：{error or status}")
        return

    if st.session_state.get("loaded_live_job_id") == job_id:
        return
    try:
        _load_task(
            task_dir,
            str(job.get("identity_root") or ""),
            str(job.get("source_root") or ""),
            "",
            mode="Live Analysis",
        )
    except (ResultDataError, OSError, RuntimeError) as exc:
        st.error(f"任务完成，但结果文件格式检查未通过：{exc}")
        return
    st.session_state["live_task_status"] = dict(result)
    st.session_state["live_selected_modules"] = selected
    st.session_state["loaded_live_job_id"] = job_id
    st.rerun(scope="app")


@st.fragment(run_every=3.0)
def _watch_service_start(service_bundle: Mapping[str, Any]) -> None:
    if not _services_starting(service_bundle):
        st.rerun(scope="app")


def _render_shutdown(service_bundle: Mapping[str, Any]) -> None:
    if st.button("关闭系统", help="停止本地服务并退出程序", width="stretch"):
        manager = service_bundle.get("manager")
        if manager is not None:
            try:
                manager.close()
            except Exception as exc:
                st.warning(f"持久服务关闭时返回警告：{exc}")
        st.success("系统正在关闭，可以关闭此浏览器页面。")
        threading.Timer(
            0.8,
            lambda: os.kill(os.getpid(), signal.SIGTERM),
        ).start()
        st.stop()


def _preview_for(query: Any, routing_edge: int) -> InputPreview:
    """同一个文件只计算一次哈希，避免大视频在每次页面刷新时重复读取。"""

    token = (str(query.name), getattr(query, "size", 0), str(getattr(query, "file_id", "")))
    cached = st.session_state.get("live_preview")
    if isinstance(cached, tuple) and cached[0] == token:
        return cached[1]
    preview = describe_input(str(query.name), query.getvalue(), routing_edge=routing_edge)
    st.session_state["live_preview"] = (token, preview)
    return preview


def _render_query_panel(query: Any, routing_edge: int) -> InputPreview | None:
    if query is None:
        st.info("选好文件后，这里会显示预览和文件哈希。")
        return None
    preview = _preview_for(query, routing_edge)
    media, details = st.columns([1, 1.7])
    with media:
        if preview.kind == "video":
            st.video(query.getvalue() if isinstance(query, ExampleQuery) else query)
        elif preview.readable:
            st.image(query.getvalue(), width="stretch")
    rows: list[tuple[str, str]] = [
        ("文件", preview.file_name),
        ("大小", preview.size_label),
    ]
    if preview.width and preview.height:
        rows.append(("尺寸", f"{preview.width} × {preview.height}"))
    if preview.route_label:
        rows.append(("整图分支", preview.route_label))
    if preview.route_reason:
        rows.append(("依据", preview.route_reason))
    rows.append(("SHA-256", preview.sha256))
    with details:
        key_values(rows)
    for note in preview.notes:
        st.warning(note)
    return preview


def _render_library_settings(
    live_config_error: str, default_identity: str, default_source: str
) -> tuple[str, str]:
    if live_config_error:
        st.error(f"配置读取失败：{live_config_error}")
        return default_identity, default_source
    config, config_path = load_config(APP_CONFIG)
    return render_case_library_settings(default_identity, default_source, config, config_path)


def _render_live_mode(
    *,
    modules: Mapping[str, ModuleAvailability],
    service_bundle: Mapping[str, Any],
) -> None:
    try:
        default_identity, default_source, _ = _live_defaults()
        config, _ = load_config(APP_CONFIG)
        routing_edge = routing_edge_from_config(config)
        live_config_error = ""
    except (OrchestrationError, KeyError, TypeError) as exc:
        default_identity, default_source = "", ""
        routing_edge = 256
        live_config_error = str(exc)

    active_job = _job_snapshot(st.session_state.get("live_job_id"))
    analysis_running = bool(
        active_job and str(active_job.get("status")) in {"queued", "running"}
    )
    loaded = st.session_state.get("loaded_results")
    showing_results = isinstance(loaded, Mapping) and loaded.get("mode") == "Live Analysis"
    if analysis_running or showing_results:
        # 分析开始后页面只留进度和结果，不用再滚过上传区。
        title, action = st.columns([4, 1])
        name = str((active_job or {}).get("query_name") or "")
        title.subheader(f"研判：{name}" if name else "研判结果")
        if action.button(
            "开始新的研判",
            disabled=analysis_running,
            width="stretch",
            key="live_new_analysis",
        ):
            for key in ("loaded_results", "live_task_status", "loaded_live_job_id", "live_job_id", "live_preview"):
                st.session_state.pop(key, None)
            st.rerun()
        _render_active_job()
        _render_live_summary()
        _render_loaded_results("Live Analysis")
        return

    st.subheader("新建研判")
    try:
        st.info(upload_notice(llm_context(APP_CONFIG)[2]))
    except (OSError, ValueError):
        st.warning("无法读取大模型配置，请先在设置页检查。")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        with st.container(border=True):
            step_title(1, "选择涉案图片或视频")
            uploaded_query = st.file_uploader(
                "上传涉案图片或视频",
                type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv"],
                accept_multiple_files=False,
                label_visibility="collapsed",
                help=(
                    "上传文件只暂存于本机用户数据目录，调度器为每次分析创建独立任务目录。"
                    "视频输入会先抽帧选关键帧，再逐帧鉴伪/身份检索，素材检索只用首要帧。"
                ),
                key="live_query",
            )
            st.caption("图片最大 20 MiB（JPG/PNG/WEBP），视频最大 200 MiB（MP4/MOV/AVI/MKV/WEBM）。大模型上传范围见上方提示。")
            examples = discover_examples()
            example_names = {EXAMPLE_LABELS[path.name]: path for path in examples}
            example_label = st.selectbox(
                "没有素材时可以用内置示例",
                ["不使用示例", *example_names],
                key="live_example",
                disabled=uploaded_query is not None,
            )
            identity_root_text, source_root_text = _render_library_settings(
                live_config_error, default_identity, default_source
            )
    query: Any = uploaded_query
    if query is None and example_label in example_names:
        query = ExampleQuery(example_names[example_label])
    with right:
        with st.container(border=True):
            step_title(2, "确认输入")
            preview = _render_query_panel(query, routing_edge)

    upload_error = ""
    if query is not None:
        try:
            _validate_upload(query)
        except ResultDataError as exc:
            upload_error = str(exc)
            st.error(upload_error)
    if preview is not None and not preview.readable:
        upload_error = upload_error or "文件无法读取。"
    upload_blocked = query is None or bool(upload_error)
    config_blocked = bool(live_config_error)
    starting = _services_starting(service_bundle)
    common_blocked = upload_blocked or config_blocked or starting

    with st.container(border=True):
        step_title(3, "选择分析方式")
        columns = st.columns([1.2, 1, 1, 1])
        comprehensive_clicked = columns[0].button(
            "一键综合分析",
            type="primary",
            disabled=common_blocked or any(not modules[key].runnable for key in MODULE_KEYS),
            width="stretch",
        )
        deepfake_clicked = columns[1].button(
            "AIGC 鉴伪",
            disabled=common_blocked or not modules["deepfake"].runnable,
            width="stretch",
        )
        identity_clicked = columns[2].button(
            "人脸身份比对",
            disabled=common_blocked or not modules["identity"].runnable,
            width="stretch",
        )
        source_clicked = columns[3].button(
            "原始素材检索",
            disabled=common_blocked or not modules["source"].runnable,
            width="stretch",
        )
        st.caption("各部分回答不同问题：是否生成或伪造、文件自带什么标识、原图在哪、可能是谁。结果并列呈现，不合成总概率。")

    for key in MODULE_KEYS:
        item = modules[key]
        if item.empty:
            st.error(f"{item.title}为空或未初始化：{item.message}；相应按钮已禁用。")
        elif not item.available:
            st.warning(f"{item.title}不可用：{item.message}；相应按钮已禁用。")
    if starting:
        st.info("模型服务正在后台启动，完成后按钮自动可用。身份库首次使用要先建索引，约需 2 分钟。")
        _watch_service_start(service_bundle)

    selected: frozenset[str] | None = None
    if deepfake_clicked:
        selected = frozenset({"deepfake"})
    elif identity_clicked:
        selected = frozenset({"identity"})
    elif source_clicked:
        selected = frozenset({"source"})
    elif comprehensive_clicked:
        selected = frozenset(MODULE_KEYS)

    if selected is not None:
        st.session_state.pop("loaded_results", None)
        st.session_state.pop("live_task_status", None)
        st.session_state.pop("loaded_live_job_id", None)
        try:
            job_id = _start_live_analysis(
                query,
                identity_root_text,
                source_root_text,
                selected,
                service_bundle.get("manager"),
            )
            st.session_state["live_job_id"] = job_id
            st.rerun()
        except (ResultDataError, OrchestrationError, OSError, RuntimeError) as exc:
            st.error(f"分析失败：{exc}")
        except Exception as exc:
            st.error(f"分析失败：{type(exc).__name__}: {exc}")

    # 上一个任务失败时没有结果可载入，错误信息留在输入页下方。
    _render_active_job()


@st.fragment(run_every=1.0)
def _render_batch_job() -> None:
    job = _job_snapshot(st.session_state.get("batch_job_id"))
    if not job:
        return
    items = list(job.get("items") or [])
    finished = sum(1 for item in items if item.get("status") not in {"queued", "running"})
    total = len(items)
    running = str(job.get("status")) in {"queued", "running"}
    st.progress(finished / total if total else 1.0, text=f"已完成 {finished} / {total}")
    columns = (
        ("deepfake_label", "鉴伪结论"),
        ("deepfake_score", "鉴伪分数"),
        ("identity_label", "身份结论"),
        ("identity_top", "身份第一候选"),
        ("source_label", "素材结论"),
        ("source_top", "素材第一候选"),
        ("runtime_seconds", "耗时(秒)"),
    )
    rows = []
    for item in items:
        summary = item.get("summary") if isinstance(item.get("summary"), Mapping) else {}
        rows.append(
            {
                "文件": item.get("name"),
                "状态": status_label(item.get("status")),
                **{label: display_cell(key, summary.get(key)) for key, label in columns},
                "说明": item.get("error") or "",
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    if running:
        return
    summaries = [item["summary"] for item in items if isinstance(item.get("summary"), Mapping)]
    if summaries:
        st.download_button(
            "导出结果表（CSV）",
            data=summaries_to_csv(summaries),
            file_name=f"批量研判结果_{job.get('job_id')}.csv",
            mime="text/csv; charset=utf-8",
            key="batch_csv",
        )
    st.caption("每张图片的完整证据都保存为独立任务，可以在“历史任务”里打开。")


def _render_batch_mode(
    *,
    modules: Mapping[str, ModuleAvailability],
    service_bundle: Mapping[str, Any],
) -> None:
    st.subheader("批量研判")
    try:
        st.info(upload_notice(llm_context(APP_CONFIG)[2]))
    except (OSError, ValueError):
        st.warning("无法读取大模型配置，请先在设置页检查。")
    st.caption(f"一次上传多张图片（最多 {MAX_BATCH_FILES} 张），系统逐张分析并汇总成一张结果表。")
    try:
        default_identity, default_source, _ = _live_defaults()
    except (OrchestrationError, KeyError, TypeError):
        default_identity, default_source = "", ""
    with st.container(border=True):
        step_title(1, "选择图片")
        uploads = st.file_uploader(
            "上传多张图片",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="batch_files",
        )
        step_title(2, "选择分析模块")
        columns = st.columns(3)
        chosen = [
            key
            for column, key in zip(columns, MODULE_KEYS, strict=True)
            if column.checkbox(
                MODULE_LABELS[key],
                value=key == "deepfake",
                disabled=not modules[key].runnable,
                key=f"batch_module_{key}",
            )
        ]
        job = _job_snapshot(st.session_state.get("batch_job_id"))
        running = bool(job and str(job.get("status")) in {"queued", "running"})
        starting = _services_starting(service_bundle)
        if starting:
            st.info("模型服务正在后台启动，完成后按钮自动可用。")
            _watch_service_start(service_bundle)
        if st.button(
            f"开始批量研判（{len(uploads or [])} 张）",
            type="primary",
            disabled=not uploads or not chosen or running or starting,
            key="batch_start",
        ):
            try:
                st.session_state["batch_job_id"] = _start_batch_analysis(
                    list(uploads or []),
                    default_identity,
                    default_source,
                    chosen,
                    service_bundle.get("manager"),
                )
                st.rerun()
            except (ResultDataError, OrchestrationError, OSError, RuntimeError) as exc:
                st.error(f"批量研判未能开始：{exc}")
    _render_batch_job()


def _render_history_mode() -> None:
    st.subheader("历史任务")
    try:
        (
            default_task_dir,
            default_identity_root,
            default_source_root,
        ) = _archived_task_defaults()
        config, config_path = load_config(APP_CONFIG)
        tasks_root = resolve_path(config["runtime"]["tasks_root"], config_path)
        archived_config_error = ""
    except (OrchestrationError, KeyError, TypeError) as exc:
        default_task_dir = default_identity_root = default_source_root = ""
        tasks_root = None
        archived_config_error = str(exc)
    if archived_config_error:
        st.error(f"归档任务配置读取失败：{archived_config_error}")

    summaries = list_tasks(tasks_root) if tasks_root is not None else []
    if summaries:
        st.caption(f"共 {len(summaries)} 个任务，最新的在前。每个任务的证据都独立保存在本机。")
        st.dataframe(summaries_to_table(summaries), hide_index=True, width="stretch", height=280)
        left, right = st.columns([3, 1])
        options = {
            f"{item['started_at']} · {item['file_name']} · {item['status_label']} · {item['task_id']}": item
            for item in summaries
        }
        chosen_label = left.selectbox("打开任务查看证据", ["请选择", *options], key="history_choice")
        right.download_button(
            "导出全部（CSV）",
            data=summaries_to_csv(summaries),
            file_name="历史任务汇总.csv",
            mime="text/csv; charset=utf-8",
            key="history_csv",
            width="stretch",
        )
        chosen = options.get(chosen_label)
        if chosen is not None and st.session_state.get("history_loaded") != chosen["task_dir"]:
            try:
                _load_task(
                    chosen["task_dir"],
                    chosen.get("identity_library") or default_identity_root,
                    chosen.get("source_library") or default_source_root,
                    "",
                )
                st.session_state["history_loaded"] = chosen["task_dir"]
            except (ResultDataError, OSError, RuntimeError) as exc:
                st.error(f"任务未能载入：{exc}")
    else:
        st.info("还没有任务。去“新建研判”分析一张图片后，结果会出现在这里。")

    with st.expander("从其他目录读取任务"):
        st.caption("用于打开其他电脑或旧版本生成的任务目录；只读取结果文件，不会修改。")
        task_dir_text = st.text_input("task_dir", value=default_task_dir)
        identity_root_text = st.text_input("批准的身份库根", value=default_identity_root)
        source_root_text = st.text_input("批准的素材库根", value=default_source_root)
        resource_root_text = st.text_input("额外批准资源根（可留空）", value="")
        if st.button(
            "读取 Task Result",
            type="primary",
            disabled=bool(archived_config_error),
        ):
            try:
                _load_task(
                    task_dir_text,
                    identity_root_text,
                    source_root_text,
                    resource_root_text,
                )
                st.session_state["history_loaded"] = task_dir_text
            except (ResultDataError, OSError, RuntimeError) as exc:
                st.error(f"Task Result 拒绝加载：{exc}")
    _render_loaded_results("Task Result")


def _render_guide_page() -> None:
    st.subheader("使用说明")
    st.markdown(
        """
#### 系统做什么

上传一张图片或一段视频，系统分别回答下面的问题，并把证据并列展示：

| 部分 | 回答的问题 | 怎么读结果 |
| --- | --- | --- |
| AIGC 鉴伪 | 是不是 AI 生成或换脸 | 看总体档位：证据不足 / 需人工复核 / 高风险提示。通用 AIGC 分支定主结论；人脸分支看人脸区域，最多把档位抬一级；整图分支只作提示。分数不相加 |
| 文件标识 | 文件自带什么来源信息 | 读 GB 45438 隐式标识、C2PA 内容凭证和元数据。有标识是线索，没有标识不说明任何事 |
| 原始素材检索 | 原图可能是哪张 | 看四态结论：高可信候选 / 证据不足 / 库中未发现可靠候选 / 本次无法比对。纯文生内容本来没有原图 |
| 人脸身份比对 | 这张脸更像库里的谁 | 同样看四态结论；只在有人脸时有意义，分数是相似度，不是身份确认 |

#### 基本流程

1. 在“新建研判”选择文件，或先用内置示例试一遍。
2. 确认右侧显示的文件信息和哈希。
3. 点“一键综合分析”，等待进度完成。
4. 依次查看各结果页，需要留存时导出报告或生成证据包。

大量图片用“批量研判”，之前的结果都在“历史任务”里。

#### 必须知道的边界

- 系统只提供侦查线索和辅助研判信息，不替代原始介质固定、哈希校验、人工复核或司法鉴定。
- 各部分的分数含义不同，不能相加，也没有“综合可信度”。
- 身份和素材只在当前库内比较：库里没有的人或图不可能被找到，高可信候选也不等于确认身份或来源。
- 分档边界来自有限的真实样本：强压缩、大幅缩放、翻拍或来源差异大的内容，档位可能失效。
- 换脸是最弱的一环：补充盲测上大约三分之二的换脸图进不了复核，低清视频上基本检不出来。“证据不足”不能排除换脸，要结合素材检索和身份比对。

#### 常见问题

- **第一次分析为什么慢？** 身份库要建索引，素材库要建特征缓存，只做一次。之后素材检索只需几秒。
- **分析结果在哪里？** 在程序目录旁边的“用户数据”文件夹里，每个任务一个目录。
- **怎么退出？** 点左下角“关闭系统”，或直接关掉启动时弹出的黑色窗口。
"""
    )


def main() -> None:
    full_name, short_name = product_names()
    st.set_page_config(
        page_title=short_name,
        page_icon="🔍",
        layout="wide",
    )
    inject_theme()

    status_result = _cached_system_status(str(APP_CONFIG))
    snapshot = status_result["snapshot"]
    modules, device = _render_status_strip(snapshot)
    service_bundle = _cached_runtime_services(str(APP_CONFIG))

    identity_libraries = discover_libraries(IDENTITY_LIBRARY_ROOT)
    source_libraries = discover_libraries(SOURCE_LIBRARY_ROOT)

    with st.sidebar:
        render_html(
            f'<div class="fsts-brand"><div class="fsts-brand-name">{escape(short_name)}</div>'
            '<div class="fsts-brand-sub">图片和视频的 AIGC 检测与溯源</div></div>'
        )
        mode_options = ["Live Analysis", "Batch Analysis", "Task Result", "System Status", "LLM Settings", "Guide"]
        if os.environ.get("FSTS_ENABLE_DEV_MOCK") == "1":
            mode_options.append("Mock Demo")
        mode = st.radio(
            "功能导航",
            mode_options,
            format_func=PAGE_LABELS.get,
            label_visibility="collapsed",
            width="stretch",
        )
        st.divider()
        _render_sidebar_readiness(modules, service_bundle)
        st.divider()
        if mode == "Mock Demo":
            st.warning("模拟数据：仅用于隐藏的开发布局测试，不是算法结果或准确率证据。")
            identity_names = [path.name for path in identity_libraries]
            source_names = [path.name for path in source_libraries]
            selected_identity = st.selectbox(
                "选择演示身份库",
                identity_names or ["(无演示库)"],
                disabled=not identity_names,
            )
            selected_source = st.selectbox(
                "选择演示素材库",
                source_names or ["(无演示库)"],
                disabled=not source_names,
            )
            load_button = st.button(
                "加载 Mock 演示",
                type="primary",
                disabled=not identity_names or not source_names,
                width="stretch",
            )
        else:
            selected_identity = selected_source = ""
            load_button = False
        _render_shutdown(service_bundle)

    st.title(full_name)
    render_html(
        '<div class="fsts-header">'
        + pill(f"版本 {_product_version()}")
        + pill("本地检测 · 大模型按配置调用", "muted")
        + '<span class="fsts-header-note">结果仅作侦查线索和辅助研判，不替代司法鉴定。</span>'
        + "</div>"
    )
    if status_result.get("error"):
        st.error(f"系统状态检查失败：{status_result['error']}")

    if mode == "Mock Demo":
        st.warning("当前页面正在显示模拟数据，禁止将其作为检测或检索结论。")
        st.subheader("开发模拟数据")
        uploaded_preview = st.file_uploader(
            "可选：仅在内存中预览查询图片",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            help="该图片不会写入源码目录。",
            key="mock_preview",
        )
        if uploaded_preview is not None:
            st.image(
                uploaded_preview,
                caption=f"仅内存预览：{uploaded_preview.name}",
                width=420,
            )
        else:
            st.info("无需上传即可加载内置 Mock。")
        if load_button:
            try:
                identity_path = next(
                    path for path in identity_libraries if path.name == selected_identity
                )
                source_path = next(
                    path for path in source_libraries if path.name == selected_source
                )
                _load_mock(identity_path, source_path)
            except (ResultDataError, StopIteration) as exc:
                st.error(f"Mock 结果拒绝加载：{exc}")
        _render_loaded_results("Mock Demo")
    elif mode == "Batch Analysis":
        _render_batch_mode(modules=modules, service_bundle=service_bundle)
    elif mode == "Task Result":
        _render_history_mode()
    elif mode == "System Status":
        _render_system_page(
            modules=modules,
            device=device,
            service_bundle=service_bundle,
            inspection_error=status_result.get("error"),
        )
    elif mode == "LLM Settings":
        render_llm_settings(APP_CONFIG)
    elif mode == "Guide":
        _render_guide_page()
    else:
        _render_live_mode(
            modules=modules,
            service_bundle=service_bundle,
        )


if __name__ == "__main__":
    main()
