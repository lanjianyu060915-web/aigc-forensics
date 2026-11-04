from __future__ import annotations

import json
import secrets
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from orchestrator import OrchestrationError, run_task
from runtime_services import RuntimeServiceManager
from system_status import inspect_system_status


FIXTURES = Path(__file__).resolve().parent / "fixtures"
MOCK_PERSISTENT_WORKER = FIXTURES / "mock_persistent_worker.py"
TEMP_ROOT = Path(__file__).resolve().parent / ".backend_tmp"


@pytest.fixture
def backend_tmp_path() -> Iterator[Path]:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEMP_ROOT / f"test_{secrets.token_hex(8)}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _write_config(tmp_path: Path, *, source_timeout: float = 10) -> Path:
    identity_library = tmp_path / "identity_library"
    source_library = tmp_path / "source_library"
    (identity_library / "person_001").mkdir(parents=True)
    source_library.mkdir()
    (identity_library / "person_001" / "reference.jpg").write_bytes(b"fixture")
    (source_library / "source.jpg").write_bytes(b"fixture")
    config: dict[str, Any] = {
        "schema_version": "test",
        "runtime": {
            "tasks_root": str(tmp_path / "tasks"),
            "inbox_root": str(tmp_path / "inbox"),
            "identity_cache": str(tmp_path / "cache"),
        },
        "deepfake": {
            "enabled": True,
            "python": str(tmp_path / "missing_deepfake_python.exe"),
            "runner": str(tmp_path / "missing_deepfake_runner.py"),
            "install_root": str(tmp_path / "missing_install"),
            "torch_home": str(tmp_path / "deepfake_cache"),
            "timeout_seconds": 10,
        },
        "identity": {
            "python": sys.executable,
            "runner": str(FIXTURES / "failing_identity_runner.py"),
            "library": str(identity_library),
            "top_k": 5,
            "timeout_seconds": 10,
            "ctx": -1,
            "model_root": str(tmp_path / "models"),
            "offline": True,
        },
        "source": {
            "python": sys.executable,
            "runner": str(FIXTURES / "successful_source_runner.py"),
            "library": str(source_library),
            "top_k": 5,
            "timeout_seconds": source_timeout,
        },
        "frontend": {"python": sys.executable},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return path


class SuccessfulServices:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run_deepfake(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("deepfake", kwargs))
        return {
            "status": "success",
            "query_image": str(kwargs["query_path"]),
            "is_deepfake": False,
            "score": 0.1,
            "label": "real",
            "runtime_seconds": 0.001,
            "details": {"fixture": True},
            "error_message": None,
        }

    def run_identity(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("identity", kwargs))
        return {
            "status": "success",
            "query_image": str(kwargs["query_path"]),
            "candidates": [{"rank": 1, "identity_id": "person_001", "score": 1.0}],
            "runtime_seconds": 0.001,
            "error_message": None,
        }


def test_all_modules_use_services_and_emit_required_progress(
    backend_tmp_path: Path,
) -> None:
    config_path = _write_config(backend_tmp_path)
    query = backend_tmp_path / "query.jpg"
    query.write_bytes(b"fixture")
    services = SuccessfulServices()
    progress: list[dict[str, Any]] = []

    status = run_task(
        query,
        config_path=config_path,
        services=services,
        progress_callback=progress.append,
    )
    task_dir = Path(status["task_dir"])

    assert status["status"] == "success"
    assert status["deepfake"]["execution_mode"] == "service"
    assert status["identity"]["execution_mode"] == "service"
    assert status["source"]["execution_mode"] == "subprocess"
    assert [name for name, _kwargs in services.calls] == ["deepfake", "identity"]
    required_event_fields = {
        "event_id",
        "timestamp",
        "task_id",
        "module",
        "stage",
        "status",
        "message",
        "progress_current",
        "progress_total",
        "progress_percent",
        "elapsed_seconds",
        "details",
    }
    assert progress
    assert all(required_event_fields <= event.keys() for event in progress)
    assert all(event["task_id"] == status["task_id"] for event in progress)
    stages = [event["stage"] for event in progress]
    assert stages[0] == "task_created"
    assert "deepfake_inference_running" in stages
    assert "identity_service_reuse" in stages
    assert "library_checked" in stages
    assert stages[-1] == "completed"
    assert progress[-1]["status"] == "success"
    assert json.loads((task_dir / "detection_result.json").read_text(encoding="utf-8"))["status"] == "success"
    assert json.loads((task_dir / "identity_result.json").read_text(encoding="utf-8"))["status"] == "success"
    assert json.loads((task_dir / "source_result.json").read_text(encoding="utf-8"))["status"] == "success"
    deepfake_call = services.calls[0][1]
    assert Path(deepfake_call["output_path"]).name == ".deepfake_service_result.json"


def test_deepfake_insufficient_evidence_does_not_claim_model_scores(
    backend_tmp_path: Path,
) -> None:
    class InsufficientServices(SuccessfulServices):
        def run_deepfake(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("deepfake", kwargs))
            return {
                "status": "success",
                "query_image": str(kwargs["query_path"]),
                "is_deepfake": None,
                "score": None,
                "label": "insufficient_evidence",
                "runtime_seconds": 0.001,
                "details": {"valid_frame_count": 0, "invalid_frame_count": 1},
                "warnings": ["未检测到满足质量要求的人脸。"],
                "error_message": None,
            }

    config_path = _write_config(backend_tmp_path)
    query = backend_tmp_path / "query.jpg"
    query.write_bytes(b"fixture")
    progress: list[dict[str, Any]] = []

    status = run_task(
        query,
        config_path=config_path,
        selected_modules={"deepfake"},
        services=InsufficientServices(),
        progress_callback=progress.append,
    )

    assert status["status"] == "success"
    stages = [event["stage"] for event in progress]
    assert "deepfake_inference_skipped" in stages
    assert "spatial_inference_completed" not in stages
    assert "frequency_inference_completed" not in stages
    assert "fusion_and_threshold_completed" not in stages


def test_selected_modules_write_explicit_skipped_results(backend_tmp_path: Path) -> None:
    config_path = _write_config(backend_tmp_path)
    query = backend_tmp_path / "query.jpg"
    query.write_bytes(b"fixture")

    status = run_task(query, config_path=config_path, selected_modules={"source"})
    task_dir = Path(status["task_dir"])
    unified = json.loads((task_dir / "unified_result.json").read_text(encoding="utf-8"))

    assert status["status"] == "success"
    assert status["selected_modules"] == ["source"]
    assert status["deepfake"]["status"] == "skipped"
    assert status["identity"]["status"] == "skipped"
    assert status["source"]["status"] == "success"
    assert unified["deepfake_detection"]["status"] == "skipped"
    assert unified["identity_search"]["status"] == "skipped"
    assert unified["errors"] == []
    assert json.loads((task_dir / "detection_result.json").read_text(encoding="utf-8"))["status"] == "skipped"
    assert json.loads((task_dir / "identity_result.json").read_text(encoding="utf-8"))["status"] == "skipped"


def test_real_service_manager_protocol_integrates_with_orchestrator(
    backend_tmp_path: Path,
) -> None:
    config_path = _write_config(backend_tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    common = {
        "python": sys.executable,
        "persistent_worker": str(MOCK_PERSISTENT_WORKER),
        "persistent_worker_args": [],
        "persistent_enabled": True,
        "startup_timeout_seconds": 5,
    }
    config["deepfake"].update(common)
    config["identity"].update(common)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    query = backend_tmp_path / "query.jpg"
    query.write_bytes(b"fixture")

    manager = RuntimeServiceManager(config_path)
    try:
        manager.start()
        status = run_task(query, config_path=config_path, services=manager)
    finally:
        manager.close()

    assert status["status"] == "success"
    assert status["deepfake"]["execution_mode"] == "service"
    assert status["identity"]["execution_mode"] == "service"
    assert status["source"]["execution_mode"] == "subprocess"


def test_service_failure_falls_back_without_stopping_source(backend_tmp_path: Path) -> None:
    config_path = _write_config(backend_tmp_path)
    query = backend_tmp_path / "query.jpg"
    query.write_bytes(b"fixture")

    class FailingIdentityService:
        def run_identity(self, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("worker unavailable")

    status = run_task(
        query,
        config_path=config_path,
        selected_modules={"identity", "source"},
        services=FailingIdentityService(),
    )

    assert status["status"] == "partial_success"
    assert status["identity"]["status"] == "error"
    assert status["identity"]["execution_mode"] == "subprocess"
    assert "worker unavailable" in status["identity"]["fallback_reason"]
    assert status["source"]["status"] == "success"


def test_empty_library_is_immediately_unavailable_and_service_is_not_called(
    backend_tmp_path: Path,
) -> None:
    config_path = _write_config(backend_tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    empty_library = backend_tmp_path / "empty_source"
    empty_library.mkdir()
    config["source"]["library"] = str(empty_library)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    query = backend_tmp_path / "query.jpg"
    query.write_bytes(b"fixture")

    status = run_task(query, config_path=config_path, selected_modules={"source"})

    assert status["status"] == "error"
    assert status["source"]["execution_mode"] == "preflight"
    assert "图库为空" in status["source"]["error"]


def test_subprocess_timeout_is_structured(backend_tmp_path: Path) -> None:
    config_path = _write_config(backend_tmp_path, source_timeout=0.1)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["source"]["runner"] = str(FIXTURES / "slow_search_runner.py")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    query = backend_tmp_path / "query.jpg"
    query.write_bytes(b"fixture")

    status = run_task(query, config_path=config_path, selected_modules={"source"})

    assert status["status"] == "error"
    assert status["source"]["timed_out"] is True
    assert "超时" in status["source"]["error"]


def test_unknown_selected_module_is_controlled(backend_tmp_path: Path) -> None:
    config_path = _write_config(backend_tmp_path)
    query = backend_tmp_path / "query.jpg"
    query.write_bytes(b"fixture")

    with pytest.raises(OrchestrationError, match="未知模块"):
        run_task(query, config_path=config_path, selected_modules={"unknown"})


def test_empty_selected_modules_is_rejected(backend_tmp_path: Path) -> None:
    config_path = _write_config(backend_tmp_path)
    query = backend_tmp_path / "query.jpg"
    query.write_bytes(b"fixture")

    with pytest.raises(OrchestrationError, match="至少必须包含一个模块"):
        run_task(query, config_path=config_path, selected_modules=set())


@pytest.mark.parametrize("module", ["deepfake", "identity", "source"])
def test_selected_disabled_module_is_an_error(
    backend_tmp_path: Path,
    module: str,
) -> None:
    config_path = _write_config(backend_tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[module]["enabled"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")
    query = backend_tmp_path / "query.jpg"
    query.write_bytes(b"fixture")

    status = run_task(
        query,
        config_path=config_path,
        selected_modules={module},
    )

    assert status["status"] == "error"
    assert status[module]["status"] == "error"
    assert status[module]["execution_mode"] == "disabled"
    assert "配置关闭" in status[module]["error"]


@pytest.mark.parametrize("timeout", [float("nan"), float("inf")])
def test_nonfinite_module_timeout_is_rejected(
    backend_tmp_path: Path,
    timeout: float,
) -> None:
    config_path = _write_config(backend_tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["source"]["timeout_seconds"] = timeout
    config_path.write_text(json.dumps(config), encoding="utf-8")
    query = backend_tmp_path / "query.jpg"
    query.write_bytes(b"fixture")

    status = run_task(
        query,
        config_path=config_path,
        selected_modules={"source"},
    )

    assert status["status"] == "error"
    assert status["source"]["execution_mode"] == "preflight"
    assert "有限数值" in status["source"]["error"]


def test_service_fallback_shares_one_module_deadline(
    backend_tmp_path: Path,
) -> None:
    config_path = _write_config(backend_tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["identity"]["timeout_seconds"] = 0.12
    # If a second full timeout were granted, this runner would turn the task
    # into a false success after the service attempt had already used its budget.
    config["identity"]["runner"] = str(FIXTURES / "successful_source_runner.py")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    query = backend_tmp_path / "query.jpg"
    query.write_bytes(b"fixture")

    class SlowIdentityService:
        def run_identity(self, **kwargs: Any) -> dict[str, Any]:
            time.sleep(0.4)
            return {
                "status": "success",
                "query_image": str(kwargs["query_path"]),
                "candidates": [],
                "runtime_seconds": 0.4,
                "error_message": None,
            }

    status = run_task(
        query,
        config_path=config_path,
        selected_modules={"identity"},
        services=SlowIdentityService(),
    )
    assert status["status"] == "error"
    assert status["identity"]["timed_out"] is True
    assert status["identity"]["execution_mode"] == "deadline_exhausted"
    assert "总预算" in status["identity"]["error"]


def test_system_status_marks_empty_libraries_unavailable(backend_tmp_path: Path) -> None:
    config_path = _write_config(backend_tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    empty_identity = backend_tmp_path / "empty_identity"
    empty_source = backend_tmp_path / "empty_source"
    empty_identity.mkdir()
    empty_source.mkdir()
    config["identity"]["library"] = str(empty_identity)
    config["source"]["library"] = str(empty_source)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    status = inspect_system_status(config_path)

    assert status["identity"]["available"] is False
    assert status["identity"]["library"]["status"] == "empty"
    assert status["source"]["available"] is False
    assert status["source"]["library"]["status"] == "empty"
    assert status["source"]["index_status"] == "not_persistent"
    assert status["source"]["index"]["persistent"] is False
    assert status["source"]["index"]["build_policy"] == "realtime_scan"
