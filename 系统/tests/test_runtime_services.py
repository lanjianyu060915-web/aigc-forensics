from __future__ import annotations

import json
from pathlib import Path
import secrets
import shutil
import sys
from typing import Iterator

import pytest

from runtime_services import (
    RuntimeServiceError,
    RuntimeServiceManager,
    RuntimeServiceTimeout,
    RuntimeServiceUnavailable,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mock_persistent_worker.py"
TEMP_ROOT = Path(__file__).resolve().parent / ".runtime_service_tmp"


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEMP_ROOT / f"test_{secrets.token_hex(8)}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _config(tmp_path: Path, *, identity_enabled: bool = True) -> Path:
    library = tmp_path / "library"
    library.mkdir()
    shared = {
        "python": sys.executable,
        "persistent_worker": str(FIXTURE),
        "persistent_worker_args": [],
        "startup_timeout_seconds": 5,
        "timeout_seconds": 5,
    }
    payload = {
        "runtime": {
            "tasks_root": str(tmp_path / "tasks"),
            "identity_cache": str(tmp_path / "cache"),
        },
        "deepfake": dict(shared),
        "identity": {
            **shared,
            "persistent_enabled": identity_enabled,
            "library": str(library),
            "model_root": str(tmp_path / "models"),
            "ctx": -1,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_manager_reuses_workers_and_ignores_stdout_noise(
    workspace_tmp_path: Path,
) -> None:
    events: list[dict] = []
    manager = RuntimeServiceManager(
        _config(workspace_tmp_path), progress_callback=events.append
    )
    try:
        status = manager.start()
        assert status["ready"] is True
        assert status["deepfake"]["healthy"] is True
        assert status["identity"]["healthy"] is True
        assert status["deepfake"]["noise_lines_ignored"] >= 2
        assert status["identity"]["noise_lines_ignored"] >= 2
        deepfake_pid = status["deepfake"]["pid"]
        identity_pid = status["identity"]["pid"]

        query = workspace_tmp_path / "query.jpg"
        query.write_bytes(b"mock query; no image decoding")
        deepfake_output = workspace_tmp_path / "deepfake.json"
        identity_output = workspace_tmp_path / "identity.json"
        library = workspace_tmp_path / "library"

        first = manager.run_deepfake(
            query_path=query,
            output_path=deepfake_output,
            timeout=3,
        )
        second = manager.run_deepfake(query, workspace_tmp_path / "deepfake2.json", 3)
        identity = manager.run_identity(
            query_path=query,
            output_path=identity_output,
            library_path=library,
            top_k=7,
            timeout=3,
        )

        assert first["status"] == "success"
        assert second["status"] == "success"
        assert identity["mock_top_k"] == 7
        assert json.loads(deepfake_output.read_text(encoding="utf-8")) == first
        assert json.loads(identity_output.read_text(encoding="utf-8")) == identity

        after = manager.get_status()
        assert after["deepfake"]["pid"] == deepfake_pid
        assert after["identity"]["pid"] == identity_pid
        assert after["deepfake"]["requests_completed"] == 2
        assert after["identity"]["requests_completed"] == 1
        assert any(event.get("state") == "ready" for event in events)
    finally:
        manager.close()


def test_timeout_is_controlled_and_worker_is_retired(workspace_tmp_path: Path) -> None:
    manager = RuntimeServiceManager(
        _config(workspace_tmp_path, identity_enabled=False)
    )
    try:
        manager.start()
        query = workspace_tmp_path / "timeout.jpg"
        query.write_bytes(b"timeout fixture")
        with pytest.raises(RuntimeServiceTimeout):
            manager.run_deepfake(query, workspace_tmp_path / "result.json", 0.05)
        status = manager.get_status()["deepfake"]
        assert status["ready"] is False
        assert status["state"] == "error"
    finally:
        manager.close()


def test_unexpected_worker_exit_is_detected(workspace_tmp_path: Path) -> None:
    manager = RuntimeServiceManager(
        _config(workspace_tmp_path, identity_enabled=False)
    )
    try:
        manager.start()
        query = workspace_tmp_path / "exit.jpg"
        query.write_bytes(b"exit fixture")
        with pytest.raises(RuntimeServiceUnavailable):
            manager.run_deepfake(query, workspace_tmp_path / "result.json", 3)
        assert manager.get_status()["deepfake"]["ready"] is False
    finally:
        manager.close()


def test_global_persistent_worker_switch_is_a_hard_disable(
    workspace_tmp_path: Path,
) -> None:
    config_path = _config(workspace_tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["runtime"]["persistent_workers_enabled"] = False
    payload["deepfake"]["persistent_enabled"] = True
    payload["identity"]["persistent_enabled"] = True
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    manager = RuntimeServiceManager(config_path)
    try:
        status = manager.start()
        assert status["health"] == "disabled"
        assert status["deepfake"]["state"] == "disabled"
        assert status["identity"]["state"] == "disabled"
        assert status["deepfake"]["pid"] is None
        assert status["identity"]["pid"] is None
    finally:
        manager.close()


@pytest.mark.parametrize("timeout", [float("nan"), float("inf")])
def test_nonfinite_request_timeout_is_rejected(
    workspace_tmp_path: Path,
    timeout: float,
) -> None:
    manager = RuntimeServiceManager(
        _config(workspace_tmp_path, identity_enabled=False)
    )
    try:
        manager.start()
        query = workspace_tmp_path / "query.jpg"
        query.write_bytes(b"fixture")
        with pytest.raises(RuntimeServiceError, match="有限数值"):
            manager.run_deepfake(
                query,
                workspace_tmp_path / "result.json",
                timeout,
            )
    finally:
        manager.close()
