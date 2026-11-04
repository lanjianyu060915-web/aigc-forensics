from __future__ import annotations

import json
import re
import secrets
import shutil
import sys
from pathlib import Path
from typing import Iterator

import pytest

import launcher
from orchestrator import _new_task_dir, run_task


FIXTURES = Path(__file__).resolve().parent / "fixtures"
TEMP_ROOT = Path(__file__).resolve().parent / ".run_tmp"
TASK_ID_PATTERN = re.compile(r"^case_\d{8}_\d{6}_[0-9a-f]{4}$")


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    """Avoid pytest's user temp root, which may be ACL-restricted on Windows."""
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEMP_ROOT / f"test_{secrets.token_hex(8)}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _write_config(tmp_path: Path) -> Path:
    identity_library = tmp_path / "identity_library"
    source_library = tmp_path / "source_library"
    identity_library.mkdir()
    source_library.mkdir()
    (identity_library / "person_001").mkdir()
    (identity_library / "person_001" / "reference.jpg").write_bytes(b"fixture")
    (source_library / "source.jpg").write_bytes(b"fixture")

    config = {
        "schema_version": "test",
        "runtime": {
            "tasks_root": str(tmp_path / "tasks"),
            "inbox_root": str(tmp_path / "inbox"),
            "identity_cache": str(tmp_path / "cache"),
        },
        "deepfake": {
            "enabled": False,
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
            "timeout_seconds": 10,
        },
        "frontend": {"python": sys.executable},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config_path


def test_task_id_format_and_uniqueness(workspace_tmp_path: Path) -> None:
    first_id, first_dir = _new_task_dir(workspace_tmp_path / "tasks")
    second_id, second_dir = _new_task_dir(workspace_tmp_path / "tasks")

    assert TASK_ID_PATTERN.fullmatch(first_id)
    assert TASK_ID_PATTERN.fullmatch(second_id)
    assert first_id != second_id
    assert first_dir != second_dir
    assert first_dir.is_dir()
    assert second_dir.is_dir()


def test_missing_query_is_controlled(
    workspace_tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(workspace_tmp_path)

    exit_code = launcher.main(
        [
            "--query",
            str(workspace_tmp_path / "missing.jpg"),
            "--config",
            str(config_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code != 0
    assert "受控错误" in captured.err
    assert "查询图片不存在或不是文件" in captured.err
    assert "Traceback" not in captured.err
    assert not (workspace_tmp_path / "tasks").exists()


def test_source_runs_after_identity_failure(workspace_tmp_path: Path) -> None:
    config_path = _write_config(workspace_tmp_path)
    query = workspace_tmp_path / "query.jpg"
    query.write_bytes(b"lightweight fixture; image decoding is intentionally skipped")

    status = run_task(query, config_path=config_path)
    task_dir = Path(status["task_dir"])
    identity_result = json.loads(
        (task_dir / "identity_result.json").read_text(encoding="utf-8")
    )
    source_result = json.loads(
        (task_dir / "source_result.json").read_text(encoding="utf-8")
    )
    unified = json.loads((task_dir / "unified_result.json").read_text(encoding="utf-8"))

    assert status["status"] == "partial_success"
    assert status["deepfake"]["status"] == "error"
    assert status["identity"]["status"] == "error"
    assert status["source"]["status"] == "success"
    assert identity_result["status"] == "error"
    assert source_result["status"] == "success"
    assert source_result["candidates"][0]["source_id"] == "fixture-source"
    assert unified["identity_search"]["status"] == "error"
    assert unified["source_search"]["status"] == "success"
    assert (task_dir / "logs" / "identity_stderr.log").is_file()
    assert (task_dir / "logs" / "source_stdout.log").read_text(
        encoding="utf-8"
    ).strip() == "SOURCE_FIXTURE_RAN"
