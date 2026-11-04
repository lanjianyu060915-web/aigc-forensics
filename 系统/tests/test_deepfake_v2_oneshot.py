"""二代鉴伪内核单次入口：与常驻路径同一内核、同一分数。这里使用真实模型。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from orchestrator import load_config, resolve_path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.json"
EXAMPLES = ROOT / "examples"


def _section() -> tuple[dict, Path]:
    config, config_path = load_config(CONFIG)
    return config["deepfake"], config_path


def _environment(tmp_path: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["FSTS_DATA_ROOT"] = str(tmp_path / "data")
    return environment


def _run_oneshot(query: Path, output: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    section, config_path = _section()
    return subprocess.run(
        [
            str(resolve_path(section["python"], config_path)),
            *section["python_args"],
            str(resolve_path(section["oneshot_runner"], config_path)),
            "--query",
            str(query),
            "--output",
            str(output),
            "--install-root",
            str(resolve_path(section["install_root"], config_path)),
            "--device",
            "cpu",
        ],
        cwd=str(resolve_path(section["oneshot_runner"], config_path).parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env=_environment(tmp_path),
    )


def _persistent_score(query: Path, output: Path, tmp_path: Path) -> dict:
    section, config_path = _section()
    request = {
        "protocol": "face-source-trace-runtime-v1",
        "type": "request",
        "request_id": "oneshot_compare",
        "action": "run",
        "query_path": str(query),
        "output_path": str(output),
    }
    shutdown = {**request, "request_id": "stop", "action": "shutdown"}
    completed = subprocess.run(
        [
            str(resolve_path(section["python"], config_path)),
            *section["python_args"],
            str(resolve_path(section["persistent_worker"], config_path)),
            "--install-root",
            str(resolve_path(section["install_root"], config_path)),
            "--writable-root",
            str(tmp_path / "persistent"),
        ],
        cwd=str(resolve_path(section["persistent_worker"], config_path).parent),
        input=json.dumps(request) + "\n" + json.dumps(shutdown) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["integrated_query_002.jpg"])
def test_oneshot_and_persistent_paths_give_the_same_score(name: str, tmp_path: Path) -> None:
    query = EXAMPLES / name
    oneshot_output = tmp_path / "oneshot" / "detection_result.json"
    persistent_output = tmp_path / "persistent_out" / "detection_result.json"

    completed = _run_oneshot(query, oneshot_output, tmp_path)
    assert completed.returncode == 0, completed.stderr

    oneshot = json.loads(oneshot_output.read_text(encoding="utf-8"))
    persistent = _persistent_score(query, persistent_output, tmp_path)

    assert oneshot["status"] == "success"
    assert oneshot["method"]["branch"] == persistent["method"]["branch"]
    assert oneshot["score"] == persistent["score"]
    assert oneshot["details"].get("band") == persistent["details"].get("band")


def test_oneshot_reports_structured_error_for_missing_query(tmp_path: Path) -> None:
    output = tmp_path / "out" / "detection_result.json"
    completed = _run_oneshot(tmp_path / "missing.jpg", output, tmp_path)

    assert completed.returncode == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["error_message"]


def test_config_keeps_frozen_runner_for_hash_validation() -> None:
    section, _ = _section()
    assert section["runner"].endswith("run_deepfake_detection.py")
    assert section["oneshot_runner"].endswith("run_deepfake_detection_v2.py")
