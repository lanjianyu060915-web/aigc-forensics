"""2.1 新增的编排行为：底图缓存参数、原始文件名、鉴伪回退走二代内核。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import textwrap

from orchestrator import run_task


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _config(tmp_path: Path, *, source_extra: dict | None = None, deepfake: dict | None = None) -> Path:
    identity_library = tmp_path / "identity_library"
    source_library = tmp_path / "source_library"
    (identity_library / "person_001").mkdir(parents=True)
    source_library.mkdir()
    (identity_library / "person_001" / "reference.jpg").write_bytes(b"fixture")
    (source_library / "source.jpg").write_bytes(b"fixture")
    config = {
        "schema_version": "test",
        "runtime": {
            "tasks_root": str(tmp_path / "tasks"),
            "inbox_root": str(tmp_path / "inbox"),
            "identity_cache": str(tmp_path / "cache"),
        },
        "deepfake": deepfake or {"enabled": False, "timeout_seconds": 10},
        "identity": {
            "python": sys.executable,
            "runner": str(FIXTURES / "failing_identity_runner.py"),
            "library": str(identity_library),
            "timeout_seconds": 10,
            "model_root": str(tmp_path / "models"),
        },
        "source": {
            "python": sys.executable,
            "runner": str(FIXTURES / "successful_source_runner.py"),
            "library": str(source_library),
            "top_k": 5,
            "timeout_seconds": 10,
            **(source_extra or {}),
        },
        "frontend": {"python": sys.executable},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return path


def _query(tmp_path: Path) -> Path:
    query = tmp_path / "upload_abc.jpg"
    query.write_bytes(b"lightweight fixture")
    return query


def _argv(task_dir: Path, module: str) -> list[str]:
    record = json.loads((task_dir / "logs" / f"{module}_command.json").read_text(encoding="utf-8"))
    return record["argv"]


def test_source_receives_feature_cache_directory(tmp_path: Path) -> None:
    cache = tmp_path / "用户 数据" / "cache" / "source"
    config = _config(tmp_path, source_extra={"feature_cache": str(cache)})

    status = run_task(_query(tmp_path), selected_modules=("source",), config_path=config)

    argv = _argv(Path(status["task_dir"]), "source")
    assert argv[argv.index("--feature-cache-dir") + 1] == str(cache)
    assert cache.is_dir()
    assert status["source"]["status"] == "success"


def test_source_without_cache_config_passes_no_argument(tmp_path: Path) -> None:
    config = _config(tmp_path)

    status = run_task(_query(tmp_path), selected_modules=("source",), config_path=config)

    assert "--feature-cache-dir" not in _argv(Path(status["task_dir"]), "source")


def test_unusable_cache_directory_does_not_stop_the_search(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, not a directory", encoding="utf-8")
    config = _config(tmp_path, source_extra={"feature_cache": str(blocked / "cache")})

    status = run_task(_query(tmp_path), selected_modules=("source",), config_path=config)

    task_dir = Path(status["task_dir"])
    assert status["source"]["status"] == "success"
    assert "--feature-cache-dir" not in _argv(task_dir, "source")
    events = (task_dir / "logs" / "progress_events.jsonl").read_text(encoding="utf-8")
    assert "source_feature_cache_unavailable" in events


def test_original_file_name_is_recorded_and_defaults_to_source_name(tmp_path: Path) -> None:
    config = _config(tmp_path)
    query = _query(tmp_path)

    named = run_task(query, selected_modules=("source",), config_path=config, original_file_name="现场照片.jpg")
    plain = run_task(query, selected_modules=("source",), config_path=config)

    assert named["input"]["original_file_name"] == "现场照片.jpg"
    assert plain["input"]["original_file_name"] == "upload_abc.jpg"


def _write_oneshot(tmp_path: Path) -> Path:
    runner = tmp_path / "fake_oneshot.py"
    runner.write_text(
        textwrap.dedent(
            """
            import argparse, json
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--query", required=True)
            parser.add_argument("--output", required=True)
            args, _ = parser.parse_known_args()
            Path(args.output).write_text(json.dumps({
                "status": "success", "query_image": args.query, "is_deepfake": True,
                "score": 0.5, "label": "fake", "runtime_seconds": 0.01,
                "details": {"branch": "vit_full_image"}, "error_message": None,
            }), encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    return runner


def test_deepfake_subprocess_uses_oneshot_runner_when_configured(tmp_path: Path) -> None:
    oneshot = _write_oneshot(tmp_path)
    install_root = tmp_path / "install"
    install_root.mkdir()
    config = _config(
        tmp_path,
        deepfake={
            "enabled": True,
            "python": sys.executable,
            "python_args": ["-B"],
            "runner": str(tmp_path / "frozen_adapter_not_executed.py"),
            "oneshot_runner": str(oneshot),
            "install_root": str(install_root),
            "device": "cpu",
            "timeout_seconds": 30,
        },
    )

    status = run_task(_query(tmp_path), selected_modules=("deepfake",), config_path=config)

    task_dir = Path(status["task_dir"])
    assert status["deepfake"]["status"] == "success"
    assert str(oneshot) in _argv(task_dir, "deepfake")
    events = (task_dir / "logs" / "progress_events.jsonl").read_text(encoding="utf-8")
    assert "二代鉴伪内核" in events
