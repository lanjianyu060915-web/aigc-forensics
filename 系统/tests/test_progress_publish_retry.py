"""进度快照短暂共享冲突的回归检查。"""
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import progress_events as progress


def writer_at(root):
    return progress.ProgressEventWriter(
        "任务", "identity", root / "events.jsonl", root / "latest.json",
        min_interval_seconds=0,
    )


def sharing_error():
    error = PermissionError("快照被读者临时占用")
    error.winerror = 5
    return error


def test_transient_replace_failure_preserves_single_event(tmp_path, monkeypatch):
    writer = writer_at(tmp_path)
    real_replace = os.replace
    calls = []
    sleeps = []
    def replace(source, target):
        calls.append((source, target))
        if len(calls) < 3:
            raise sharing_error()
        real_replace(source, target)
    monkeypatch.setattr(progress.os, "replace", replace)
    monkeypatch.setattr(progress.time, "sleep", sleeps.append)
    event = writer.emit("完成", "success", "完成", force=True)
    records = [json.loads(line) for line in writer.events_path.read_text("utf-8").splitlines()]
    assert records == [event]
    assert json.loads(writer.latest_path.read_text("utf-8")) == event
    assert len(calls) == 3 and sleeps == [0.01, 0.02]
    assert not list(tmp_path.glob("*.tmp"))


def test_persistent_replace_failure_is_bounded_and_visible(tmp_path, monkeypatch):
    writer = writer_at(tmp_path)
    calls = []
    sleeps = []
    def replace(source, target):
        calls.append((source, target))
        raise sharing_error()
    monkeypatch.setattr(progress.os, "replace", replace)
    monkeypatch.setattr(progress.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError):
        writer.emit("完成", "success", "完成", force=True)
    assert len(calls) == 6 and len(sleeps) == 5
    assert len(writer.events_path.read_text("utf-8").splitlines()) == 1
    assert not writer.latest_path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_other_replace_error_is_not_retried(tmp_path, monkeypatch):
    writer = writer_at(tmp_path)
    def replace(source, target):
        raise OSError("设备写入失败")
    monkeypatch.setattr(progress.os, "replace", replace)
    monkeypatch.setattr(progress.time, "sleep", lambda _: pytest.fail("不应重试其他错误"))
    with pytest.raises(OSError, match="设备写入失败"):
        writer.emit("完成", "success", "完成", force=True)


def test_initial_snapshot_read_uses_process_lock(tmp_path, monkeypatch):
    held = []
    class Lock:
        def __init__(self, path): pass
        def __enter__(self): held.append(True)
        def __exit__(self, *args): held.pop()
    def read(path):
        assert held == [True]
        return None
    monkeypatch.setattr(progress, "_CrossProcessLock", Lock)
    monkeypatch.setattr(progress, "_read_json_object", read)
    writer_at(tmp_path)
    assert held == []


@pytest.mark.skipif(os.name != "nt", reason="仅 Windows 会阻止替换共享读取的文件")
def test_windows_open_reader_releases_snapshot_on_retry(tmp_path, monkeypatch):
    writer = writer_at(tmp_path)
    writer.emit("开始", "running", "开始", force=True)
    with writer.latest_path.open("rb") as reader:
        sleeps = []
        def release_reader(delay):
            sleeps.append(delay)
            reader.close()
        monkeypatch.setattr(progress.time, "sleep", release_reader)
        event = writer.emit("完成", "success", "完成", force=True)
    assert sleeps == [0.01]
    records = [json.loads(line) for line in writer.events_path.read_text("utf-8").splitlines()]
    assert len(records) == 2 and records[-1] == event
    assert json.loads(writer.latest_path.read_text("utf-8")) == event
