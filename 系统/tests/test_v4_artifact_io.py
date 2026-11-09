"""Independent synthetic files only; no product models or team fixtures."""
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import artifact_io as aio


def sharing_error(code=32):
    error = PermissionError("synthetic sharing violation")
    error.winerror = code
    return error


def test_publish_failure_preserves_complete_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    path.write_text('{"previous":true}', encoding="utf-8")
    attempts, sleeps, audit = [], [], []
    def fail(*args):
        attempts.append(args)
        raise sharing_error()
    monkeypatch.setattr(aio.os, "replace", fail)
    monkeypatch.setattr(aio.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError) as caught:
        aio.atomic_write_json(path, {"next": True}, role="synthetic_status", audit=audit)
    assert json.loads(path.read_text("utf-8")) == {"previous": True}
    assert len(attempts) == 6 and sleeps == [0.01, 0.02, 0.04, 0.08, 0.16]
    assert caught.value.artifact_io_record is audit[-1]
    assert audit[-1]["status"] == "failed" and audit[-1]["attempts"] == 6
    assert not list(tmp_path.glob("*.tmp"))


def test_transient_failure_records_recovery_once(tmp_path, monkeypatch):
    real = os.replace
    calls, audit = [], []
    def replace(source, target):
        calls.append(1)
        if len(calls) < 3:
            raise sharing_error(5)
        real(source, target)
    monkeypatch.setattr(aio.os, "replace", replace)
    monkeypatch.setattr(aio.time, "sleep", lambda _: None)
    path = tmp_path / "status.json"
    record = aio.atomic_write_json(path, {"synthetic": "你好"}, audit=audit)
    assert len(audit) == 1 and record["attempts"] == 3
    assert len(record["failures"]) == 2 and len(record["sha256"]) == 64
    assert json.loads(path.read_text("utf-8")) == {"synthetic": "你好"}


def test_nonsharing_error_has_no_retry_and_no_partial_destination(tmp_path, monkeypatch):
    def fail(*_):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(aio.os, "replace", fail)
    monkeypatch.setattr(aio.time, "sleep", lambda _: pytest.fail("unexpected retry"))
    with pytest.raises(OSError) as caught:
        aio.atomic_write_text(tmp_path / "new.txt", "complete")
    assert caught.value.artifact_io_record["attempts"] == 1
    assert not (tmp_path / "new.txt").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_fsync_failure_leaves_old_artifact(tmp_path, monkeypatch):
    path = tmp_path / "old.txt"
    path.write_text("old", encoding="utf-8")
    def fail(*_):
        raise OSError("synthetic disk full")
    monkeypatch.setattr(aio.os, "fsync", fail)
    audit = []
    with pytest.raises(OSError):
        aio.atomic_write_text(path, "new", audit=audit)
    assert path.read_text("utf-8") == "old"
    assert audit[-1]["phase"] == "write_temporary"
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_json_does_not_create_directory(tmp_path):
    path = tmp_path / "not_created" / "bad.json"
    audit = []
    with pytest.raises(ValueError):
        aio.atomic_write_json(path, {"score": float("nan")}, audit=audit)
    assert not path.parent.exists()
    assert audit[0]["phase"] == "serialize" and audit[0]["status"] == "failed"


def test_stage_receipt_is_immutable_and_keeps_attempt_identity(tmp_path):
    args = dict(task_id="synthetic-task", attempt_id="attempt-1", stage="report", status="completed",
                input_digest="a" * 64, config_digest="b" * 64)
    path = aio.write_stage_receipt(tmp_path, **args)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        aio.write_stage_receipt(tmp_path, **{**args, "status": "failed"})
    assert path.read_bytes() == before
    new = aio.write_stage_receipt(tmp_path, **{**args, "attempt_id": "attempt-2"})
    assert new != path and json.loads(new.read_text("utf-8"))["attempt_id"] == "attempt-2"


def test_report_repair_never_requests_inference_or_cloud():
    plan = aio.plan_recovery("render_report", verified_stages={"inference", "decision"},
                             bindings_match=True, authorized=True)
    assert plan.allowed and not plan.run_model and not plan.call_cloud
    missing = aio.plan_recovery("render_report", verified_stages=set(), bindings_match=True, authorized=True)
    assert not missing.allowed and not missing.run_model


@pytest.mark.parametrize("kwargs,reason", [
    ({"authorized": False}, "current_authorization_required"),
    ({"stop_requested": True}, "stop_requested"),
    ({"bindings_match": False}, "input_or_config_binding_mismatch"),
])
def test_recovery_requires_current_authorization_and_same_bindings(kwargs, reason):
    args = dict(verified_stages={"decision"}, authorized=True, bindings_match=True)
    plan = aio.plan_recovery("render_report", **{**args, **kwargs})
    assert not plan.allowed and plan.reason == reason


def test_verified_inference_is_not_rerun_for_missing_report():
    plan = aio.plan_recovery("rerun_inference", verified_stages={"inference"}, bindings_match=True, authorized=True)
    assert not plan.allowed
    explicit = aio.plan_recovery("rerun_inference", verified_stages=set(), bindings_match=True, authorized=True)
    assert explicit.allowed and explicit.run_model


def test_cloud_retry_is_distinct_and_bounded():
    args = dict(verified_stages={"decision"}, bindings_match=True, authorized=True)
    plan = aio.plan_recovery("retry_cloud", cloud_attempts=1, max_cloud_attempts=2, **args)
    assert plan.call_cloud and not plan.run_model
    denied = aio.plan_recovery("retry_cloud", cloud_attempts=2, max_cloud_attempts=2, **args)
    assert not denied.allowed
    complete = aio.plan_recovery("write_receipt", **{**args, "verified_stages": {"decision", "receipt"}})
    assert not complete.allowed


def test_receipt_rejects_path_traversal_and_recovery_rejects_truthy_flags(tmp_path):
    with pytest.raises(ValueError):
        aio.write_stage_receipt(tmp_path, task_id="../outside", attempt_id="one", stage="report",
                                status="completed", input_digest="a" * 64, config_digest="b" * 64)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        aio.plan_recovery("render_report", verified_stages={"decision"}, bindings_match=True, authorized="false")
    plan = aio.plan_recovery("rerun_inference", verified_stages={"decision"}, bindings_match=True, authorized=True)
    assert not plan.allowed and not plan.run_model
