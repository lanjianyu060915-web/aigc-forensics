"""New resource marker JSON and worker snapshots only; no subprocess/media."""
import json
from pathlib import Path

import pytest

from subprocess_resources_v4 import check_task_continuation


def audit(**changes):
    return {"schema_version": "subprocess_resources.v1", "status": "completed",
            "tree_cleanup": {"status": "not_requested"}, **changes}


def write(root, value, relative="logs/video_subprocess.resources.json"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), "utf-8")
    return path


def healthy():
    return {"ready": True, "health": "ok", "general_aigc": {"state": "ready", "healthy": True}}


def test_completed_normal_audit_passes_and_only_named_current_task_logs_are_read(tmp_path):
    write(tmp_path, audit())
    write(tmp_path, audit(), "frame_results/deepfake/0001/logs/deepfake_subprocess.resources.json")
    (tmp_path / "logs/unrelated.json").write_text("NOT JSON: do not open", "utf-8")
    (tmp_path / "original.mp4").write_bytes(b"not a media fixture; must not open")
    result = check_task_continuation(tmp_path, healthy(), "partial_success")
    assert result["status"] == "passed" and result["resource_audit_count"] == 2
    assert all(len(row["sha256"]) == 64 for row in result["resource_audits"])


@pytest.mark.parametrize("value,reason", [
    (audit(tree_cleanup={"status": "unknown"}), "subprocess_tree_cleanup_unknown"),
    (audit(pipe_drain_status="incomplete"), "subprocess_pipe_drain_incomplete"),
    (audit(status="timeout"), "subprocess_interruption_cleanup_unverified"),
    (audit(tree_cleanup={}), "invalid_subprocess_resource_audit"),
    ({"schema_version": "wrong"}, "invalid_subprocess_resource_audit"),
])
def test_bad_or_uncertain_cleanup_rejects(value, reason, tmp_path):
    write(tmp_path, value)
    result = check_task_continuation(tmp_path, healthy(), "success")
    assert result["status"] == "rejected"
    assert any(item.startswith(reason) for item in result["reasons"])


@pytest.mark.parametrize("raw", ['{"duplicate":1,"duplicate":2}', '{"not_finite":NaN}', 'not-json'])
def test_invalid_json_is_not_silently_accepted(tmp_path, raw):
    path = write(tmp_path, audit())
    path.write_text(raw, "utf-8")
    assert check_task_continuation(tmp_path, healthy())["status"] == "rejected"


def test_timeout_with_recorded_cleanup_and_drain_can_continue(tmp_path):
    write(tmp_path, audit(status="timeout", tree_cleanup={"status": "tree_termination_requested"}, pipe_drain_status="completed"))
    assert check_task_continuation(tmp_path, healthy(), "error")["status"] == "passed"


def test_unhealthy_worker_error_stops_even_when_task_reported_success(tmp_path):
    worker = {"ready": False, "health": "degraded", "general_aigc": {"state": "error", "healthy": False, "error": "synthetic timeout"}}
    result = check_task_continuation(tmp_path, worker, "success")
    assert result["status"] == "rejected" and result["worker_errors"][0]["worker"] == "general_aigc"
    assert result["automatic_worker_restart"] is False


def test_no_audit_is_observational_absence_not_cleanup_claim(tmp_path):
    result = check_task_continuation(tmp_path, healthy())
    assert result["status"] == "passed" and result["resource_audit_count"] == 0
    assert result["missing_audits_semantics"] == "not_observed_not_proof_of_cleanup"
