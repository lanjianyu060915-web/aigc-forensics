"""Synthetic subprocess protocol and fake resource samplers only."""
import io
import json
import os
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime_services as services
import runtime_progress as progress
from frontend.components.runtime_status import runtime_status_rows, startup_needs_refresh

FIXTURE = Path(__file__).with_name("v4_fake_startup_worker.py")


class FakeMonitor:
    def __init__(self, *, pid):
        self.pid = pid
    def start(self):
        return self
    def summary(self):
        return {"sample_count": 1, "sampling_interval_seconds": 0.25, "exact_peak": False,
                "memory": {"status": "available", "sampled_peak_bytes": 1024},
                "gpu_memory": {"status": "unavailable", "sampled_peak_bytes": None}}
    def stop(self, timeout_seconds=0.5):
        return self.summary()


@pytest.fixture(autouse=True)
def no_real_resource_queries(monkeypatch):
    monkeypatch.setattr(services, "ResourceMonitor", FakeMonitor)


def channel(mode="ready", timeout=5, callback=None):
    return services._WorkerChannel(name="general_aigc", command=[sys.executable, "-u", str(FIXTURE), "--mode", mode],
                                   cwd=FIXTURE.parent, environment=dict(os.environ), startup_timeout=timeout,
                                   progress_callback=callback)


def test_real_protocol_progress_reaches_snapshot_without_request_id():
    events = []
    worker = channel(callback=events.append)
    try:
        worker.start()
        result = worker.snapshot()
        stages = [item["stage"] for item in result["startup_history"]]
        assert stages[:2] == ["creating_process", "process_created"]
        assert stages[2:] == ["importing", "loading_model", "checking_library", "rebuilding_index", "warming_up", "ready"]
        assert result["warmup_status"] == "not_performed"
        assert result["noise_lines_ignored"] == 0
        assert result["startup_elapsed_seconds"] >= 0 and result["ready"]
        assert result["startup_resources"]["exact_peak"] is False
        assert worker.request("health", {}, 1)["ok"] is True
        assert any(e.get("stage") == "rebuilding_index" for e in events)
    finally:
        worker.stop()


def test_failed_startup_keeps_true_error_and_resource_record():
    worker = channel("fail")
    try:
        with pytest.raises(services.RuntimeServiceUnavailable, match="synthetic load failure"):
            worker.start()
        result = worker.snapshot()
        assert result["stage"] == "failed" and result["ready"] is False
        assert "synthetic load failure" in result["error"]
        assert result["startup_resources"]["sample_count"] == 1
    finally:
        worker.stop()


def test_permanent_loading_times_out_and_does_not_claim_readiness():
    worker = channel("hang", timeout=0.2)
    try:
        with pytest.raises(services.RuntimeServiceTimeout):
            worker.start()
        result = worker.snapshot()
        assert result["stage"] == "failed" and not result["ready"]
        assert not startup_needs_refresh({"general_aigc": result, "health": "error"})
        assert result["warmup_status"] == "not_reported"
    finally:
        worker.stop()


def test_explicit_channel_restart_gets_new_attempt_and_keeps_history():
    worker = channel()
    try:
        worker.start()
        old = worker.snapshot()
        worker.stop()
        worker.start()
        new = worker.snapshot()
        assert new["startup_attempt_id"] != old["startup_attempt_id"]
        assert new["restart_count"] == 1 and new["ready"]
        assert any(e["stage"] == "restarting" for e in new["startup_history"])
    finally:
        worker.stop()


def test_progress_missing_key_is_accepted_but_does_not_complete_handshake():
    worker = channel()
    worker._state = "starting"
    worker._startup_started = 0
    event = {"protocol": services.PROTOCOL_NAME, "type": "startup_progress", "ready": False,
             "health": "starting", "stage": "loading_model", "stage_status": "running"}
    assert worker._is_protocol_message(event)
    worker._dispatch_message(event)
    assert not worker._startup_event.is_set() and not worker.snapshot()["ready"]
    worker._dispatch_message({**event, "stage": "invented_stage"})
    assert worker.snapshot()["noise_lines_ignored"] == 1


def test_manager_truthfully_reports_pending_services_and_explicit_restart(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    manager = services.RuntimeServiceManager(config)
    def spec(name):
        if name != "deepfake":
            manager._static_status[name] = manager._status_record("disabled")
            return None
        return [sys.executable, "-u", str(FIXTURE)], FIXTURE.parent, dict(os.environ), 5
    manager._worker_spec = spec
    during = []
    manager.progress_callback = lambda event: during.append(manager.get_status()) if event.get("stage") == "creating_process" else None
    try:
        result = manager.start()
        assert during[0]["identity"]["state"] == "pending"
        assert during[0]["health"] == "starting"
        assert result["health"] == "ok"
        again = manager.restart_service("deepfake")
        assert again["deepfake"]["restart_count"] == 1
        with pytest.raises(services.RuntimeServiceUnavailable):
            manager.restart_service("source")
    finally:
        manager.close()


def test_startup_reporter_emits_counters_without_library_paths(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(progress, "_output", output)
    progress.StartupIndexReporter().emit("build_identity_cache", "running", "synthetic index",
                                         details={"library": "private path", "identity_id": "private label", "library_image_count": 3},
                                         progress_current=1, progress_total=3)
    value = json.loads(output.getvalue().strip())
    assert value["stage"] == "rebuilding_index"
    assert value["details"] == {"library_image_count": 3, "progress_current": 1, "progress_total": 3}
    assert "private" not in output.getvalue()


def test_ui_rows_do_not_invent_source_worker_or_warmup():
    rows = runtime_status_rows({"general_aigc": {"stage": "warming_up", "stage_status": "skipped",
                                                "startup_elapsed_seconds": 3}},
                               source_cache={"state": "cache_ready", "message": "cache checked"})
    general = next(row for row in rows if row["service"] == "general_aigc")
    source = next(row for row in rows if row["service"] == "source")
    assert general["stage_label"] == "未单独执行预热"
    assert source["kind"] == "reference_cache" and source["pid"] is None
    assert source["warmup_status"] == "not_applicable"
    assert not startup_needs_refresh({"health": "ok", "general_aigc": {"state": "ready"}})
    assert startup_needs_refresh({"health": "starting"})


def test_unknown_profile_is_rejected_before_worker_creation(tmp_path):
    config = tmp_path / "unknown.json"
    config.write_text(json.dumps({"execution_profile": {"schema_version": "unknown"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported_execution_profile_schema"):
        services.RuntimeServiceManager(config)


def test_independent_startup_gate_failure_cannot_create_process(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"execution_profile": {"schema_version": "execution_profile.independent.v1", "mode": "engineering_only"}}), encoding="utf-8")
    calls = []
    monkeypatch.setattr(services, "trusted_policy_authority", lambda: object())
    def audit(config, **kwargs):
        calls.append(kwargs.get("for_inference", False))
        if kwargs.get("for_inference"):
            raise ValueError("explicit_inference_profile_required")
        return {}
    monkeypatch.setattr(services, "audit_execution_profile", audit)
    monkeypatch.setattr(services.subprocess, "Popen", lambda *a, **k: pytest.fail("must not create model process"))
    manager = services.RuntimeServiceManager(config)
    try:
        with pytest.raises(ValueError, match="explicit_inference_profile_required"):
            manager.start()
        assert calls == [False, True]
        assert not manager._started
    finally:
        manager.close()


def test_request_failure_retains_its_own_resource_measurement():
    worker = channel()
    try:
        worker.start()
        with pytest.raises(services.RuntimeServiceRemoteError) as caught:
            worker.request("reject", {}, 2)
        measurement = caught.value.runtime_measurement
        assert measurement["status"] == "failed" and measurement["request_id"]
        assert measurement["pid"] == worker.snapshot()["pid"]
        assert measurement["sampling_scope"] == "request_lock_held_execution"
        assert measurement["resources"]["sample_count"] == 1
    finally:
        worker.stop()


def test_request_queue_is_excluded_from_sampling_and_responses_keep_own_records(monkeypatch):
    worker = channel()
    try:
        worker.start()
        first_sampling = threading.Event()
        samples = []
        class ExecutionMonitor(FakeMonitor):
            def __init__(self, *, pid):
                assert worker._request_lock._is_owned()
                super().__init__(pid=pid)
                self.marker = len(samples) + 1
                samples.append(self.marker)
            def start(self):
                first_sampling.set()
                return self
            def summary(self):
                result = super().summary()
                result["memory"]["sampled_peak_bytes"] = self.marker
                return result
        monkeypatch.setattr(services, "ResourceMonitor", ExecutionMonitor)
        responses = {}
        def run_first():
            responses["first"] = worker.request("wait", {}, 2)
        thread = threading.Thread(target=run_first)
        thread.start()
        assert first_sampling.wait(1)
        responses["second"] = worker.request("health", {}, 2)
        thread.join(1)
        assert not thread.is_alive()
        first = services.RuntimeServiceManager._payload_from_response("general_aigc", responses["first"])["runtime_service_measurement"]
        second = services.RuntimeServiceManager._payload_from_response("general_aigc", responses["second"])["runtime_service_measurement"]
        assert first["request_id"] != second["request_id"]
        assert second["queue_seconds"] > 0.01
        assert first["resources"]["memory"]["sampled_peak_bytes"] == 1
        assert second["resources"]["memory"]["sampled_peak_bytes"] == 2
        assert first["sampling_stopped_before_unlock"] and second["sampling_stopped_before_unlock"]
    finally:
        worker.stop()
