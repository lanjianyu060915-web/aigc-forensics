"""Only fake Popen, monitor, budget and native API values; no process launch."""
from pathlib import Path
from types import SimpleNamespace
import ast
import contextvars
import json
import os
import subprocess
import sys

import pytest

import resource_monitor as resources
import subprocess_resources_v4 as child


class FakeMonitor:
    instances = []

    def __init__(self, *, pid):
        self.pid = pid
        self.stopped = False
        self.instances.append(self)

    def start(self):
        return self

    def stop(self, timeout_seconds):
        self.stopped = True
        return {"memory": {"status": "available", "sampled_peak_bytes": 123},
                "gpu_memory": {"status": "unavailable", "sampled_peak_bytes": None}}


class FakeProcess:
    pid = 43210

    def __init__(self, responses):
        self.responses = list(responses)
        self.returncode = None
        self.timeouts = []
        self.killed = False

    def communicate(self, timeout):
        self.timeouts.append(timeout)
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        self.returncode = -9 if self.killed else 0
        return value

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def setup_process(monkeypatch, responses):
    process = FakeProcess(responses)
    launches = []
    def popen(*args, **kwargs):
        launches.append((args, kwargs))
        return process
    monkeypatch.setattr(child.subprocess, "Popen", popen)
    monkeypatch.setattr(child, "_WINDOWS", True)
    return process, launches


def invoke(**kwargs):
    return child.run_monitored_subprocess(["fake.exe", "--synthetic"], cwd=Path("."), timeout=5,
                                         env={"SYNTHETIC": "1"}, monitor_factory=FakeMonitor, **kwargs)


def test_completed_process_compatibility_pid_audit_and_no_shell(monkeypatch):
    process, launches = setup_process(monkeypatch, [("out", "err")])
    receipts = []
    result = invoke(audit_callback=receipts.append, hide_console=True)
    assert isinstance(result, subprocess.CompletedProcess)
    assert (result.returncode, result.stdout, result.stderr) == (0, "out", "err")
    assert result.resource_audit is receipts[0] and receipts[0]["pid"] == process.pid
    assert receipts[0]["resources"]["gpu_memory"]["sampled_peak_bytes"] is None
    options = launches[0][1]
    assert options["shell"] is False and options["creationflags"] == 0x08000000
    assert options["env"] == {"SYNTHETIC": "1"} and FakeMonitor.instances[-1].stopped
    assert 0 <= process.timeouts[0] <= 5


def test_timeout_attaches_audit_and_drains_with_bounded_cleanup(monkeypatch):
    error = subprocess.TimeoutExpired("fake", 5, output="partial")
    process, _ = setup_process(monkeypatch, [error, ("all-out", "all-err")])
    cleaned = []
    def cleanup(value, **kwargs):
        cleaned.append((value.pid, kwargs))
        value.kill()
        return {"status": "tree_termination_requested"}
    monkeypatch.setattr(child, "_terminate_owned_process_tree", cleanup)
    receipts = []
    with pytest.raises(subprocess.TimeoutExpired) as captured:
        invoke(audit_callback=receipts.append)
    assert captured.value is error and error.stdout == "all-out" and error.stderr == "all-err"
    assert error.resource_audit["status"] == "timeout"
    assert error.resource_audit["pipe_drain_status"] == "completed"
    assert cleaned == [(43210, {"cleanup_timeout": 2.0})] and process.timeouts[-1] == 2.0
    assert receipts[0] is error.resource_audit


def test_undrained_timeout_is_explicit_and_never_waits_unbounded(monkeypatch):
    process, _ = setup_process(monkeypatch, [subprocess.TimeoutExpired("fake", 5), subprocess.TimeoutExpired("fake", 2)])
    monkeypatch.setattr(child, "_terminate_owned_process_tree", lambda *a, **k: {"status": "unknown"})
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        invoke()
    audit = caught.value.resource_audit
    assert audit["tree_cleanup"]["status"] == "unknown" and audit["pipe_drain_status"] == "incomplete"
    assert process.timeouts[-1] == 2.0 and len(process.timeouts) == 2


def test_launch_failure_has_no_pid_and_still_persists_audit(monkeypatch):
    def fail(*a, **k):
        raise OSError("synthetic launch failure")
    monkeypatch.setattr(child.subprocess, "Popen", fail)
    receipts = []
    with pytest.raises(OSError) as caught:
        invoke(audit_callback=receipts.append)
    assert receipts[0]["pid"] is None and receipts[0]["status"] == "launch_failed"
    assert caught.value.resource_audit is receipts[0]


def test_monitor_unavailable_does_not_invent_zero_or_fail_inference(monkeypatch):
    setup_process(monkeypatch, [("out", "")])
    def no_monitor(**kwargs):
        raise OSError("synthetic")
    result = child.run_monitored_subprocess(["fake"], cwd=Path("."), timeout=5, env={}, monitor_factory=no_monitor)
    assert result.returncode == 0 and result.resource_audit["resources"] is None
    assert result.resource_audit["resource_monitor_error_type"] == "OSError"


def test_monitor_start_keyboard_interrupt_cleans_owned_child_with_bounded_drain(monkeypatch):
    process, _ = setup_process(monkeypatch, [("interrupted-out", "")])
    error = KeyboardInterrupt("new synthetic monitor interruption")
    monitors, cleaned, receipts = [], [], []
    class InterruptedMonitor(FakeMonitor):
        def start(self):
            monitors.append(self)
            raise error
    def cleanup(value, **kwargs):
        cleaned.append((value.pid, kwargs))
        value.kill()
        return {"status": "tree_termination_requested"}
    monkeypatch.setattr(child, "_terminate_owned_process_tree", cleanup)
    with pytest.raises(KeyboardInterrupt) as captured:
        child.run_monitored_subprocess(["fake"], cwd=Path("."), timeout=5, env={},
            monitor_factory=InterruptedMonitor, audit_callback=receipts.append)
    assert captured.value is error and monitors[0].stopped
    assert cleaned == [(43210, {"cleanup_timeout": 2.0})]
    assert process.timeouts == [2.0] and process.killed
    assert receipts[0]["status"] == "execution_interrupted"
    assert receipts[0]["pipe_drain_status"] == "completed"


def test_failed_audit_write_does_not_report_success(monkeypatch):
    setup_process(monkeypatch, [("out", "")])
    def no_disk(_):
        raise OSError("synthetic disk full")
    with pytest.raises(OSError) as caught:
        invoke(audit_callback=no_disk)
    assert caught.value.resource_audit["audit_persistence_error_type"] == "OSError"


def test_windows_tree_cleanup_targets_only_owned_live_pid(monkeypatch):
    process = FakeProcess([])
    calls = []
    monkeypatch.setattr(child, "_WINDOWS", True)
    monkeypatch.setattr(child, "_TASKKILL", SimpleNamespace(is_file=lambda: True))
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(child.subprocess, "run", run)
    result = child._terminate_owned_process_tree(process)
    assert calls[0][0][1:] == ["/PID", "43210", "/T", "/F"]
    assert calls[0][1]["shell"] is False and calls[0][1]["timeout"] == 2.0
    assert result["status"] == "tree_termination_requested" and process.killed


def test_cleanup_never_targets_self_or_exited_root(monkeypatch):
    monkeypatch.setattr(child, "_WINDOWS", True)
    monkeypatch.setattr(child.subprocess, "run", lambda *a, **k: pytest.fail("no taskkill"))
    process = FakeProcess([])
    process.pid = os.getpid()
    assert child._terminate_owned_process_tree(process)["reason"] == "invalid_owned_child_pid"
    process.pid = 43210
    process.returncode = 0
    assert child._terminate_owned_process_tree(process)["reason"] == "root_exited_descendants_unverified"


def budget():
    return {"minimum_available_memory_bytes": 100, "gpu_required": True,
            "gpu_device_index": 0, "minimum_available_gpu_bytes": 200}


def budget_observation(memory=100, used=800, total=1000):
    return {"available_memory": {"status": "available", "bytes": memory},
            "device_gpu_memory": {"status": "available", "devices": [{"device_index": 0, "bytes": used, "total_bytes": total}]}}


def test_budget_boundary_pass_is_receipt_not_reservation():
    result = resources.check_execution_budget(budget(), sampler=budget_observation)
    assert result["status"] == "passed" and result["selected_device_available_bytes"] == 200
    assert result["model_parameters_changed"] is False and result["other_processes_terminated"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("observation,reason", [
    (budget_observation(memory=99), "available_memory_below_floor"),
    (budget_observation(used=801), "available_device_memory_below_floor"),
    (budget_observation(memory=None), "available_memory_unobserved"),
    (budget_observation(used=None), "required_device_memory_unobserved"),
    (budget_observation(memory=float("nan")), "available_memory_unobserved"),
    ({}, "required_device_memory_unobserved"),
])
def test_budget_unknown_or_insufficient_rejects_with_serializable_receipt(observation, reason):
    with pytest.raises(resources.ExecutionBudgetRejected) as caught:
        resources.check_execution_budget(budget(), sampler=lambda: observation)
    assert reason in caught.value.receipt["reasons"]
    json.dumps(caught.value.receipt, allow_nan=False)


def test_budget_requires_explicit_values_and_device_identity():
    with pytest.raises(ValueError):
        resources.check_execution_budget({})
    criteria = budget()
    criteria["gpu_device_index"] = 1
    with pytest.raises(resources.ExecutionBudgetRejected) as caught:
        resources.check_execution_budget(criteria, sampler=budget_observation)
    assert caught.value.receipt["selected_device_available_bytes"] is None


def test_gpu_not_required_is_explicit_not_unknown_as_zero():
    criteria = budget()
    criteria["gpu_required"] = False
    result = resources.check_execution_budget(criteria, sampler=lambda: {"available_memory": {"status": "available", "bytes": 100}})
    assert result["status"] == "passed" and result["selected_device_available_bytes"] is None


@pytest.mark.parametrize("success", [True, False])
def test_native_system_memory_uses_fixed_abi_with_fake_dll(monkeypatch, success):
    calls = []
    class Function:
        def __call__(self, pointer):
            value = pointer._obj
            assert value.dwLength == 64
            value.ullTotalPhys = 8192
            value.ullAvailPhys = 4096
            return int(success)
    api = SimpleNamespace(GlobalMemoryStatusEx=Function())
    def dll(name, **options):
        calls.append((name, options))
        return api
    monkeypatch.setattr(resources, "_WINDOWS", True)
    monkeypatch.setattr(resources.ctypes, "WinDLL", dll, raising=False)
    result = resources._system_available_memory()
    assert calls == [("kernel32.dll", {"use_last_error": True, "winmode": 0x800})]
    assert result["bytes"] == (4096 if success else None)
    assert result["status"] == ("available" if success else "unavailable")


def test_subprocess_nonzero_exit_preserves_completed_process_semantics(monkeypatch):
    process, _ = setup_process(monkeypatch, [])
    def nonzero(timeout):
        process.returncode = 7
        return "", "synthetic exit"
    process.communicate = nonzero
    result = invoke()
    assert result.returncode == 7 and result.resource_audit["status"] == "nonzero_exit"


def test_missing_tree_cleanup_tool_marks_unknown_and_kills_only_root(monkeypatch):
    process = FakeProcess([])
    monkeypatch.setattr(child, "_WINDOWS", True)
    monkeypatch.setattr(child, "_TASKKILL", SimpleNamespace(is_file=lambda: False))
    monkeypatch.setattr(child.subprocess, "run", lambda *a, **k: pytest.fail("no command search"))
    result = child._terminate_owned_process_tree(process)
    assert result["status"] == "unknown" and process.killed


def test_orchestrator_adapter_passes_isolated_env_and_saves_audit(monkeypatch, tmp_path):
    # Compile only this function: no orchestrator imports or historical assets.
    source = Path(__file__).resolve().parents[1] / "orchestrator.py"
    tree = ast.parse(source.read_text("utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_subprocess")
    namespace = {"__name__": "synthetic_adapter", "Path": Path, "subprocess": subprocess, "os": os, "sys": sys,
                 "Mapping": dict, "_TASK_ENVIRONMENT": contextvars.ContextVar("synthetic", default={"ISOLATED": "1"})}
    records = []
    namespace["atomic_write_json"] = lambda path, value, **kwargs: records.append((path, value, kwargs))
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    def fake_run(command, **kwargs):
        assert kwargs["env"] == {"ISOLATED": "1", "EXTRA": "2"}
        kwargs["audit_callback"]({"pid": 123, "status": "completed"})
        return subprocess.CompletedProcess(command, 0, "ok", "")
    monkeypatch.setattr(child, "run_monitored_subprocess", fake_run)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    path = tmp_path / "synthetic.resources.json"
    result = namespace["_run_subprocess"](["fake"], cwd=tmp_path, timeout=3, hide_console=True,
        executable_worker=False, extra_environment={"EXTRA": "2"}, resource_audit_path=path)
    assert result.stdout == "ok" and records[0][0] == path and records[0][2]["overwrite"] is False
