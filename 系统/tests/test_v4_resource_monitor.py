"""Synthetic samplers, clocks and queue reservations; no GPU/model access."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resource_monitor import ProcessResourceSampler, ResourceBudgetExceeded, ResourceBudgetQueue, ResourceMonitor
import resource_monitor


class FakeCFunction:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


def fake_windows_api(monkeypatch, *, open_result=321, info_result=1, raises=False):
    """No native DLL/process calls: inspect ctypes arguments and fill memory."""
    calls = {"opened": [], "closed": [], "dlls": [], "sizes": []}
    def open_process(*args):
        calls["opened"].append(args)
        return open_result
    def memory_info(handle, pointer, size):
        assert handle == 321
        calls["sizes"].append(size)
        if raises:
            raise OSError("synthetic")
        counters = pointer._obj
        counters.WorkingSetSize = 1234
        counters.PeakWorkingSetSize = 5678
        return info_result
    kernel = SimpleNamespace(OpenProcess=FakeCFunction(open_process), CloseHandle=FakeCFunction(lambda h: calls["closed"].append(h)))
    psapi = SimpleNamespace(GetProcessMemoryInfo=FakeCFunction(memory_info))
    def dll(name, **kwargs):
        calls["dlls"].append((name, kwargs))
        return kernel if name == "kernel32.dll" else psapi
    monkeypatch.setattr(resource_monitor, "_WINDOWS", True)
    monkeypatch.setattr(resource_monitor.ctypes, "WinDLL", dll, raising=False)
    return calls


def test_windows_native_memory_abi_and_handle_lifetime(monkeypatch):
    calls = fake_windows_api(monkeypatch)
    value = resource_monitor._windows_process_memory(123)
    assert value["bytes"] == 1234 and value["process_lifetime_peak_working_set_bytes"] == 5678
    assert value["peak_scope"] == "process_lifetime_not_monitor_or_task_interval"
    assert value["scope"] == "single_pid_excludes_children"
    assert calls["opened"] == [(0x1000, False, 123)] and calls["closed"] == [321]
    assert all(options == {"use_last_error": True, "winmode": 0x800} for _, options in calls["dlls"])
    struct = resource_monitor._ProcessMemoryCounters
    assert struct.PeakWorkingSetSize.offset == 8
    assert calls["sizes"] == [resource_monitor.ctypes.sizeof(struct)]


@pytest.mark.parametrize("mode,expected", [("open", "windows_open_process_failed"),
                                         ("info", "windows_get_process_memory_info_failed"),
                                         ("raise", "windows_native_memory_OSError")])
def test_windows_native_failures_are_unknown_and_close_open_handles(monkeypatch, mode, expected):
    calls = fake_windows_api(monkeypatch, open_result=0 if mode == "open" else 321,
                             info_result=0 if mode == "info" else 1, raises=mode == "raise")
    result = resource_monitor._windows_process_memory(123)
    assert result == {"status": "unavailable", "bytes": None, "reason": expected}
    assert calls["closed"] == ([] if mode == "open" else [321])


def test_psutil_absence_falls_back_to_native_for_requested_pid(monkeypatch):
    calls = fake_windows_api(monkeypatch)
    monkeypatch.setitem(sys.modules, "psutil", None)
    result = ProcessResourceSampler(456)._memory()
    assert result["bytes"] == 1234 and result["fallback_reason"] == "psutil_not_installed"
    assert calls["opened"] == [(0x1000, False, 456)]


def test_psutil_error_falls_back_to_native(monkeypatch):
    fake_windows_api(monkeypatch)
    def inaccessible(_pid):
        raise PermissionError("synthetic")
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=inaccessible))
    result = ProcessResourceSampler(123)._memory()
    assert result["bytes"] == 1234 and result["fallback_reason"] == "PermissionError"


def test_psutil_available_has_priority_and_carries_lifetime_peak(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda _pid: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=10, peak_wset=20))))
    monkeypatch.setattr(resource_monitor, "_windows_process_memory", lambda _pid: pytest.fail("no native call"))
    result = ProcessResourceSampler(123)._memory()
    assert result["source"] == "psutil.rss" and result["process_lifetime_peak_working_set_bytes"] == 20


def test_non_windows_missing_psutil_remains_unavailable(monkeypatch):
    monkeypatch.setattr(resource_monitor, "_WINDOWS", False)
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert ProcessResourceSampler(123)._memory()["reason"] == "psutil_not_installed"


def fake_smi(monkeypatch, output="0, 2, 10\n1, 3, 20\n", returncode=0, raises=None):
    calls = []
    path = SimpleNamespace(is_file=lambda: True)
    monkeypatch.setattr(resource_monitor, "_WINDOWS", True)
    monkeypatch.setattr(resource_monitor, "_NVIDIA_SMI_PATHS", (path,))
    def run(args, **kwargs):
        calls.append((args, kwargs))
        if raises:
            raise raises
        return SimpleNamespace(returncode=returncode, stdout=output)
    monkeypatch.setattr(resource_monitor.subprocess, "run", run)
    return calls, path


def test_device_budget_smi_is_separate_fixed_no_shell_and_cached(monkeypatch):
    calls, path = fake_smi(monkeypatch)
    tick = [0.0]
    sampler = resource_monitor.DeviceGPUBudgetSampler(clock=lambda: tick[0])
    result = sampler()
    assert result["bytes"] == 5 * 1024**2 and result["total_bytes"] == 30 * 1024**2
    assert result["attributable_to_monitored_pid"] is False and result["purpose"] == "budget_observation_only"
    args, kwargs = calls[0]
    assert args == [str(path), "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"]
    assert kwargs["shell"] is False and kwargs["timeout"] == 1.0 and kwargs["creationflags"] == 0x08000000
    tick[0] = 0.4
    assert sampler()["cached"] and len(calls) == 1
    tick[0] = 1.1
    assert not sampler()["cached"] and len(calls) == 2


@pytest.mark.parametrize("output", ["0, N/A, 100\n", "0, 101, 100\n", "0, -1, 100\n", "0, 1, 2\n0, 1, 2\n", "", "not-csv"])
def test_invalid_smi_readings_are_unknown_not_zero(monkeypatch, output):
    fake_smi(monkeypatch, output=output)
    result = resource_monitor.DeviceGPUBudgetSampler()()
    assert result["status"] == "unavailable" and result["bytes"] is None
    assert result["attributable_to_monitored_pid"] is False


def test_smi_timeout_and_missing_fixed_executable_are_unavailable(monkeypatch):
    fake_smi(monkeypatch, raises=resource_monitor.subprocess.TimeoutExpired("synthetic", 1))
    assert resource_monitor.DeviceGPUBudgetSampler()()["reason"] == "nvidia_smi_timeout"
    monkeypatch.setattr(resource_monitor, "_NVIDIA_SMI_PATHS", (SimpleNamespace(is_file=lambda: False),))
    monkeypatch.setattr(resource_monitor.subprocess, "run", lambda *_a, **_kw: pytest.fail("must not search PATH"))
    assert resource_monitor.DeviceGPUBudgetSampler()()["reason"] == "fixed_nvidia_smi_executable_unavailable"


def test_device_budget_never_fills_unavailable_per_pid_gpu(monkeypatch):
    sampler = ProcessResourceSampler(123)
    monkeypatch.setattr(sampler, "_memory", lambda: {"status": "available", "bytes": 20})
    monkeypatch.setattr(sampler, "_gpu", lambda: resource_monitor._unavailable("nvml_unavailable"))
    monkeypatch.setattr(sampler, "_device_budget", lambda: {"status": "available", "bytes": 100})
    result = sampler()
    assert result["gpu_memory"]["bytes"] is None and result["device_gpu_memory"]["bytes"] == 100


def test_sampled_and_process_lifetime_peaks_are_not_conflated():
    sample = observation(10)
    sample["memory"].update({"process_lifetime_peak_working_set_bytes": 900,
                             "peak_scope": "process_lifetime_not_monitor_or_task_interval", "source": "synthetic.native"})
    sample["device_gpu_memory"] = {"status": "available", "bytes": 100}
    monitor = ResourceMonitor(sampler=lambda: sample)
    monitor.sample_once()
    summary = monitor.summary()
    assert summary["memory"]["sampled_peak_bytes"] == 10
    assert summary["memory"]["process_lifetime_peak_working_set_bytes"] == 900
    assert summary["memory"]["sources"] == ["synthetic.native"]
    assert summary["gpu_memory"]["sampled_peak_bytes"] is None
    assert summary["device_gpu_memory"]["sampled_peak_bytes"] == 100
    assert summary["device_gpu_memory"]["attributable_to_monitored_pid"] is False


def observation(memory, gpu=None):
    def item(value):
        return {"status": "unavailable", "bytes": None, "reason": "synthetic_missing"} if value is None else {"status": "available", "bytes": value}
    return {"pid": 123, "memory": item(memory), "gpu_memory": item(gpu)}


def test_sampled_peak_and_missing_data_are_distinct():
    rows = iter([observation(10, 2), observation(30), observation(20, 5)])
    ticks = iter([0, 1, 2, 3, 4])
    monitor = ResourceMonitor(sampler=lambda: next(rows), interval_seconds=0.5, clock=lambda: next(ticks))
    for _ in range(3):
        monitor.sample_once()
    summary = monitor.summary()
    assert summary["memory"]["sampled_peak_bytes"] == 30
    assert summary["gpu_memory"]["sampled_peak_bytes"] == 5
    assert summary["gpu_memory"]["status"] == "partial"
    assert summary["gpu_memory"]["unavailable_samples"] == 1
    assert summary["sample_count"] == 3 and summary["exact_peak"] is False
    assert summary["sampling_interval_seconds"] == 0.5


def test_failed_sampler_is_not_reported_as_zero_consumption():
    def fail():
        raise OSError("synthetic sample failure")
    monitor = ResourceMonitor(sampler=fail)
    monitor.sample_once()
    summary = monitor.summary()
    assert summary["sampler_failures"] == 1
    assert summary["memory"]["sampled_peak_bytes"] is None
    assert summary["gpu_memory"]["status"] == "unavailable"


def test_monitor_stops_without_waiting_for_long_interval():
    monitor = ResourceMonitor(sampler=lambda: observation(1), interval_seconds=30).start()
    summary = monitor.stop(0.5)
    assert summary["sample_count"] == 1 and not summary["monitor_running"]
    with pytest.raises(RuntimeError):
        monitor.start()


def test_queue_reservation_released_after_worker_failure():
    queue = ResourceBudgetQueue(memory_budget_bytes=100, gpu_budget_bytes=10)
    with pytest.raises(RuntimeError):
        with queue.reserve("synthetic", estimated_memory_bytes=60, estimated_gpu_bytes=5) as lease:
            assert queue.snapshot()["active"] == 1
            raise RuntimeError("synthetic worker failure")
    assert queue.snapshot()["active"] == 0
    assert lease.released_at is not None and lease.to_dict()["execution_seconds"] >= 0


def test_oversize_request_fails_without_altering_input_or_reservations():
    queue = ResourceBudgetQueue(gpu_budget_bytes=10)
    with pytest.raises(ResourceBudgetExceeded):
        queue.acquire("synthetic", estimated_gpu_bytes=11)
    assert queue.snapshot()["active"] == queue.snapshot()["waiting"] == 0


def test_queue_timeout_removes_ticket_and_later_request_can_run():
    queue = ResourceBudgetQueue()
    first = queue.acquire("first")
    with pytest.raises(TimeoutError):
        queue.acquire("timeout", timeout_seconds=0)
    assert queue.snapshot()["waiting"] == 0
    queue.release(first)
    with queue.reserve("later", timeout_seconds=0):
        assert queue.snapshot()["active"] == 1


def test_queue_serializes_workers_and_measures_wait():
    queue = ResourceBudgetQueue()
    first = queue.acquire("first")
    started, acquired = threading.Event(), threading.Event()
    results = []
    def worker():
        started.set()
        with queue.reserve("second", timeout_seconds=2) as lease:
            results.append(lease)
            acquired.set()
    thread = threading.Thread(target=worker)
    thread.start()
    assert started.wait(1)
    assert not acquired.is_set()
    queue.release(first)
    assert acquired.wait(1)
    thread.join(1)
    assert not thread.is_alive() and results[0].to_dict()["queue_seconds"] >= 0
    assert queue.snapshot()["active"] == queue.snapshot()["waiting"] == 0


def test_budget_reservations_can_limit_multiple_active_slots():
    queue = ResourceBudgetQueue(max_active=2, memory_budget_bytes=100)
    first = queue.acquire("first", estimated_memory_bytes=80)
    with pytest.raises(TimeoutError):
        queue.acquire("second", estimated_memory_bytes=30, timeout_seconds=0)
    queue.release(first)
    with queue.reserve("second", estimated_memory_bytes=30):
        assert queue.snapshot()["reserved_memory_bytes"] == 30


def test_queue_rejects_double_release():
    queue = ResourceBudgetQueue()
    lease = queue.acquire("one")
    queue.release(lease)
    with pytest.raises(ValueError):
        queue.release(lease)


def test_nvml_duplicate_process_rows_are_not_double_counted(monkeypatch):
    closed = []
    fake = SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: closed.append(True),
                           nvmlDeviceGetCount=lambda: 1, nvmlDeviceGetHandleByIndex=lambda i: i,
                           nvmlDeviceGetComputeRunningProcesses=lambda _: [SimpleNamespace(pid=123, usedGpuMemory=20)],
                           nvmlDeviceGetGraphicsRunningProcesses=lambda _: [SimpleNamespace(pid=123, usedGpuMemory=20)])
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    sampler = ProcessResourceSampler(123)
    result = sampler._gpu()
    assert result["status"] == "available" and result["bytes"] == 20
    sampler.close()
    assert closed == [True]


def test_wddm_unknown_memory_is_unavailable_not_zero(monkeypatch):
    fake = SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
                           nvmlDeviceGetCount=lambda: 1, nvmlDeviceGetHandleByIndex=lambda i: i,
                           nvmlDeviceGetComputeRunningProcesses=lambda _: [SimpleNamespace(pid=123, usedGpuMemory=(1 << 64) - 1)],
                           nvmlDeviceGetGraphicsRunningProcesses=lambda _: [])
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    sampler = ProcessResourceSampler(123)
    result = sampler._gpu()
    assert result["status"] == "unavailable" and result["bytes"] is None
    sampler.close()


def test_process_memory_uses_requested_pid_with_synthetic_psutil(monkeypatch):
    pids = []
    def process(pid):
        pids.append(pid)
        return SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1234))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=process))
    sampler = ProcessResourceSampler(123)
    assert sampler._memory()["bytes"] == 1234 and pids == [123]


def test_owned_sampler_eventually_closes_after_bounded_stop_timeout(monkeypatch):
    sampling, unblock, closed = threading.Event(), threading.Event(), threading.Event()
    class BlockingSampler:
        def __init__(self, pid=None):
            self.calls = 0
        def __call__(self):
            self.calls += 1
            if self.calls > 1:
                sampling.set()
                assert unblock.wait(2)
            return observation(1)
        def close(self):
            closed.set()
    monkeypatch.setattr(resource_monitor, "ProcessResourceSampler", BlockingSampler)
    monitor = ResourceMonitor(interval_seconds=0.001).start()
    try:
        assert sampling.wait(1)
        result = monitor.stop(timeout_seconds=0)
        assert result["monitor_running"] and not closed.is_set()
    finally:
        unblock.set()
    assert closed.wait(1)
    monitor._thread.join(1)
    assert not monitor.summary()["monitor_running"]
