"""Sampled resource observations and an explicit in-process development queue.

No model imports, input decoding, resource-dependent resizing or model changes.
Queue budgets are reservations from caller estimates, not claims of free system
memory or cross-process GPU exclusivity. A coordinator must share one queue.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable
import uuid


_WINDOWS = os.name == "nt"
_NVIDIA_SMI_PATHS = (
    Path("C:/Windows/System32/nvidia-smi.exe"),
    Path("C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe"),
)


class _ProcessMemoryCounters(ctypes.Structure):
    # Windows DWORD is always 32-bit; SIZE_T follows the process pointer width.
    _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def _windows_process_memory(pid: int) -> dict:
    """Read one PID, excluding child processes, without optional dependencies."""
    if not _WINDOWS:
        return _unavailable("windows_native_memory_not_supported")
    try:
        # Restrict DLL resolution to the Windows system directory.
        kernel = ctypes.WinDLL("kernel32.dll", use_last_error=True, winmode=0x800)
        psapi = ctypes.WinDLL("psapi.dll", use_last_error=True, winmode=0x800)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.restype = ctypes.c_int
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters), ctypes.c_uint32]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return _unavailable("windows_open_process_failed")
        try:
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return _unavailable("windows_get_process_memory_info_failed")
            return {"status": "available", "bytes": int(counters.WorkingSetSize),
                    "source": "windows.GetProcessMemoryInfo.WorkingSetSize", "scope": "single_pid_excludes_children",
                    "process_lifetime_peak_working_set_bytes": int(counters.PeakWorkingSetSize),
                    "peak_scope": "process_lifetime_not_monitor_or_task_interval"}
        finally:
            kernel.CloseHandle(handle)
    except Exception as exc:
        return _unavailable("windows_native_memory_" + type(exc).__name__)


class DeviceGPUBudgetSampler:
    """Separate device-wide observations, never attributed to the monitored PID.

    Fixed executable locations, fixed arguments, no shell/PATH search. Readings
    include other applications and display allocations; they are budget context,
    not per-task GPU accounting. Cache for one second to bound probe overhead.
    """

    def __init__(self, *, clock: Callable = time.perf_counter):
        self.clock = clock
        self._last_at = None
        self._last = None

    def __call__(self) -> dict:
        now = self.clock()
        if self._last_at is not None and now - self._last_at < 1.0:
            return {**self._last, "cached": True, "sample_age_seconds": max(0.0, now - self._last_at)}
        result = self._read()
        result.update({"scope": "device_aggregate_includes_other_processes", "purpose": "budget_observation_only",
                       "attributable_to_monitored_pid": False, "cached": False, "sample_age_seconds": 0.0})
        self._last, self._last_at = result, self.clock()
        return dict(result)

    def _read(self) -> dict:
        if not _WINDOWS:
            return _unavailable("fixed_windows_nvidia_smi_not_supported")
        try:
            executable = next((path for path in _NVIDIA_SMI_PATHS if path.is_file()), None)
            if executable is None:
                return _unavailable("fixed_nvidia_smi_executable_unavailable")
            completed = subprocess.run([str(executable), "--query-gpu=index,memory.used,memory.total",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                       encoding="ascii", errors="strict", timeout=1.0, check=False,
                                       shell=False, creationflags=0x08000000)
            if completed.returncode != 0:
                return _unavailable("nvidia_smi_failed")
            if len(completed.stdout) > 65536:
                return _unavailable("nvidia_smi_output_exceeds_budget")
            devices, seen = [], set()
            for line in completed.stdout.splitlines():
                cells = [part.strip() for part in line.split(",")]
                if len(cells) != 3 or not all(cell.isascii() and cell.isdigit() for cell in cells):
                    return _unavailable("nvidia_smi_invalid_device_memory")
                index, used, total = map(int, cells)
                if index in seen or total <= 0 or used > total:
                    return _unavailable("nvidia_smi_invalid_device_memory")
                seen.add(index)
                devices.append({"device_index": index, "bytes": used * 1024**2, "total_bytes": total * 1024**2})
            if not devices:
                return _unavailable("nvidia_smi_no_devices")
            return {"status": "available", "bytes": sum(row["bytes"] for row in devices),
                    "total_bytes": sum(row["total_bytes"] for row in devices), "devices": devices,
                    "source": "nvidia-smi.device_memory", "executable": str(executable)}
        except subprocess.TimeoutExpired:
            return _unavailable("nvidia_smi_timeout")
        except Exception as exc:
            return _unavailable("nvidia_smi_" + type(exc).__name__)


def _unavailable(reason: str) -> dict:
    return {"status": "unavailable", "bytes": None, "reason": reason}


class ProcessResourceSampler:
    """Lazy optional psutil/NVML access; WDDM or permissions may hide GPU use."""

    def __init__(self, pid: int | None = None):
        self.pid = os.getpid() if pid is None else pid
        if type(self.pid) is not int or self.pid < 1:
            raise ValueError("pid must be a positive integer")
        self._nvml = None
        self._nvml_attempted = False
        self._device_budget = DeviceGPUBudgetSampler()

    def _memory(self) -> dict:
        try:
            import psutil
            info = psutil.Process(self.pid).memory_info()
            value = info.rss
            if type(value) is not int or value < 0:
                raise ValueError("invalid_rss")
            result = {"status": "available", "bytes": value, "source": "psutil.rss",
                      "scope": "single_pid_excludes_children"}
            peak = getattr(info, "peak_wset", None)
            if type(peak) is int and peak >= value:
                result.update({"process_lifetime_peak_working_set_bytes": peak,
                               "peak_scope": "process_lifetime_not_monitor_or_task_interval"})
            return result
        except ImportError:
            reason = "psutil_not_installed"
        except Exception as exc:
            reason = type(exc).__name__
        if _WINDOWS:
            return {**_windows_process_memory(self.pid), "fallback_reason": reason}
        return _unavailable(reason)

    def _gpu(self) -> dict:
        if not self._nvml_attempted:
            self._nvml_attempted = True
            try:
                import pynvml
                pynvml.nvmlInit()
                self._nvml = pynvml
            except Exception:
                self._nvml = None
        if self._nvml is None:
            return _unavailable("nvml_unavailable")
        nvml = self._nvml
        devices = []
        try:
            count = nvml.nvmlDeviceGetCount()
            if not count:
                return _unavailable("no_nvml_devices")
            for index in range(count):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                # Compute and graphics lists can contain the same process. Use
                # the larger reported allocation, never double-count the PID.
                allocations = []
                for getter in (nvml.nvmlDeviceGetComputeRunningProcesses,
                               nvml.nvmlDeviceGetGraphicsRunningProcesses):
                    rows = getter(handle)
                    for row in rows:
                        if row.pid != self.pid:
                            continue
                        value = row.usedGpuMemory
                        if not isinstance(value, int) or value < 0 or value >= (1 << 63):
                            return _unavailable("driver_process_memory_unavailable")
                        allocations.append(value)
                devices.append({"device_index": index, "bytes": max(allocations, default=0)})
            return {"status": "available", "bytes": sum(d["bytes"] for d in devices),
                    "source": "nvml.process_allocations", "devices": devices}
        except Exception as exc:
            return _unavailable(type(exc).__name__)

    def __call__(self) -> dict:
        return {"pid": self.pid, "memory": self._memory(), "gpu_memory": self._gpu(),
                "device_gpu_memory": self._device_budget()}

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None


class ResourceMonitor:
    """Bounded aggregates of sampled peaks; unobserved peaks remain unknown."""

    def __init__(self, *, sampler: Callable | None = None, interval_seconds: float = 0.25,
                 clock: Callable = time.perf_counter, pid: int | None = None):
        if isinstance(interval_seconds, bool) or not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("sampling interval must be positive and finite")
        self.sampler = sampler if sampler is not None else ProcessResourceSampler(pid)
        self._owns_sampler = sampler is None
        self.interval_seconds = float(interval_seconds)
        self.clock = clock
        self._started = self.clock()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._count = 0
        self._failures = 0
        self._last = None
        self._peaks = {"memory": None, "gpu_memory": None, "device_gpu_memory": None}
        self._available = {key: 0 for key in self._peaks}
        self._reasons = {key: set() for key in self._peaks}
        self._sources = {key: set() for key in self._peaks}
        self._lifetime_peak_working_set = None

    def sample_once(self) -> dict:
        try:
            sample = dict(self.sampler())
            failure = False
        except Exception as exc:
            sample = {"memory": _unavailable(type(exc).__name__),
                      "gpu_memory": _unavailable(type(exc).__name__)}
            failure = True
        sample["elapsed_seconds"] = max(0.0, self.clock() - self._started)
        with self._lock:
            self._count += 1
            self._failures += int(failure)
            for key in self._peaks:
                observation = sample.get(key, {})
                value = observation.get("bytes") if isinstance(observation, dict) else None
                valid = isinstance(observation, dict) and observation.get("status") == "available" and type(value) is int and value >= 0
                if valid:
                    self._available[key] += 1
                    prior = self._peaks[key]
                    self._peaks[key] = value if prior is None else max(prior, value)
                    if observation.get("source"):
                        self._sources[key].add(str(observation["source"]))
                    peak = observation.get("process_lifetime_peak_working_set_bytes") if key == "memory" else None
                    if type(peak) is int and peak >= value and observation.get("peak_scope") == "process_lifetime_not_monitor_or_task_interval":
                        self._lifetime_peak_working_set = max(self._lifetime_peak_working_set or 0, peak)
                else:
                    reason = observation.get("reason", "invalid_or_missing_sample") if isinstance(observation, dict) else "invalid_sample"
                    self._reasons[key].add(str(reason))
            self._last = sample
        return sample

    def summary(self) -> dict:
        with self._lock:
            resources = {}
            for key in self._peaks:
                available = self._available[key]
                resources[key] = {"status": "available" if available == self._count and available else "partial" if available else "unavailable",
                                  "sampled_peak_bytes": self._peaks[key], "available_samples": available,
                                  "unavailable_samples": self._count - available,
                                  "unavailable_reasons": sorted(self._reasons[key]),
                                  "sources": sorted(self._sources[key])}
            resources["memory"].update({"scope": "single_pid_excludes_children",
                                        "process_lifetime_peak_working_set_bytes": self._lifetime_peak_working_set,
                                        "peak_scope": "process_lifetime_not_monitor_or_task_interval"})
            resources["gpu_memory"]["scope"] = "monitored_pid_driver_allocations"
            resources["device_gpu_memory"].update({"scope": "device_aggregate_includes_other_processes",
                                                  "purpose": "budget_observation_only", "attributable_to_monitored_pid": False,
                                                  "minimum_probe_interval_seconds": 1.0})
            return {"sampling_interval_seconds": self.interval_seconds, "exact_peak": False,
                    "sample_count": self._count, "sampler_failures": self._failures,
                    "elapsed_seconds": max(0.0, self.clock() - self._started),
                    "monitor_running": bool(self._thread and self._thread.is_alive()),
                    "last_sample": self._last, **resources}

    def start(self) -> "ResourceMonitor":
        if self._thread is not None:
            raise RuntimeError("monitor cannot be started twice")
        self.sample_once()
        def run():
            try:
                while not self._stop.wait(self.interval_seconds):
                    self.sample_once()
            finally:
                # A stop timeout may return while a sample is blocked. Release
                # optional NVML when that sample eventually returns as well.
                if self._owns_sampler:
                    self.sampler.close()
        self._thread = threading.Thread(target=run, name="resource-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout_seconds: float = 2.0) -> dict:
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("stop timeout must be non-negative and finite")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_seconds)
        result = self.summary()
        # A blocked sampler remains visible as running; don't close NVML while
        # its worker may still be using it.
        if self._owns_sampler and not result["monitor_running"]:
            self.sampler.close()
        return result

    def __enter__(self):
        return self.start()

    def __exit__(self, *_args):
        self.stop()


class ResourceBudgetExceeded(RuntimeError):
    pass


class ExecutionBudgetRejected(ResourceBudgetExceeded):
    """A failed or unobservable preflight; receipt is safe to save locally."""

    def __init__(self, receipt):
        super().__init__("execution_resource_budget_not_satisfied")
        self.receipt = receipt


def _system_available_memory() -> dict:
    if not _WINDOWS:
        return _unavailable("windows_global_memory_not_supported")
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]
    try:
        kernel = ctypes.WinDLL("kernel32.dll", use_last_error=True, winmode=0x800)
        kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
        kernel.GlobalMemoryStatusEx.restype = ctypes.c_int
        state = MemoryStatus()
        state.dwLength = ctypes.sizeof(state)
        if not kernel.GlobalMemoryStatusEx(ctypes.byref(state)):
            return _unavailable("windows_global_memory_status_failed")
        if state.ullAvailPhys > state.ullTotalPhys or state.ullTotalPhys == 0:
            return _unavailable("windows_global_memory_status_invalid")
        return {"status": "available", "bytes": int(state.ullAvailPhys), "total_bytes": int(state.ullTotalPhys),
                "source": "windows.GlobalMemoryStatusEx.ullAvailPhys", "scope": "system_available_physical_memory"}
    except Exception as exc:
        return _unavailable("windows_global_memory_" + type(exc).__name__)


def execution_budget_sample() -> dict:
    """Fresh observation; device sampler cache is not reused across preflights."""
    return {"available_memory": _system_available_memory(), "device_gpu_memory": DeviceGPUBudgetSampler()()}


def check_execution_budget(budget: dict, *, sampler: Callable | None = None) -> dict:
    """Check an explicit headroom floor; not a reservation or peak guarantee.

    Missing measurements reject a required resource. Never lower model quality,
    switch devices or terminate another process to make the budget pass.
    """
    required = {"minimum_available_memory_bytes", "gpu_required", "gpu_device_index", "minimum_available_gpu_bytes"}
    if not isinstance(budget, dict) or set(budget) != required:
        raise ValueError("explicit_execution_budget_fields_required")
    if type(budget["gpu_required"]) is not bool or any(type(budget[key]) is not int or budget[key] < 0 for key in required - {"gpu_required"}):
        raise ValueError("invalid_execution_budget_values")
    if budget["minimum_available_memory_bytes"] <= 0 or (budget["gpu_required"] and budget["minimum_available_gpu_bytes"] <= 0):
        raise ValueError("required_resource_headroom_must_be_positive")
    try:
        observation = (sampler or execution_budget_sample)()
        if not isinstance(observation, dict):
            raise TypeError("invalid_budget_observation")
        # Receipts must remain writable even when a sampler returns NaN or an
        # invalid object. Invalid observations become unknown, never zero.
        json.dumps(observation, allow_nan=False)
    except Exception as exc:
        observation = {"available_memory": _unavailable(type(exc).__name__), "device_gpu_memory": _unavailable(type(exc).__name__)}
    reasons = []
    memory = observation.get("available_memory", {})
    memory = memory if isinstance(memory, dict) else {}
    available = memory.get("bytes")
    if memory.get("status") != "available" or type(available) is not int or available < 0:
        reasons.append("available_memory_unobserved")
    elif available < budget["minimum_available_memory_bytes"]:
        reasons.append("available_memory_below_floor")
    gpu_available = None
    if budget["gpu_required"]:
        gpu = observation.get("device_gpu_memory", {})
        gpu = gpu if isinstance(gpu, dict) else {}
        devices = gpu.get("devices")
        matches = [row for row in devices if isinstance(row, dict) and type(row.get("device_index")) is int
                   and row["device_index"] == budget["gpu_device_index"]] if isinstance(devices, list) else []
        if gpu.get("status") == "available" and len(matches) == 1:
            used, total = matches[0].get("bytes"), matches[0].get("total_bytes")
            if type(used) is int and type(total) is int and 0 <= used <= total and total > 0:
                gpu_available = total - used
        if gpu_available is None:
            reasons.append("required_device_memory_unobserved")
        elif gpu_available < budget["minimum_available_gpu_bytes"]:
            reasons.append("available_device_memory_below_floor")
    receipt = {"schema_version": "execution_resource_budget.v1", "status": "rejected" if reasons else "passed",
               "checked_unix_seconds": time.time(), "budget": dict(budget), "observations": observation,
               "selected_device_available_bytes": gpu_available, "reasons": reasons,
               "scope": "instantaneous_headroom_floor_not_reservation_or_peak_guarantee",
               "model_parameters_changed": False, "other_processes_terminated": False}
    if reasons:
        raise ExecutionBudgetRejected(receipt)
    return receipt


@dataclass(frozen=True)
class ResourceLease:
    ticket_id: str
    label: str
    estimated_memory_bytes: int
    estimated_gpu_bytes: int
    queued_at: float
    acquired_at: float
    released_at: float | None = None

    def to_dict(self) -> dict:
        return {**asdict(self), "queue_seconds": max(0.0, self.acquired_at - self.queued_at),
                "execution_seconds": None if self.released_at is None else max(0.0, self.released_at - self.acquired_at)}


class ResourceBudgetQueue:
    """FIFO, in-process reservations. Default serializes one heavy GPU task.

    Limits concern estimates reserved by this queue, not measured global free
    memory. Unknown estimates are explicit zero reservations, not measured zero
    consumption. Production's existing internal concurrency is unchanged until
    a caller intentionally opts into this queue.
    """

    def __init__(self, *, max_active: int = 1, memory_budget_bytes: int | None = None,
                 gpu_budget_bytes: int | None = None, clock: Callable = time.perf_counter):
        if type(max_active) is not int or max_active < 1:
            raise ValueError("max_active must be positive")
        for value in (memory_budget_bytes, gpu_budget_bytes):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("budgets must be non-negative integer bytes")
        self.max_active = max_active
        self.memory_budget_bytes = memory_budget_bytes
        self.gpu_budget_bytes = gpu_budget_bytes
        self.clock = clock
        self._condition = threading.Condition()
        self._waiting = deque()
        self._active = {}

    def _fits(self, memory: int, gpu: int) -> bool:
        return (len(self._active) < self.max_active and
                (self.memory_budget_bytes is None or memory + sum(x.estimated_memory_bytes for x in self._active.values()) <= self.memory_budget_bytes) and
                (self.gpu_budget_bytes is None or gpu + sum(x.estimated_gpu_bytes for x in self._active.values()) <= self.gpu_budget_bytes))

    def acquire(self, label: str, *, estimated_memory_bytes: int = 0,
                estimated_gpu_bytes: int = 0, timeout_seconds: float | None = None) -> ResourceLease:
        for value in (estimated_memory_bytes, estimated_gpu_bytes):
            if type(value) is not int or value < 0:
                raise ValueError("resource estimates must be non-negative integer bytes")
        if timeout_seconds is not None and (isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds < 0):
            raise ValueError("timeout must be finite and non-negative")
        if ((self.memory_budget_bytes is not None and estimated_memory_bytes > self.memory_budget_bytes) or
                (self.gpu_budget_bytes is not None and estimated_gpu_bytes > self.gpu_budget_bytes)):
            raise ResourceBudgetExceeded("request exceeds configured reservation budget; input unchanged")
        queued_at = self.clock()
        ticket = uuid.uuid4().hex
        with self._condition:
            self._waiting.append(ticket)
            try:
                while self._waiting[0] != ticket or not self._fits(estimated_memory_bytes, estimated_gpu_bytes):
                    remaining = None if timeout_seconds is None else timeout_seconds - (self.clock() - queued_at)
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError("resource queue deadline exceeded; input unchanged")
                    self._condition.wait(remaining)
                self._waiting.popleft()
                lease = ResourceLease(ticket, str(label), estimated_memory_bytes, estimated_gpu_bytes, queued_at, self.clock())
                self._active[ticket] = lease
                self._condition.notify_all()
                return lease
            except BaseException:
                if ticket in self._waiting:
                    self._waiting.remove(ticket)
                self._condition.notify_all()
                raise

    def release(self, lease: ResourceLease) -> None:
        with self._condition:
            if self._active.get(lease.ticket_id) is not lease:
                raise ValueError("lease is not active in this queue")
            del self._active[lease.ticket_id]
            object.__setattr__(lease, "released_at", self.clock())
            self._condition.notify_all()

    @contextmanager
    def reserve(self, label: str, **kwargs):
        lease = self.acquire(label, **kwargs)
        try:
            yield lease
        finally:
            self.release(lease)

    def snapshot(self) -> dict:
        with self._condition:
            return {"scope": "in_process", "active": len(self._active), "waiting": len(self._waiting),
                    "max_active": self.max_active,
                    "reserved_memory_bytes": sum(x.estimated_memory_bytes for x in self._active.values()),
                    "reserved_gpu_bytes": sum(x.estimated_gpu_bytes for x in self._active.values())}
