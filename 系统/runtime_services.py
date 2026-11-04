"""Optional persistent model workers for the integrated desktop application.

The workers deliberately remain separate OS processes.  The frozen deepfake
adapter and the identity module can therefore keep using their own Python 3.12
environments while their expensive model objects survive across UI queries.

This module only depends on the Python standard library so the frontend may
import it even when neither inference environment is currently available.
"""

from __future__ import annotations

import atexit
from collections import deque
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import uuid


PROTOCOL_NAME = "face-source-trace-runtime-v1"
# 按此顺序逐个启动：通用 AIGC 模型加载最慢，排最后，不拖住其他模块就绪。
WORKER_NAMES = ("deepfake", "identity", "general_aigc")


class RuntimeServiceError(RuntimeError):
    """Base class for a controlled persistent-runtime failure."""

    def __init__(
        self,
        message: str,
        *,
        service: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.service = service
        self.request_id = request_id


class RuntimeServiceUnavailable(RuntimeServiceError):
    """A worker is disabled, failed to start, or exited unexpectedly."""


class RuntimeServiceTimeout(RuntimeServiceError):
    """A worker did not answer within the caller's deadline."""


class RuntimeServiceProtocolError(RuntimeServiceError):
    """A worker returned a malformed or unsuccessful protocol response."""


class RuntimeServiceRemoteError(RuntimeServiceError):
    """A healthy worker rejected a request before producing a result payload."""


ProgressCallback = Callable[[dict[str, Any]], None]


def _notify(callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback(dict(event))
    except Exception:
        # Progress reporting must never take down model services or a query.
        return


def _expand_tokens(value: str | Path, config_path: Path) -> str:
    raw = str(value)
    configured_data_root = os.environ.get("FSTS_DATA_ROOT")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if configured_data_root:
        data_root = Path(configured_data_root).expanduser()
    elif local_app_data:
        data_root = Path(local_app_data) / "FaceSourceTrace"
    else:
        data_root = Path.home() / "AppData" / "Local" / "FaceSourceTrace"
    replacements = {
        "{APP_ROOT}": str(config_path.parent),
        "{CONFIG_ROOT}": str(config_path.parent),
        "{DATA_ROOT}": str(data_root),
        "{CURRENT_PYTHON}": sys.executable,
    }
    for token, replacement in replacements.items():
        raw = raw.replace(token, replacement)
    return os.path.expandvars(os.path.expanduser(raw))


def _resolve_path(value: str | Path, config_path: Path) -> Path:
    path = Path(_expand_tokens(value, config_path))
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve(strict=False)


def _positive_timeout(value: Any, *, label: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeServiceError(f"{label} 必须是正数，实际值：{value!r}") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeServiceError(f"{label} 必须是大于 0 的有限数值，实际值：{timeout}")
    return timeout


def _detection_scale_chain(section: Mapping[str, Any]) -> list[int] | None:
    """Validate the optional identity detection scale chain (M2 upgrade).

    Only forwarded when the configuration states it explicitly.  The v2 worker
    carries its own calibrated default, so an absent key leaves every other
    identity worker on its frozen single-scale behaviour.
    """

    if "det_size_chain" not in section:
        return None
    raw = section.get("det_size_chain")
    if not isinstance(raw, list) or not raw:
        raise RuntimeServiceUnavailable(
            "identity.det_size_chain 必须是至少含一档的整数数组", service="identity"
        )
    chain: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int) or item < 64:
            raise RuntimeServiceUnavailable(
                f"identity.det_size_chain 元素必须是 >=64 的整数，实际值：{item!r}",
                service="identity",
            )
        if item not in chain:
            chain.append(item)
    return chain


class _WorkerChannel:
    """Locked request/response channel over a worker's stdin/stdout JSON lines."""

    _EXIT_MARKER = "__runtime_worker_exit__"

    def __init__(
        self,
        *,
        name: str,
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
        startup_timeout: float,
        progress_callback: ProgressCallback | None,
    ) -> None:
        self.name = name
        self.command = [str(item) for item in command]
        self.cwd = Path(cwd)
        self.environment = dict(environment)
        self.startup_timeout = float(startup_timeout)
        self.progress_callback = progress_callback

        self._process: subprocess.Popen[str] | None = None
        self._state_lock = threading.RLock()
        self._request_lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._pending: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._startup_event = threading.Event()
        self._startup_message: dict[str, Any] | None = None
        self._closing = False

        self._state = "stopped"
        self._ready = False
        self._health = "stopped"
        self._error: str | None = None
        self._started_at: float | None = None
        self._last_response_at: float | None = None
        self._completed_requests = 0
        self._details: dict[str, Any] = {}
        self._noise_count = 0
        self._noise_tail: deque[str] = deque(maxlen=10)
        self._stderr_tail: deque[str] = deque(maxlen=20)

    def start(self) -> None:
        with self._state_lock:
            if self._process is not None and self._process.poll() is None and self._ready:
                return
            self._state = "starting"
            self._ready = False
            self._health = "starting"
            self._error = None
            self._closing = False
            self._startup_event.clear()
            self._startup_message = None

        _notify(
            self.progress_callback,
            {
                "type": "runtime_service",
                "service": self.name,
                "state": "starting",
                "message": f"正在初始化 {self.name} 持久工作进程",
            },
        )
        creationflags = 0
        if os.name == "nt":
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            process = subprocess.Popen(
                self.command,
                cwd=str(self.cwd),
                env=self.environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except (OSError, ValueError) as exc:
            message = f"无法启动 {self.name} 持久工作进程：{type(exc).__name__}: {exc}"
            self._set_error(message)
            raise RuntimeServiceUnavailable(message, service=self.name) from exc

        with self._state_lock:
            self._process = process
            self._started_at = time.time()

        threading.Thread(
            target=self._read_stdout,
            name=f"{self.name}-runtime-stdout",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._read_stderr,
            name=f"{self.name}-runtime-stderr",
            daemon=True,
        ).start()

        deadline = time.monotonic() + self.startup_timeout
        while not self._startup_event.is_set():
            returncode = process.poll()
            if returncode is not None:
                message = self._exit_message(returncode, "初始化完成前")
                self._set_error(message)
                self._force_terminate()
                raise RuntimeServiceUnavailable(message, service=self.name)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                message = (
                    f"{self.name} 持久工作进程初始化超时"
                    f"（{self.startup_timeout:.1f} 秒）"
                )
                self._set_error(message)
                self._force_terminate()
                raise RuntimeServiceTimeout(message, service=self.name)
            self._startup_event.wait(min(0.1, remaining))

        message = self._startup_message or {}
        if message.get("ready") is not True or message.get("health") != "ok":
            detail = str(message.get("error") or "工作进程没有报告 ready/ok")
            failure = f"{self.name} 持久工作进程初始化失败：{detail}"
            self._set_error(failure)
            self._force_terminate()
            raise RuntimeServiceUnavailable(failure, service=self.name)

        with self._state_lock:
            self._state = "ready"
            self._ready = True
            self._health = "ok"
            self._error = None
            self._last_response_at = time.time()
        _notify(
            self.progress_callback,
            {
                "type": "runtime_service",
                "service": self.name,
                "state": "ready",
                "message": f"{self.name} 模型已初始化并可复用",
                "pid": process.pid,
            },
        )

    def _read_stdout(self) -> None:
        process = self._process
        stream = process.stdout if process is not None else None
        if stream is None:
            self._mark_process_ended("工作进程 stdout 不可用")
            return
        try:
            for raw_line in iter(stream.readline, ""):
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self._record_noise(line)
                    continue
                if not self._is_protocol_message(message):
                    self._record_noise(line)
                    continue
                self._dispatch_message(message)
        except (OSError, ValueError) as exc:
            if not self._closing:
                self._mark_process_ended(
                    f"读取工作进程 stdout 失败：{type(exc).__name__}: {exc}"
                )
            return
        returncode = process.poll() if process is not None else None
        self._mark_process_ended(self._exit_message(returncode, "运行期间"))

    def _read_stderr(self) -> None:
        process = self._process
        stream = process.stderr if process is not None else None
        if stream is None:
            return
        try:
            for raw_line in iter(stream.readline, ""):
                line = raw_line.rstrip("\r\n")
                if line:
                    with self._state_lock:
                        self._stderr_tail.append(line)
        except (OSError, ValueError):
            return

    @staticmethod
    def _is_protocol_message(message: Any) -> bool:
        return (
            isinstance(message, dict)
            and message.get("protocol") == PROTOCOL_NAME
            and "request_id" in message
            and isinstance(message.get("ready"), bool)
            and isinstance(message.get("health"), str)
        )

    def _dispatch_message(self, message: dict[str, Any]) -> None:
        if message.get("type") == "ready" and message.get("request_id") is None:
            with self._state_lock:
                self._startup_message = dict(message)
                details = message.get("details")
                self._details = dict(details) if isinstance(details, dict) else {}
            self._startup_event.set()
            return

        request_id = message.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            self._record_noise(json.dumps(message, ensure_ascii=False))
            return
        with self._pending_lock:
            response_queue = self._pending.get(request_id)
        if response_queue is None:
            self._record_noise(json.dumps(message, ensure_ascii=False))
            return
        details = message.get("details")
        if isinstance(details, dict):
            with self._state_lock:
                self._details = dict(details)
        response_queue.put(dict(message))

    def _record_noise(self, line: str) -> None:
        with self._state_lock:
            self._noise_count += 1
            self._noise_tail.append(line[:1000])

    def _mark_process_ended(self, message: str) -> None:
        with self._state_lock:
            closing = self._closing
            if closing:
                self._ready = False
                if self._state == "stopping":
                    self._health = "stopped"
            elif self._state != "error":
                self._state = "error"
                self._ready = False
                self._health = "error"
                self._error = message
        self._startup_event.set()
        marker = {self._EXIT_MARKER: True, "error": message}
        with self._pending_lock:
            pending = list(self._pending.values())
        for response_queue in pending:
            response_queue.put(dict(marker))

    def _exit_message(self, returncode: int | None, when: str) -> str:
        suffix = "未知" if returncode is None else str(returncode)
        stderr = ""
        with self._state_lock:
            if self._stderr_tail:
                stderr = f"；stderr: {self._stderr_tail[-1]}"
        return f"{self.name} 持久工作进程在{when}退出，退出码 {suffix}{stderr}"

    def _set_error(self, message: str) -> None:
        with self._state_lock:
            self._state = "error"
            self._ready = False
            self._health = "error"
            self._error = message
        _notify(
            self.progress_callback,
            {
                "type": "runtime_service",
                "service": self.name,
                "state": "error",
                "message": message,
            },
        )

    def _assert_alive(self) -> subprocess.Popen[str]:
        process = self._process
        if process is None:
            message = self._error or f"{self.name} 持久工作进程尚未启动"
            raise RuntimeServiceUnavailable(message, service=self.name)
        returncode = process.poll()
        if returncode is not None:
            message = self._exit_message(returncode, "请求前")
            self._set_error(message)
            raise RuntimeServiceUnavailable(message, service=self.name)
        if not self._ready:
            message = self._error or f"{self.name} 持久工作进程尚未 ready"
            raise RuntimeServiceUnavailable(message, service=self.name)
        return process

    def request(self, action: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        deadline_seconds = _positive_timeout(timeout, label=f"{self.name} 请求 timeout")
        with self._request_lock:
            process = self._assert_alive()
            request_id = uuid.uuid4().hex
            response_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
            with self._pending_lock:
                self._pending[request_id] = response_queue
            request = {
                "protocol": PROTOCOL_NAME,
                "type": "request",
                "request_id": request_id,
                "ready": True,
                "health": "ok",
                "action": str(action),
                **dict(payload),
            }
            try:
                if process.stdin is None:
                    raise OSError("工作进程 stdin 不可用")
                process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                with self._pending_lock:
                    self._pending.pop(request_id, None)
                message = f"无法向 {self.name} 工作进程发送请求：{type(exc).__name__}: {exc}"
                self._set_error(message)
                raise RuntimeServiceUnavailable(
                    message, service=self.name, request_id=request_id
                ) from exc

            response: dict[str, Any] | None = None
            deadline = time.monotonic() + deadline_seconds
            try:
                while response is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        message = (
                            f"{self.name} 持久工作进程请求超时"
                            f"（{deadline_seconds:.1f} 秒，request_id={request_id}）"
                        )
                        self._set_error(message)
                        self._force_terminate()
                        raise RuntimeServiceTimeout(
                            message, service=self.name, request_id=request_id
                        )
                    try:
                        response = response_queue.get(timeout=min(0.1, remaining))
                    except queue.Empty:
                        returncode = process.poll()
                        if returncode is not None:
                            message = self._exit_message(returncode, "请求期间")
                            self._set_error(message)
                            raise RuntimeServiceUnavailable(
                                message,
                                service=self.name,
                                request_id=request_id,
                            )
            finally:
                with self._pending_lock:
                    self._pending.pop(request_id, None)

            assert response is not None
            if response.get(self._EXIT_MARKER):
                message = str(response.get("error") or f"{self.name} 工作进程已退出")
                raise RuntimeServiceUnavailable(
                    message, service=self.name, request_id=request_id
                )
            if response.get("request_id") != request_id:
                raise RuntimeServiceProtocolError(
                    f"{self.name} 响应 request_id 不匹配",
                    service=self.name,
                    request_id=request_id,
                )
            if response.get("health") != "ok":
                message = str(response.get("error") or f"{self.name} 报告 health 异常")
                self._set_error(message)
                raise RuntimeServiceUnavailable(
                    message, service=self.name, request_id=request_id
                )
            if response.get("ok") is not True:
                message = str(response.get("error") or f"{self.name} 拒绝了请求")
                raise RuntimeServiceRemoteError(
                    message, service=self.name, request_id=request_id
                )
            with self._state_lock:
                self._completed_requests += 1
                self._last_response_at = time.time()
            return response

    def snapshot(self) -> dict[str, Any]:
        process = self._process
        if process is not None and process.poll() is not None and self._ready:
            self._mark_process_ended(self._exit_message(process.returncode, "状态检查时"))
        with self._state_lock:
            return {
                "state": self._state,
                "ready": self._ready,
                "healthy": self._health == "ok" and self._ready,
                "health": self._health,
                "pid": process.pid if process is not None and process.poll() is None else None,
                "error": self._error,
                "started_at": self._started_at,
                "last_response_at": self._last_response_at,
                "requests_completed": self._completed_requests,
                "details": dict(self._details),
                "noise_lines_ignored": self._noise_count,
                "noise_tail": list(self._noise_tail),
                "stderr_tail": list(self._stderr_tail),
            }

    def _force_terminate(self) -> None:
        process = self._process
        if process is None:
            return
        with self._state_lock:
            previous_closing = self._closing
            self._closing = True
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        except (OSError, subprocess.SubprocessError):
            pass
        finally:
            with self._state_lock:
                self._closing = previous_closing

    def stop(self) -> None:
        process = self._process
        if process is None:
            with self._state_lock:
                if self._state != "error":
                    self._state = "stopped"
                    self._health = "stopped"
                    self._ready = False
            return
        with self._state_lock:
            was_ready = self._ready
            self._closing = True
            self._state = "stopping"
        acquired = self._request_lock.acquire(timeout=0.5)
        try:
            if acquired and was_ready and process.poll() is None:
                try:
                    self.request("shutdown", {}, timeout=1.0)
                except RuntimeServiceError:
                    pass
        finally:
            if acquired:
                self._request_lock.release()
        self._force_terminate()
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        with self._state_lock:
            self._closing = False
            self._ready = False
            self._health = "stopped"
            self._state = "stopped"
        _notify(
            self.progress_callback,
            {
                "type": "runtime_service",
                "service": self.name,
                "state": "stopped",
                "message": f"{self.name} 持久工作进程已停止",
            },
        )


class RuntimeServiceManager:
    """Own optional deepfake and identity workers for one frontend process."""

    def __init__(
        self,
        config_path: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve(strict=False)
        self.progress_callback = progress_callback
        self._lock = threading.RLock()
        self._channels: dict[str, _WorkerChannel] = {}
        self._static_status: dict[str, dict[str, Any]] = {}
        self._started = False
        self._closed = False
        self._config = self._load_config()
        atexit.register(self.close)

    def _load_config(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeServiceError(
                f"无法读取持久运行服务配置 {self.config_path}：{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeServiceError(f"配置根节点必须是 JSON 对象：{self.config_path}")
        return payload

    def _worker_spec(self, name: str) -> tuple[list[str], Path, dict[str, str], float] | None:
        section = self._config.get(name)
        if not isinstance(section, dict):
            if name == "general_aigc":
                # 这个分支可选：配置里没有就按关闭处理，不影响其他模块的就绪状态。
                self._static_status[name] = self._status_record("disabled")
                return None
            self._static_status[name] = self._status_record(
                "unavailable", error=f"配置缺少 {name} 节"
            )
            return None
        runtime = self._config.get("runtime")
        runtime = runtime if isinstance(runtime, dict) else {}
        global_enabled = runtime.get("persistent_workers_enabled", True)
        section_enabled = section.get("enabled", True)
        persistent_enabled = section.get("persistent_enabled", True)
        for label, value in (
            ("runtime.persistent_workers_enabled", global_enabled),
            (f"{name}.enabled", section_enabled),
            (f"{name}.persistent_enabled", persistent_enabled),
        ):
            if not isinstance(value, bool):
                raise RuntimeServiceUnavailable(
                    f"{label} 必须为布尔值，实际值：{value!r}", service=name
                )
        # The runtime-level setting is a hard kill switch.  Per-module settings
        # may further disable a worker but can never override a global shutdown.
        enabled = global_enabled and section_enabled and persistent_enabled
        if not enabled:
            self._static_status[name] = self._status_record("disabled")
            return None

        packaged_worker_value = section.get("worker_executable")
        if packaged_worker_value:
            if not isinstance(packaged_worker_value, str):
                raise RuntimeServiceUnavailable(
                    f"{name}.worker_executable 必须是字符串", service=name
                )
            executable = _resolve_path(packaged_worker_value, self.config_path)
            if not executable.is_file():
                raise RuntimeServiceUnavailable(
                    f"{name} 持久 worker 可执行文件不存在：{executable}", service=name
                )
            raw_worker_args = section.get("worker_executable_args", [])
            if not isinstance(raw_worker_args, list) or not all(
                isinstance(item, (str, int, float)) for item in raw_worker_args
            ):
                raise RuntimeServiceUnavailable(
                    f"{name}.worker_executable_args 必须是标量数组", service=name
                )
            command = [str(executable), *(str(item) for item in raw_worker_args)]
            worker_cwd = executable.parent
        else:
            executable_value = section.get("python")
            if not isinstance(executable_value, str) or not executable_value:
                raise RuntimeServiceUnavailable(
                    f"{name}.python 未配置", service=name
                )
            executable = _resolve_path(executable_value, self.config_path)
            default_worker = (
                self.config_path.parent
                / "modules"
                / {
                    "deepfake": "deepfake_detection",
                    "general_aigc": "general_aigc",
                }.get(name, "identity_search")
                / "persistent_worker.py"
            )
            worker_value = section.get("persistent_worker", default_worker)
            if isinstance(worker_value, bool):
                worker_value = default_worker
            worker = _resolve_path(worker_value, self.config_path)
            if not executable.is_file():
                raise RuntimeServiceUnavailable(
                    f"{name} Python 不存在：{executable}", service=name
                )
            if not worker.is_file():
                raise RuntimeServiceUnavailable(
                    f"{name} 持久 worker 不存在：{worker}", service=name
                )
            raw_python_args = section.get("python_args", [])
            if not isinstance(raw_python_args, list) or not all(
                isinstance(item, str) for item in raw_python_args
            ):
                raise RuntimeServiceUnavailable(
                    f"{name}.python_args 必须是字符串数组", service=name
                )
            command = [str(executable), *raw_python_args, "-u", str(worker)]
            worker_cwd = worker.parent
        if "persistent_worker_args" in section:
            explicit = section["persistent_worker_args"]
            if not isinstance(explicit, list) or not all(
                isinstance(item, (str, int, float)) for item in explicit
            ):
                raise RuntimeServiceUnavailable(
                    f"{name}.persistent_worker_args 必须是标量数组", service=name
                )
            command.extend(str(item) for item in explicit)
        elif name == "deepfake":
            command.extend(self._deepfake_worker_args(section, runtime))
        elif name == "general_aigc":
            command.extend(self._general_aigc_worker_args(section))
        else:
            command.extend(self._identity_worker_args(section, runtime))

        startup_timeout = _positive_timeout(
            section.get(
                "startup_timeout_seconds",
                runtime.get("persistent_startup_timeout_seconds", section.get("timeout_seconds", 180)),
            ),
            label=f"{name} startup_timeout_seconds",
        )
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        configured_environment = section.get("persistent_environment", {})
        if configured_environment:
            if not isinstance(configured_environment, dict):
                raise RuntimeServiceUnavailable(
                    f"{name}.persistent_environment 必须是对象", service=name
                )
            environment.update(
                {
                    str(key): _expand_tokens(str(value), self.config_path)
                    for key, value in configured_environment.items()
                }
            )
        return command, worker_cwd, environment, startup_timeout

    def _deepfake_worker_args(
        self, section: Mapping[str, Any], runtime: Mapping[str, Any]
    ) -> list[str]:
        install_root = _resolve_path(
            section.get(
                "install_root",
                Path.home()
                / "AppData"
                / "Local"
                / "Programs"
                / "DeepfakeDetectionSystem",
            ),
            self.config_path,
        )
        tasks_root = _resolve_path(
            runtime.get("tasks_root", "runtime/tasks"), self.config_path
        )
        writable_root = _resolve_path(
            section.get(
                "persistent_writable_root",
                tasks_root.parent / "persistent_services" / "deepfake",
            ),
            self.config_path,
        )
        args = [
            "--install-root",
            str(install_root),
            "--device",
            str(section.get("device", "cpu")),
            "--writable-root",
            str(writable_root),
        ]
        if not bool(section.get("preload_model", True)):
            args.append("--no-preload")
        return args

    def _general_aigc_worker_args(self, section: Mapping[str, Any]) -> list[str]:
        args = ["--device", str(section.get("device", "auto"))]
        if section.get("model_package"):
            args.extend(
                ["--model-package", str(_resolve_path(section["model_package"], self.config_path))]
            )
        return args

    def _identity_worker_args(
        self, section: Mapping[str, Any], runtime: Mapping[str, Any]
    ) -> list[str]:
        library = _resolve_path(section.get("library", ""), self.config_path)
        cache_dir = _resolve_path(
            runtime.get("identity_cache", "runtime/cache/identity"), self.config_path
        )
        model_root = _resolve_path(
            section.get("model_root", Path.home() / ".insightface"), self.config_path
        )
        ctx = int(section.get("ctx", -1))
        device = str(section.get("device", "cpu" if ctx == -1 else "gpu"))
        args = [
            "--library",
            str(library),
            "--cache-dir",
            str(cache_dir),
            "--model-root",
            str(model_root),
            "--device",
            device,
            "--ctx",
            str(ctx),
        ]
        chain = _detection_scale_chain(section)
        if chain is not None:
            args.extend(["--det-size-chain", ",".join(str(item) for item in chain)])
        if section.get("multi_face_policy"):
            args.extend(["--multi-face-policy", str(section["multi_face_policy"])])
        return args

    @staticmethod
    def _status_record(state: str, error: str | None = None) -> dict[str, Any]:
        return {
            "state": state,
            "ready": False,
            "healthy": False,
            "health": "disabled" if state == "disabled" else "error",
            "pid": None,
            "error": error,
            "started_at": None,
            "last_response_at": None,
            "requests_completed": 0,
            "details": {},
            "noise_lines_ignored": 0,
            "noise_tail": [],
            "stderr_tail": [],
        }

    def start(self) -> dict[str, Any]:
        """Start available workers once; report per-worker failures for fallback."""

        with self._lock:
            if self._closed:
                raise RuntimeServiceUnavailable("RuntimeServiceManager 已关闭")
            if self._started:
                return self.get_status()
            self._started = True

        for name in WORKER_NAMES:
            try:
                spec = self._worker_spec(name)
                if spec is None:
                    continue
                command, cwd, environment, startup_timeout = spec
                channel = _WorkerChannel(
                    name=name,
                    command=command,
                    cwd=cwd,
                    environment=environment,
                    startup_timeout=startup_timeout,
                    progress_callback=self.progress_callback,
                )
                with self._lock:
                    self._channels[name] = channel
                channel.start()
            except RuntimeServiceError as exc:
                with self._lock:
                    channel = self._channels.get(name)
                    if channel is None:
                        self._static_status[name] = self._status_record(
                            "unavailable", error=str(exc)
                        )
                _notify(
                    self.progress_callback,
                    {
                        "type": "runtime_service",
                        "service": name,
                        "state": "error",
                        "message": str(exc),
                    },
                )
            except Exception as exc:
                message = f"{name} 持久服务初始化异常：{type(exc).__name__}: {exc}"
                with self._lock:
                    self._static_status[name] = self._status_record(
                        "unavailable", error=message
                    )
                _notify(
                    self.progress_callback,
                    {
                        "type": "runtime_service",
                        "service": name,
                        "state": "error",
                        "message": message,
                    },
                )
        return self.get_status()

    def get_status(self) -> dict[str, Any]:
        """Return a non-blocking truthful snapshot; it never starts a model."""

        with self._lock:
            statuses: dict[str, dict[str, Any]] = {}
            for name in WORKER_NAMES:
                channel = self._channels.get(name)
                if channel is not None:
                    statuses[name] = channel.snapshot()
                else:
                    statuses[name] = dict(
                        self._static_status.get(
                            name,
                            self._status_record(
                                "not_started" if not self._started else "unavailable",
                                error=None if not self._started else f"{name} worker 不可用",
                            ),
                        )
                    )
            enabled = [
                item for item in statuses.values() if item["state"] != "disabled"
            ]
            ready_count = sum(bool(item["ready"]) for item in enabled)
            if not enabled:
                overall_health = "disabled" if self._started else "not_started"
            elif ready_count == len(enabled):
                overall_health = "ok"
            elif ready_count:
                overall_health = "degraded"
            elif self._started:
                overall_health = "error"
            else:
                overall_health = "not_started"
            return {
                "started": self._started and not self._closed,
                "closed": self._closed,
                "ready": bool(enabled) and ready_count == len(enabled),
                "health": overall_health,
                **statuses,
            }

    def _channel_for(self, name: str) -> _WorkerChannel:
        with self._lock:
            if not self._started:
                raise RuntimeServiceUnavailable(
                    "RuntimeServiceManager 尚未 start()", service=name
                )
            channel = self._channels.get(name)
            if channel is None:
                status = self._static_status.get(name, {})
                message = str(status.get("error") or f"{name} 持久服务不可用")
                raise RuntimeServiceUnavailable(message, service=name)
            if channel.snapshot().get("ready") is not True:
                status = channel.snapshot()
                message = str(status.get("error") or f"{name} 持久服务不可用")
                raise RuntimeServiceUnavailable(message, service=name)
            return channel

    @staticmethod
    def _payload_from_response(
        name: str, response: Mapping[str, Any]
    ) -> dict[str, Any]:
        payload = response.get("payload")
        if not isinstance(payload, dict):
            raise RuntimeServiceProtocolError(
                f"{name} 工作进程响应缺少 payload 对象", service=name
            )
        return dict(payload)

    def run_deepfake(
        self,
        query_path: str | Path,
        output_path: str | Path,
        timeout: float,
    ) -> dict[str, Any]:
        channel = self._channel_for("deepfake")
        response = channel.request(
            "run",
            {
                "query_path": str(Path(query_path).expanduser().resolve(strict=False)),
                "output_path": str(Path(output_path).expanduser().resolve(strict=False)),
            },
            timeout,
        )
        return self._payload_from_response("deepfake", response)

    def _wait_until_started(self, name: str, deadline: float) -> None:
        """模型还在加载时等它就绪，避免再起一个进程重复加载同一份权重。"""

        while time.monotonic() < deadline:
            with self._lock:
                if not self._started or self._closed or name in self._static_status:
                    return
                channel = self._channels.get(name)
            if channel is not None and channel.snapshot().get("state") != "starting":
                return
            time.sleep(0.2)

    def run_general_aigc(
        self,
        input_path: str | Path,
        output_path: str | Path,
        timeout: float,
        face_crops: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout)
        self._wait_until_started("general_aigc", deadline)
        channel = self._channel_for("general_aigc")
        response = channel.request(
            "run",
            {
                "input_path": str(Path(input_path).expanduser().resolve(strict=False)),
                "output_path": str(Path(output_path).expanduser().resolve(strict=False)),
                "face_crops": None if face_crops is None else [str(item) for item in face_crops],
            },
            timeout,
        )
        return self._payload_from_response("general_aigc", response)

    def run_identity(
        self,
        query_path: str | Path,
        output_path: str | Path,
        library_path: str | Path,
        top_k: int,
        timeout: float,
        task_id: str | None = None,
        progress_events_path: str | Path | None = None,
        progress_latest_path: str | Path | None = None,
    ) -> dict[str, Any]:
        channel = self._channel_for("identity")
        response = channel.request(
            "run",
            {
                "query_path": str(Path(query_path).expanduser().resolve(strict=False)),
                "output_path": str(Path(output_path).expanduser().resolve(strict=False)),
                "library_path": str(
                    Path(library_path).expanduser().resolve(strict=False)
                ),
                "top_k": int(top_k),
                "task_id": task_id,
                "progress_events_path": (
                    str(Path(progress_events_path).expanduser().resolve(strict=False))
                    if progress_events_path is not None
                    else None
                ),
                "progress_latest_path": (
                    str(Path(progress_latest_path).expanduser().resolve(strict=False))
                    if progress_latest_path is not None
                    else None
                ),
            },
            timeout,
        )
        return self._payload_from_response("identity", response)

    def close(self) -> None:
        """Stop all child processes.  Safe to call repeatedly and from atexit."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            channels = list(self._channels.values())
        for channel in channels:
            try:
                channel.stop()
            except Exception:
                continue
        with self._lock:
            self._started = False

    stop = close

    def __enter__(self) -> "RuntimeServiceManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = [
    "RuntimeServiceManager",
    "RuntimeServiceError",
    "RuntimeServiceUnavailable",
    "RuntimeServiceTimeout",
    "RuntimeServiceProtocolError",
    "RuntimeServiceRemoteError",
]
