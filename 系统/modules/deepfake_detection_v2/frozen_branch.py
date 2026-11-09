"""把请求代理给 v0.4.0 冻结鉴伪 worker（人脸裁剪分支）。
设计意图：v2 进程是鉴伪的唯一入口，低于路由阈值的输入在这里转交冻结运行时处理，
冻结侧的代码、模型与阈值一行都不改，只作为子进程被调用。
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any


PROTOCOL_NAME = "face-source-trace-runtime-v1"


class FrozenBranchError(RuntimeError):
    pass


class FrozenBranchProxy:
    """常驻一个冻结 worker 子进程，按请求转发。"""

    def __init__(self, app_root: Path, writable_root: Path, *, device: str = "cpu",
                 install_root: Path | None = None, start_timeout: float = 600.0,
                 request_timeout: float = 300.0) -> None:
        self.app_root = Path(app_root)
        self.writable_root = Path(writable_root)
        self.device = device
        self.request_timeout = request_timeout
        self.start_timeout = start_timeout
        self.python = (self.app_root / "packaging_v0.3" / "resources"
                       / "portable_python" / "python.exe")
        self.worker = (self.app_root / "modules" / "deepfake_detection"
                       / "persistent_worker.py")
        # 默认指向冻结运行时；编排层若通过 --install-root 指定则以其为准。
        self.install_root = (Path(install_root) if install_root is not None
                             else self.app_root / "packaging_v0.3" / "resources"
                             / "deepfake_runtime")
        self.process: subprocess.Popen[str] | None = None
        self.request_count = 0

    @property
    def available(self) -> bool:
        return self.python.is_file() and self.worker.is_file() and self.install_root.is_dir()

    def _read_message(self, timeout: float) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.process.stdout.readline()
            if not line:
                if self.process.poll() is not None:
                    raise FrozenBranchError(
                        f"冻结 worker 提前退出，returncode={self.process.returncode}"
                    )
                continue
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise FrozenBranchError("等待冻结 worker 响应超时")

    def start(self) -> None:
        if self.process is not None:
            return
        if not self.available:
            raise FrozenBranchError("冻结鉴伪运行时文件缺失，无法启用回退分支")
        self.writable_root.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [str(self.python), "-E", "-S", "-B", str(self.worker),
             "--install-root", str(self.install_root), "--device", self.device,
             "--writable-root", str(self.writable_root)],
            cwd=str(self.worker.parent), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            bufsize=1,
        )
        ready = self._read_message(self.start_timeout)
        if not ready.get("ok"):
            raise FrozenBranchError(f"冻结 worker 未就绪：{ready}")

    def run(self, query_path: Path, output_path: Path) -> dict[str, Any]:
        self.start()
        assert self.process is not None and self.process.stdin is not None
        request_id = f"frozen_branch_{self.request_count + 1:06d}"
        self.request_count += 1
        request = {"protocol": PROTOCOL_NAME, "type": "request", "request_id": request_id,
                   "action": "run", "query_path": str(query_path),
                   "output_path": str(output_path)}
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        while True:
            message = self._read_message(self.request_timeout)
            if message.get("request_id") == request_id and message.get("type") == "response":
                break
        payload = message.get("payload")
        if not isinstance(payload, dict):
            raise FrozenBranchError(f"冻结 worker 返回异常：{message}")
        return payload

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.stdin:
                self.process.stdin.write(json.dumps(
                    {"protocol": PROTOCOL_NAME, "type": "request",
                     "request_id": "shutdown", "action": "shutdown"}) + "\n")
                self.process.stdin.flush()
            self.process.wait(timeout=30)
        except Exception:  # noqa: BLE001
            try:
                self.process.kill()
            except Exception:  # noqa: BLE001
                pass
        finally:
            self.process = None
