"""Persistent JSON-lines host for the frozen deepfake adapter.

The installed detector remains authoritative.  This file only keeps one
``DeepfakeDetectionPipeline`` object alive and maps each native result through
the same frozen helper functions used by ``run_deepfake_detection.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, TextIO

from pyinstaller_pyz_runtime import PyInstallerArchiveError
from run_deepfake_detection import (
    DEFAULT_INSTALL_ROOT,
    _base_payload,
    _attach_visual_evidence,
    _classify_error,
    _configure_installed_runtime,
    _map_native_result,
    _validate_resources,
    _write_json,
)


PROTOCOL_NAME = "face-source-trace-runtime-v1"
PROTOCOL_OUTPUT: TextIO = sys.stdout


def _emit(
    message_type: str,
    *,
    request_id: str | None,
    ready: bool,
    health: str,
    **fields: Any,
) -> None:
    message = {
        "protocol": PROTOCOL_NAME,
        "type": message_type,
        "request_id": request_id,
        "ready": bool(ready),
        "health": str(health),
        **fields,
    }
    # Prefixing a newline prevents a third-party write without a trailing newline
    # from being concatenated with and corrupting the protocol object.
    PROTOCOL_OUTPUT.write("\n" + json.dumps(message, ensure_ascii=False) + "\n")
    PROTOCOL_OUTPUT.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="冻结鉴伪模型持久工作进程")
    parser.add_argument("--install-root", default=str(DEFAULT_INSTALL_ROOT))
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument(
        "--writable-root",
        default=str(Path(__file__).resolve().parents[2] / "runtime" / "persistent_services" / "deepfake"),
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="仅构造 Pipeline，第一次查询时再调用原模型加载逻辑",
    )
    return parser


class PersistentDeepfakeRuntime:
    """Own exactly one frozen Pipeline instance for this worker process."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.install_root = Path(args.install_root).expanduser().resolve(strict=False)
        self.writable_root = Path(args.writable_root).expanduser().resolve(strict=False)
        self.writable_root.mkdir(parents=True, exist_ok=True)
        executable, self.resources = _validate_resources(self.install_root)
        self._dll_handles = _configure_installed_runtime(
            self.install_root,
            executable,
            self.writable_root,
        )

        # Imported only after the frozen PyInstaller PYZ importer and DLL search
        # paths are installed.  No detector code or threshold is reimplemented.
        from src.pipeline import DeepfakeDetectionPipeline

        initial_output = self.writable_root / "runs"
        self.pipeline = DeepfakeDetectionPipeline(
            project_root=self.install_root,
            device=args.device,
            output_root=initial_output,
            writable_root=self.writable_root,
        )
        if not args.no_preload:
            ensure_loaded = getattr(self.pipeline, "_ensure_model_loaded", None)
            if not callable(ensure_loaded):
                raise RuntimeError("冻结 Pipeline 缺少模型预加载入口 _ensure_model_loaded")
            ensure_loaded()

    def health_details(self) -> dict[str, Any]:
        return {
            "worker": "deepfake",
            "pid": os.getpid(),
            "pipeline_initialized": self.pipeline is not None,
            "model_load_count": int(getattr(self.pipeline, "model_load_count", 0)),
            "install_root": str(self.install_root),
            "official_base_weight_required": False,
            "spatial_initialization": "mobilenet_v3_small(weights=None) + 项目完整checkpoint(strict=True)",
            "dependency_isolation": "dedicated_process",
        }

    def run(self, query_value: Any, output_value: Any) -> dict[str, Any]:
        started = time.perf_counter()
        query = Path(str(query_value)).expanduser().resolve(strict=False)
        output = Path(str(output_value)).expanduser().resolve(strict=False)
        payload = _base_payload(query, 0.0)
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            if not query.is_file():
                raise RuntimeError(f"query 不存在或不是文件：{query}")

            # The frozen Pipeline stores output locations on the instance.  The
            # protocol is strictly serial, so updating only these I/O roots lets
            # each task retain the frozen CLI layout without reloading models.
            native_output_root = (output.parent / "visualizations" / "deepfake").resolve(
                strict=False
            )
            self.pipeline.writable_root = output.parent.resolve(strict=False)
            self.pipeline.output_root = native_output_root
            self.pipeline._require_inside_writable_root(
                native_output_root, "运行结果目录"
            )
            native = self.pipeline.detect_media(query)
            payload = _map_native_result(
                native,
                query,
                time.perf_counter() - started,
                self.install_root,
            )
            _attach_visual_evidence(
                pipeline=self.pipeline,
                native=native,
                query=query,
                output=output,
                payload=payload,
            )
        except (PyInstallerArchiveError, OSError, RuntimeError, ValueError, ImportError) as exc:
            payload = _base_payload(query, time.perf_counter() - started)
            failure = _classify_error(exc)
            payload["error_message"] = failure["user_message"]
            payload["details"].update(failure)
        payload["runtime_seconds"] = time.perf_counter() - started
        _write_json(output, payload)
        return payload


def _request_id(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    value = message.get("request_id")
    return value if isinstance(value, str) and value else None


def _serve(runtime: PersistentDeepfakeRuntime) -> int:
    _emit(
        "ready",
        request_id=None,
        ready=True,
        health="ok",
        ok=True,
        details=runtime.health_details(),
    )
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            # Input noise cannot be correlated safely and is intentionally ignored.
            continue
        request_id = _request_id(message)
        if (
            not isinstance(message, dict)
            or message.get("protocol") != PROTOCOL_NAME
            or message.get("type") != "request"
            or request_id is None
        ):
            continue
        action = message.get("action")
        if action == "health":
            _emit(
                "health",
                request_id=request_id,
                ready=True,
                health="ok",
                ok=True,
                details=runtime.health_details(),
            )
            continue
        if action == "shutdown":
            _emit(
                "response",
                request_id=request_id,
                ready=True,
                health="ok",
                ok=True,
                shutting_down=True,
            )
            return 0
        if action != "run":
            _emit(
                "response",
                request_id=request_id,
                ready=True,
                health="ok",
                ok=False,
                error=f"不支持的 action：{action!r}",
            )
            continue
        try:
            payload = runtime.run(message.get("query_path"), message.get("output_path"))
            _emit(
                "response",
                request_id=request_id,
                ready=True,
                health="ok",
                ok=True,
                payload=payload,
            )
        except Exception as exc:
            _emit(
                "response",
                request_id=request_id,
                ready=True,
                health="ok",
                ok=False,
                error=f"鉴伪持久请求异常：{type(exc).__name__}: {exc}",
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        runtime = PersistentDeepfakeRuntime(args)
    except Exception as exc:
        failure = _classify_error(exc)
        _emit(
            "ready",
            request_id=None,
            ready=False,
            health="error",
            ok=False,
            error=failure["user_message"],
            details={
                "worker": "deepfake",
                "pid": os.getpid(),
                **failure,
            },
        )
        return 1
    return _serve(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
