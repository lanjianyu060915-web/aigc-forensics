"""Persistent JSON-lines host for the frozen InsightFace identity searcher."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, TextIO

from identity_search.cli import (
    _new_searcher,
    _write_json_atomic,
    check_required_dependencies,
    error_result,
    resolve_device,
)
from identity_search.errors import IdentitySearchError, InputValidationError
from identity_search.model import default_model_root, preflight_model
from progress_events import create_progress_writer


PROTOCOL_NAME = "face-source-trace-runtime-v1"
PROTOCOL_OUTPUT: TextIO = sys.stdout


def _safe_progress(
    reporter, stage: str, status: str, message: str, **kwargs: Any
) -> None:
    if reporter is None:
        return
    try:
        reporter.emit(stage, status, message, **kwargs)
    except Exception:
        return


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
    parser = argparse.ArgumentParser(description="InsightFace 身份检索持久工作进程")
    parser.add_argument("--library", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model-root", default=str(default_model_root()))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--ctx", type=int, default=-1)
    return parser


class PersistentIdentityRuntime:
    """Reuse one InsightFace extractor and cached per-library search indexes."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.cache_dir = Path(args.cache_dir).expanduser().resolve(strict=False)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device, self.ctx_id = resolve_device(args.device, args.ctx)
        check_required_dependencies()
        self.model = preflight_model(args.model_root, verify_hashes=True)

        initial_library = Path(args.library).expanduser().resolve(strict=False)
        if not initial_library.is_dir():
            raise InputValidationError(f"身份库目录不存在: {initial_library}")
        initial_searcher = _new_searcher(
            library_path=initial_library,
            cache_dir=self.cache_dir,
            model_root=self.model.root,
            model_fingerprint=self.model.fingerprint,
            device=self.device,
            ctx_id=self.ctx_id,
        )
        self._searchers: dict[str, Any] = {
            self._library_key(initial_library): initial_searcher
        }
        self._extractor = initial_searcher.extractor

    @staticmethod
    def _library_key(library: Path) -> str:
        return os.path.normcase(str(library.resolve(strict=False)))

    def _searcher_for(self, library_value: Any, progress_reporter=None):
        library = Path(str(library_value)).expanduser().resolve(strict=False)
        if not library.is_dir():
            raise InputValidationError(f"身份库目录不存在: {library}")
        key = self._library_key(library)
        searcher = self._searchers.get(key)
        if searcher is not None:
            # Preserve the frozen CLI's per-query library freshness semantics.
            # The scan/fingerprint methods are the existing algorithm's own
            # deterministic checks; only the expensive extractor stays reused.
            current_manifest = searcher._scan_library_manifest()
            current_fingerprint = searcher._index_fingerprint(current_manifest)
            if current_fingerprint == searcher._fingerprint:
                _safe_progress(
                    progress_reporter,
                    "identity_cache_ready",
                    "success",
                    "身份库特征缓存已校验并复用。",
                    progress_percent=None,
                    details={
                        "cache_status": searcher.diagnostics.get("cache_status"),
                        "library_identity_count": len(searcher.identities),
                        "library_image_count": len(current_manifest),
                        "valid_embedding_count": sum(
                            len(data["features"])
                            for data in searcher.identities.values()
                        ),
                    },
                    force=True,
                )
                return searcher

        # A different library needs a separate deterministic index, but injecting
        # the already-created extractor keeps the same InsightFace model instance.
        searcher = _new_searcher(
            library_path=library,
            cache_dir=self.cache_dir,
            model_root=self.model.root,
            model_fingerprint=self.model.fingerprint,
            device=self.device,
            ctx_id=self.ctx_id,
            extractor=self._extractor,
            progress_reporter=progress_reporter,
        )
        self._searchers[key] = searcher
        return searcher

    def health_details(self) -> dict[str, Any]:
        return {
            "worker": "identity",
            "pid": os.getpid(),
            "model": "buffalo_l",
            "model_fingerprint": self.model.fingerprint,
            "device": self.device,
            "ctx_id": self.ctx_id,
            "cached_libraries": len(self._searchers),
            "model_instances": 1,
            "dependency_isolation": "dedicated_process",
        }

    def run(
        self,
        query_value: Any,
        output_value: Any,
        library_value: Any,
        top_k_value: Any,
        task_id: Any = None,
        progress_events_path: Any = None,
        progress_latest_path: Any = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        query = Path(str(query_value)).expanduser().resolve(strict=False)
        output = Path(str(output_value)).expanduser().resolve(strict=False)
        reporter = None
        searcher = None
        try:
            reporter = create_progress_writer(
                task_id=(str(task_id) if task_id is not None else None),
                module="identity",
                events_path=(
                    str(progress_events_path)
                    if progress_events_path is not None
                    else None
                ),
                latest_path=(
                    str(progress_latest_path)
                    if progress_latest_path is not None
                    else None
                ),
            )
            _safe_progress(
                reporter,
                "validate_identity_input",
                "running",
                "正在校验身份搜索输入与配置。",
                progress_percent=None,
                details={"query_exists": query.is_file()},
                force=True,
            )
            if not query.is_file():
                raise InputValidationError(f"查询图片不存在或不是文件: {query}")
            if isinstance(top_k_value, bool):
                raise InputValidationError(f"top-k 必须为正整数，实际值: {top_k_value}")
            top_k = int(top_k_value)
            if top_k <= 0:
                raise InputValidationError(f"top-k 必须为正整数，实际值: {top_k}")
            searcher = self._searcher_for(library_value, reporter)
            result = searcher.search(
                query,
                top_k=top_k,
                progress_reporter=reporter,
                artifact_root=output.parent,
                task_id=(str(task_id) if task_id is not None else None),
            )
            result["runtime_seconds"] = time.perf_counter() - started
            if isinstance(result.get("runtime"), dict):
                result["runtime"]["seconds"] = result["runtime_seconds"]
        except IdentitySearchError as exc:
            diagnostic_values = getattr(exc, "diagnostics", {})
            result = error_result(
                str(query),
                str(exc),
                started,
                diagnostics=(
                    dict(diagnostic_values)
                    if isinstance(diagnostic_values, dict)
                    else {}
                ),
                searcher=searcher,
                device=self.device,
                ctx_id=self.ctx_id,
            )
            _safe_progress(
                reporter,
                "identity_failed",
                "error",
                str(exc),
                progress_percent=None,
                details=result["metrics"],
                force=True,
            )
        except Exception as exc:
            result = error_result(
                str(query),
                f"搜索过程异常: {type(exc).__name__}: {exc}",
                started,
                searcher=searcher,
                device=self.device,
                ctx_id=self.ctx_id,
            )
            _safe_progress(
                reporter,
                "identity_failed",
                "error",
                result["error_message"],
                progress_percent=None,
                details=result["metrics"],
                force=True,
            )
        try:
            _write_json_atomic(output, result)
        except Exception as exc:
            original = result.get("error_message")
            write_error = f"无法写入输出 JSON {output}: {type(exc).__name__}: {exc}"
            result = dict(result)
            result.update(
                status="error",
                candidates=[],
                error_message=f"{original}；{write_error}" if original else write_error,
            )
        return result


def _request_id(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    value = message.get("request_id")
    return value if isinstance(value, str) and value else None


def _serve(runtime: PersistentIdentityRuntime) -> int:
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
            payload = runtime.run(
                message.get("query_path"),
                message.get("output_path"),
                message.get("library_path"),
                message.get("top_k"),
                message.get("task_id"),
                message.get("progress_events_path"),
                message.get("progress_latest_path"),
            )
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
                error=f"身份持久请求异常：{type(exc).__name__}: {exc}",
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        runtime = PersistentIdentityRuntime(args)
    except Exception as exc:
        _emit(
            "ready",
            request_id=None,
            ready=False,
            health="error",
            ok=False,
            error=f"身份模型初始化失败：{type(exc).__name__}: {exc}",
            details={"worker": "identity", "pid": os.getpid()},
        )
        return 1
    return _serve(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
