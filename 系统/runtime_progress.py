"""Startup-only metadata events for persistent workers (no model operations)."""
from __future__ import annotations

import json
import os
import sys
import threading
import time

PROTOCOL_NAME = "face-source-trace-runtime-v1"
STARTUP_STAGES = frozenset({"importing", "checking_model", "loading_model", "checking_library",
                            "rebuilding_index", "warming_up"})
_output = sys.stdout
_started = time.perf_counter()
_sequence = 0
_lock = threading.Lock()


def emit_startup_progress(stage: str, message: str, *, status: str = "running", details=None) -> None:
    """Report an actual operation; a skipped warm-up must remain skipped."""
    if stage not in STARTUP_STAGES or status not in {"running", "completed", "skipped"}:
        raise ValueError("unsupported startup stage or status")
    global _sequence
    with _lock:
        _sequence += 1
        payload = {"protocol": PROTOCOL_NAME, "type": "startup_progress", "request_id": None,
                   "ready": False, "health": "starting", "stage": stage, "stage_status": status,
                   "message": str(message), "pid": os.getpid(), "sequence": _sequence,
                   "worker_elapsed_seconds": max(0.0, time.perf_counter() - _started),
                   "details": dict(details or {})}
        _output.write("\n" + json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        _output.flush()


def report_no_separate_warmup() -> None:
    emit_startup_progress("warming_up", "现有启动路径没有单独执行预热推理。", status="skipped",
                          details={"warmup_status": "not_performed"})


class StartupIndexReporter:
    """Translate existing identity-index events; discard paths/identity labels."""
    _MAP = {"check_identity_library": "checking_library", "identity_library_checked": "checking_library",
            "load_identity_cache": "checking_library", "identity_cache_ready": "checking_library",
            "build_identity_cache": "rebuilding_index"}
    _DETAILS = frozenset({"cache_status", "library_image_count", "library_identity_count", "valid_embedding_count"})

    def emit(self, stage, status, message, **kwargs):
        mapped = self._MAP.get(stage)
        if mapped is None:
            return
        original = kwargs.get("details") or {}
        details = {key: value for key, value in original.items() if key in self._DETAILS}
        for key in ("progress_current", "progress_total"):
            value = kwargs.get(key)
            if type(value) is int and value >= 0:
                details[key] = value
        emit_startup_progress(mapped, str(message), status="completed" if status == "success" else "running", details=details)
