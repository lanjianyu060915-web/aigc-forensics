"""Truthful, process-safe progress events shared by the v0.3 modules.

The writer deliberately treats ``progress_percent=None`` as an indeterminate
progress value.  Callers should prefer real item counts and omit a percentage
when the amount of remaining work cannot be measured without inventing one.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any
import uuid


ALLOWED_MODULES = frozenset({"deepfake", "identity", "source", "video", "overall"})
ALLOWED_STATUSES = frozenset(
    {"pending", "running", "success", "warning", "error", "skipped"}
)
AUTO_PERCENT = object()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class _CrossProcessLock:
    """Small lock-file wrapper used for the JSONL append and latest replace."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream = None

    def __enter__(self) -> "_CrossProcessLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.stream.fileno(), msvcrt.LK_LOCK, 1)
        else:  # pragma: no cover - the product target is Windows
            import fcntl

            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self.stream is not None
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - the product target is Windows
            import fcntl

            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        self.stream = None


class ProgressEventWriter:
    """Append unified progress events and atomically publish the latest event.

    One instance is intended for one task/module execution.  Multiple module
    processes may safely target the same files because both writes share a
    process-wide lock file.
    """

    def __init__(
        self,
        task_id: str,
        module: str,
        events_path: str | Path,
        latest_path: str | Path,
        min_interval_seconds: float = 0.5,
        min_percent_delta: float = 5.0,
    ) -> None:
        task_text = str(task_id).strip()
        module_text = str(module).strip().lower()
        if not task_text:
            raise ValueError("task_id must be a non-empty string")
        if module_text not in ALLOWED_MODULES:
            raise ValueError(f"unsupported progress module: {module_text!r}")
        if not math.isfinite(float(min_interval_seconds)) or min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be a non-negative finite number")
        if not math.isfinite(float(min_percent_delta)) or min_percent_delta < 0:
            raise ValueError("min_percent_delta must be a non-negative finite number")

        self.task_id = task_text
        self.module = module_text
        self.events_path = Path(events_path).expanduser().resolve(strict=False)
        self.latest_path = Path(latest_path).expanduser().resolve(strict=False)
        self.lock_path = self.events_path.with_name("progress_events.lock")
        self.min_interval_seconds = float(min_interval_seconds)
        self.min_percent_delta = float(min_percent_delta)
        self._started = time.perf_counter()
        self._elapsed_offset = 0.0
        initial = _read_json_object(self.latest_path)
        if initial and initial.get("task_id") == self.task_id:
            try:
                self._elapsed_offset = max(0.0, float(initial.get("elapsed_seconds", 0.0)))
            except (TypeError, ValueError):
                self._elapsed_offset = 0.0
        self._thread_lock = threading.RLock()
        self._last_emit_monotonic: float | None = None
        self._last_stage: str | None = None
        self._last_status: str | None = None
        self._last_percent: float | None = None
        if (
            initial
            and initial.get("task_id") == self.task_id
            and initial.get("module") == self.module
        ):
            initial_percent = initial.get("progress_percent")
            if isinstance(initial_percent, (int, float)) and not isinstance(
                initial_percent, bool
            ):
                self._last_percent = float(initial_percent)
        self._last_elapsed = self._elapsed_offset

    @staticmethod
    def _validate_count(value: Any, label: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer or None")
        return int(value)

    def _resolve_percent(
        self,
        value: Any,
        current: int | None,
        total: int | None,
        details: dict[str, Any],
    ) -> float | None:
        if value is AUTO_PERCENT:
            if current is None or total is None or total <= 0:
                percent = None
            else:
                if current > total:
                    raise ValueError("progress_current cannot exceed progress_total")
                percent = 100.0 * current / total
        elif value is None:
            percent = None
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("progress_percent must be numeric, None, or AUTO_PERCENT")
            percent = float(value)
            if not math.isfinite(percent) or not 0.0 <= percent <= 100.0:
                raise ValueError("progress_percent must be between 0 and 100")

        if percent is not None:
            percent = round(float(percent), 6)
            if self._last_percent is not None and percent < self._last_percent:
                details["reported_progress_percent"] = percent
                details["progress_percent_omitted_reason"] = "would_decrease"
                percent = None
        return percent

    def _should_emit(
        self,
        *,
        stage: str,
        status: str,
        current: int | None,
        total: int | None,
        percent: float | None,
        now_monotonic: float,
        force: bool,
    ) -> bool:
        if force or status in {"success", "warning", "error", "skipped"}:
            return True
        if self._last_emit_monotonic is None:
            return True
        if stage != self._last_stage or status != self._last_status:
            return True
        if current is not None and total is not None and current == total:
            return True
        if (
            percent is not None
            and self._last_percent is not None
            and percent - self._last_percent >= self.min_percent_delta
        ):
            return True
        return now_monotonic - self._last_emit_monotonic >= self.min_interval_seconds

    @staticmethod
    def _monotonic_timestamp(previous: dict[str, Any] | None) -> str:
        current = datetime.now().astimezone()
        # The persisted representation has millisecond precision.  Truncate
        # before comparing so two distinct microsecond values cannot serialize
        # to the same timestamp.
        current = current.replace(microsecond=(current.microsecond // 1000) * 1000)
        if previous:
            try:
                prior = datetime.fromisoformat(str(previous.get("timestamp", "")))
            except (TypeError, ValueError):
                prior = None
            if prior is not None and current <= prior:
                current = prior + timedelta(milliseconds=1)
        return current.isoformat(timespec="milliseconds")

    def emit(
        self,
        stage: str,
        status: str,
        message: str,
        *,
        progress_current: int | None = None,
        progress_total: int | None = None,
        progress_percent: Any = AUTO_PERCENT,
        details: Mapping[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        stage_text = str(stage).strip()
        status_text = str(status).strip().lower()
        message_text = str(message).strip()
        if not stage_text:
            raise ValueError("stage must be a non-empty string")
        if status_text not in ALLOWED_STATUSES:
            raise ValueError(f"unsupported progress status: {status_text!r}")
        if not message_text:
            raise ValueError("message must be a non-empty string")

        current = self._validate_count(progress_current, "progress_current")
        total = self._validate_count(progress_total, "progress_total")
        if current is not None and total is None:
            raise ValueError("progress_total is required when progress_current is set")
        if total is not None and current is None:
            raise ValueError("progress_current is required when progress_total is set")
        if current is not None and total is not None and current > total:
            raise ValueError("progress_current cannot exceed progress_total")

        with self._thread_lock:
            detail_object = _json_safe(dict(details or {}))
            percent = self._resolve_percent(
                progress_percent, current, total, detail_object
            )
            now_monotonic = time.perf_counter()
            if not self._should_emit(
                stage=stage_text,
                status=status_text,
                current=current,
                total=total,
                percent=percent,
                now_monotonic=now_monotonic,
                force=force,
            ):
                return None

            elapsed = max(
                self._last_elapsed,
                self._elapsed_offset + now_monotonic - self._started,
            )
            with _CrossProcessLock(self.lock_path):
                previous = _read_json_object(self.latest_path)
                if previous and previous.get("task_id") == self.task_id:
                    try:
                        elapsed = max(elapsed, float(previous.get("elapsed_seconds", 0.0)))
                    except (TypeError, ValueError):
                        pass
                if (
                    percent is not None
                    and previous
                    and previous.get("task_id") == self.task_id
                    and previous.get("module") == self.module
                ):
                    previous_percent = previous.get("progress_percent")
                    if (
                        isinstance(previous_percent, (int, float))
                        and not isinstance(previous_percent, bool)
                        and percent < float(previous_percent)
                    ):
                        detail_object["reported_progress_percent"] = percent
                        detail_object[
                            "progress_percent_omitted_reason"
                        ] = "would_decrease_across_processes"
                        percent = None
                event = {
                    "event_id": uuid.uuid4().hex,
                    "timestamp": self._monotonic_timestamp(previous),
                    "task_id": self.task_id,
                    "module": self.module,
                    "stage": stage_text,
                    "status": status_text,
                    "message": message_text,
                    "progress_current": current,
                    "progress_total": total,
                    "progress_percent": percent,
                    "elapsed_seconds": round(max(0.0, elapsed), 6),
                    "details": detail_object,
                }
                self.events_path.parent.mkdir(parents=True, exist_ok=True)
                line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                with self.events_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(line + "\n")
                    stream.flush()

                self.latest_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.latest_path.with_name(
                    f".{self.latest_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
                )
                try:
                    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                        json.dump(event, stream, ensure_ascii=False, indent=2)
                        stream.write("\n")
                        stream.flush()
                    os.replace(temporary, self.latest_path)
                finally:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass

            self._last_emit_monotonic = now_monotonic
            self._last_stage = stage_text
            self._last_status = status_text
            self._last_elapsed = elapsed
            if percent is not None:
                self._last_percent = percent
            return event


def create_progress_writer(
    *,
    task_id: str | None,
    module: str,
    events_path: str | Path | None,
    latest_path: str | Path | None,
    min_interval_seconds: float = 0.5,
    min_percent_delta: float = 5.0,
) -> ProgressEventWriter | None:
    """Return a writer when all progress arguments are present, otherwise None."""

    values = (task_id, events_path, latest_path)
    if all(value is None or str(value).strip() == "" for value in values):
        return None
    if any(value is None or str(value).strip() == "" for value in values):
        raise ValueError(
            "task_id, progress_events_path and progress_latest_path must be provided together"
        )
    return ProgressEventWriter(
        str(task_id),
        module,
        Path(str(events_path)),
        Path(str(latest_path)),
        min_interval_seconds=min_interval_seconds,
        min_percent_delta=min_percent_delta,
    )


__all__ = [
    "ALLOWED_MODULES",
    "ALLOWED_STATUSES",
    "AUTO_PERCENT",
    "ProgressEventWriter",
    "create_progress_writer",
]
