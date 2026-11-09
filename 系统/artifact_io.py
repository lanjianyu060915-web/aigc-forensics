"""Bounded atomic writes and explicit recovery plans; never execute recovery.

Audit records contain path roles and OS error codes, not file contents. Callers
may pass a list to retain every write outcome and persist it in their task log.
Immutable receipts use an atomic hard-link publication and refuse replacement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable
import uuid


WINDOWS_RETRY_CODES = frozenset({5, 32, 33})
MAX_REPLACE_ATTEMPTS = 6


def _error_fields(exc: BaseException) -> dict[str, Any]:
    return {"error_type": type(exc).__name__, "errno": getattr(exc, "errno", None),
            "winerror": getattr(exc, "winerror", None)}


def _publish(temporary: Path, destination: Path, *, role: str,
             operation: Callable, sleep: Callable, audit: list | None) -> dict:
    failures = []
    for attempt in range(1, MAX_REPLACE_ATTEMPTS + 1):
        try:
            operation(temporary, destination)
        except OSError as exc:
            failures.append({"attempt": attempt, **_error_fields(exc)})
            retryable = isinstance(exc, PermissionError) and getattr(exc, "winerror", None) in WINDOWS_RETRY_CODES
            if retryable and attempt < MAX_REPLACE_ATTEMPTS:
                sleep(0.01 * 2 ** (attempt - 1))
                continue
            record = {"status": "failed", "path_role": role, "phase": "publish",
                      "attempts": attempt, "failures": failures}
            if audit is not None:
                audit.append(record)
            exc.artifact_io_record = record
            raise
        record = {"status": "success", "path_role": role, "phase": "publish",
                  "attempts": attempt, "failures": failures}
        if audit is not None:
            audit.append(record)
        return record
    raise AssertionError("unreachable")


def replace_with_retry(temporary: str | Path, destination: str | Path, *,
                       role: str = "artifact", audit: list | None = None,
                       replace_fn: Callable | None = None,
                       sleep_fn: Callable | None = None) -> dict:
    """Preserve the legacy 6 attempts and 10/20/40/80/160 ms delays."""
    return _publish(Path(temporary), Path(destination), role=role,
                    operation=replace_fn or os.replace, sleep=sleep_fn or time.sleep, audit=audit)


def atomic_write_bytes(path: str | Path, data: bytes, *, role: str = "artifact",
                       overwrite: bool = True, audit: list | None = None) -> dict:
    """Serialize first, fsync a same-directory temporary, then publish atomically.

    ``overwrite=False`` requires hard-link support on the destination filesystem;
    failure is explicit and is never downgraded to an unsafe overwrite.
    """
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    destination = Path(path)
    # Do not repeat a long receipt/report filename: a valid Windows destination
    # can otherwise gain 50+ characters and fail only at its temporary sibling.
    temporary = destination.with_name(f".atomic-{uuid.uuid4().hex}.tmp")
    record = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        record = _publish(temporary, destination, role=role,
                          operation=os.replace if overwrite else os.link,
                          sleep=time.sleep, audit=audit)
        record.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), overwrite=overwrite)
        return record
    except OSError as exc:
        if not hasattr(exc, "artifact_io_record"):
            record = {"status": "failed", "path_role": role, "phase": "write_temporary",
                      "attempts": 0, "failures": [_error_fields(exc)]}
            exc.artifact_io_record = record
            if audit is not None:
                audit.append(record)
        else:
            record = exc.artifact_io_record
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            # Do not hide a publish error, or claim that a successful publication
            # failed merely because its leftover temporary could not be removed.
            cleanup = {"status": "failed", "phase": "cleanup", **_error_fields(exc)}
            if record is not None:
                record["cleanup"] = cleanup
            elif audit is not None:
                audit.append({"path_role": role, **cleanup})


def atomic_write_text(path: str | Path, text: str, *, role: str = "text",
                      overwrite: bool = True, audit: list | None = None,
                      encoding: str = "utf-8") -> dict:
    try:
        data = text.encode(encoding)
    except (UnicodeError, LookupError, AttributeError) as exc:
        record = {"status": "failed", "path_role": role, "phase": "serialize",
                  "attempts": 0, "failures": [_error_fields(exc)]}
        exc.artifact_io_record = record
        if audit is not None:
            audit.append(record)
        raise
    return atomic_write_bytes(path, data, role=role, overwrite=overwrite, audit=audit)


def atomic_write_json(path: str | Path, value: Any, *, role: str = "json",
                      overwrite: bool = True, audit: list | None = None,
                      allow_nan: bool = False, indent: int | None = 2,
                      sort_keys: bool = False) -> dict:
    # A serialization failure leaves the destination and its directory untouched.
    try:
        text = json.dumps(value, ensure_ascii=False, indent=indent, allow_nan=allow_nan, sort_keys=sort_keys) + "\n"
    except (ValueError, TypeError) as exc:
        record = {"status": "failed", "path_role": role, "phase": "serialize",
                  "attempts": 0, "failures": [_error_fields(exc)]}
        exc.artifact_io_record = record
        if audit is not None:
            audit.append(record)
        raise
    return atomic_write_text(path, text, role=role, overwrite=overwrite, audit=audit)


STAGES = frozenset({"inference", "decision", "report_ready", "report", "receipt", "cloud"})
RECOVERY_ACTIONS = frozenset({"write_receipt", "render_report", "retry_cloud", "rerun_inference"})


def write_stage_receipt(directory: str | Path, *, task_id: str, attempt_id: str,
                        stage: str, status: str, input_digest: str,
                        config_digest: str, artifacts: Iterable[dict] = (),
                        audit: list | None = None) -> Path:
    """Write a new immutable attempt receipt; caller verifies artifact bindings.

    The helper never reads artifact paths and does not certify their integrity.
    A failed stage must use a fresh attempt_id before another attempt is recorded.
    """
    if stage not in STAGES or status not in {"completed", "failed"}:
        raise ValueError("invalid stage or status")
    for name, value in (("task_id", task_id), ("attempt_id", attempt_id)):
        if (not isinstance(value, str) or not value.strip() or value in {".", ".."}
                or value != value.strip() or any(c in value for c in '/\\:*?"<>|')
                or any(ord(c) < 32 for c in value) or value.endswith(".")):
            raise ValueError(f"invalid {name}")
    for digest in (input_digest, config_digest):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError("bindings require complete SHA-256 digests")
    path = Path(directory) / f"{task_id}.{attempt_id}.{stage}.json"
    value = {"schema_version": 1, "task_id": task_id, "attempt_id": attempt_id,
             "stage": stage, "status": status, "input_digest": input_digest.lower(),
             "config_digest": config_digest.lower(), "artifacts": list(artifacts),
             "created_at": datetime.now(timezone.utc).isoformat()}
    atomic_write_json(path, value, role="stage_receipt", overwrite=False, audit=audit)
    return path


@dataclass(frozen=True)
class RecoveryPlan:
    action: str
    allowed: bool
    reason: str
    run_model: bool = False
    call_cloud: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def plan_recovery(action: str, *, verified_stages: Iterable[str], bindings_match: bool,
                  authorized: bool, stop_requested: bool = False,
                  cloud_attempts: int = 0, max_cloud_attempts: int = 2) -> RecoveryPlan:
    """Decide one explicitly requested action; never run models or auto-resume.

    The caller must verify artifacts and input/config digests before marking a
    stage verified. Report repair has no model/cloud fallback. Cloud retries are
    writer-only here and require an already verified local decision.
    """
    if action not in RECOVERY_ACTIONS:
        raise ValueError("unknown recovery action")
    if any(type(v) is not bool for v in (bindings_match, authorized, stop_requested)):
        raise ValueError("authorization, stop and binding flags must be booleans")
    stages = frozenset(verified_stages)
    if not stages <= STAGES:
        raise ValueError("unknown stage")
    if any(type(v) is not int or v < 0 for v in (cloud_attempts, max_cloud_attempts)):
        raise ValueError("cloud attempt counts must be non-negative integers")
    if not authorized:
        return RecoveryPlan(action, False, "current_authorization_required")
    if stop_requested:
        return RecoveryPlan(action, False, "stop_requested")
    if not bindings_match:
        return RecoveryPlan(action, False, "input_or_config_binding_mismatch")
    if action == "rerun_inference":
        if "inference" in stages:
            return RecoveryPlan(action, False, "inference_already_verified")
        if stages & {"decision", "report", "receipt"}:
            return RecoveryPlan(action, False, "verified_outputs_require_new_task")
        return RecoveryPlan(action, True, "explicit_inference_attempt", run_model=True)
    if "decision" not in stages:
        return RecoveryPlan(action, False, "verified_decision_required")
    target = {"write_receipt": "receipt", "render_report": "report", "retry_cloud": "cloud"}[action]
    if target in stages:
        return RecoveryPlan(action, False, "stage_already_verified")
    if action == "retry_cloud":
        if cloud_attempts >= max_cloud_attempts:
            return RecoveryPlan(action, False, "cloud_attempt_budget_exhausted")
        return RecoveryPlan(action, True, "explicit_writer_retry", call_cloud=True)
    return RecoveryPlan(action, True, "reuse_verified_decision")
