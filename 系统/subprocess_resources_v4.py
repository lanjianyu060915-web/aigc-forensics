"""Bounded subprocess execution with explicit single-PID resource observations.

Only a process created by this call is eligible for timeout cleanup. Descendant
memory is not included in the PID sampler. Windows taskkill targets that owned
live PID's tree; uncertain cleanup is recorded, never labelled successful.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from resource_monitor import ResourceMonitor

_WINDOWS = os.name == "nt"
_TASKKILL = Path("C:/Windows/System32/taskkill.exe")


def _terminate_owned_process_tree(process, *, cleanup_timeout=2.0):
    result = {"owned_pid": process.pid, "scope": "created_child_process_tree_only", "status": "unknown"}
    if type(process.pid) is not int or process.pid <= 0 or process.pid == os.getpid():
        return {**result, "reason": "invalid_owned_child_pid"}
    if _WINDOWS:
        if process.poll() is not None:
            # The root is gone; /T cannot establish which descendants remain.
            return {**result, "reason": "root_exited_descendants_unverified"}
        try:
            if not _TASKKILL.is_file():
                raise FileNotFoundError("fixed_taskkill_unavailable")
            completed = subprocess.run([str(_TASKKILL), "/PID", str(process.pid), "/T", "/F"],
                                       capture_output=True, timeout=cleanup_timeout, check=False,
                                       shell=False, creationflags=0x08000000)
            result.update(method="fixed_taskkill_owned_pid_tree", returncode=completed.returncode)
            if completed.returncode == 0:
                result["status"] = "tree_termination_requested"
            else:
                result["reason"] = "tree_termination_command_failed"
        except Exception as exc:
            result["reason"] = "tree_termination_" + type(exc).__name__
    else:
        try:
            # start_new_session below makes the owned child's PID its group ID.
            os.killpg(process.pid, signal.SIGKILL)
            result.update(status="tree_termination_requested", method="owned_process_group_sigkill")
        except ProcessLookupError:
            result.update(status="tree_termination_requested", method="owned_process_group_already_gone")
        except Exception as exc:
            result["reason"] = "process_group_termination_" + type(exc).__name__
    if process.poll() is None:
        try:
            process.kill()
            result["owned_root_kill_requested"] = True
        except Exception as exc:
            result["root_kill_error_type"] = type(exc).__name__
    return result


def run_monitored_subprocess(command, *, cwd, timeout, env, hide_console=False,
                             audit_callback=None, monitor_factory=None, cleanup_timeout=2.0):
    """Return CompletedProcess plus resource_audit; timeout retains the same API.

    Cleanup and post-kill pipe drain each have a separate bounded allowance.
    An unresolved pipe drain is left to Python's daemon pipe readers, avoiding
    an unbounded close/join in the calling thread; the audit records this limit.
    """
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("subprocess_timeout_must_be_positive_finite")
    if isinstance(cleanup_timeout, bool) or not math.isfinite(cleanup_timeout) or cleanup_timeout <= 0:
        raise ValueError("cleanup_timeout_must_be_positive_finite")
    started = time.perf_counter()
    audit = {"schema_version": "subprocess_resources.v1", "attempt_id": uuid.uuid4().hex,
             "pid": None, "status": "launch_failed", "resource_scope": "created_child_pid_excludes_descendants",
             "timeout_seconds": float(timeout), "cleanup_timeout_seconds": float(cleanup_timeout),
             "returncode": None, "resources": None, "tree_cleanup": {"status": "not_requested"}}
    process = monitor = None
    result = failure = None
    try:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace", env=env, shell=False,
                                   creationflags=0x08000000 if _WINDOWS and hide_console else 0,
                                   start_new_session=not _WINDOWS)
        audit["pid"] = process.pid
        try:
            monitor = (monitor_factory or ResourceMonitor)(pid=process.pid)
            monitor.start()
        except Exception as exc:
            audit["resource_monitor_error_type"] = type(exc).__name__
        try:
            # Model timeout includes process creation and monitor setup.
            remaining = max(0.0, float(timeout) - (time.perf_counter() - started))
            stdout, stderr = process.communicate(timeout=remaining)
            result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            audit["status"] = "completed" if process.returncode == 0 else "nonzero_exit"
        except BaseException as exc:
            failure = exc
            audit["status"] = "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "execution_interrupted"
            audit["tree_cleanup"] = _terminate_owned_process_tree(process, cleanup_timeout=cleanup_timeout)
            try:
                stdout, stderr = process.communicate(timeout=cleanup_timeout)
                audit["pipe_drain_status"] = "completed"
                if isinstance(exc, subprocess.TimeoutExpired):
                    exc.output, exc.stderr = stdout, stderr
            except Exception as drain_error:
                audit["pipe_drain_status"] = "incomplete"
                audit["pipe_drain_error_type"] = type(drain_error).__name__
        audit["returncode"] = process.poll()
    except BaseException as exc:
        failure = exc
        audit["error_type"] = type(exc).__name__
        if process is not None and audit["tree_cleanup"]["status"] == "not_requested":
            # Popen succeeded but setup (including monitor.start) was
            # interrupted before entering the communicate exception handler.
            audit["status"] = "execution_interrupted"
            audit["tree_cleanup"] = _terminate_owned_process_tree(process, cleanup_timeout=cleanup_timeout)
            try:
                process.communicate(timeout=cleanup_timeout)
                audit["pipe_drain_status"] = "completed"
            except BaseException as drain_error:
                audit["pipe_drain_status"] = "incomplete"
                audit["pipe_drain_error_type"] = type(drain_error).__name__
            audit["returncode"] = process.poll()
    finally:
        if monitor is not None:
            try:
                audit["resources"] = monitor.stop(timeout_seconds=cleanup_timeout)
            except Exception as exc:
                audit["resource_monitor_error_type"] = type(exc).__name__
        audit["elapsed_seconds"] = max(0.0, time.perf_counter() - started)
        if audit_callback is not None:
            try:
                audit_callback(audit)
            except Exception as exc:
                # No successful result can hide a missing required audit file.
                failure = exc
                audit["audit_persistence_error_type"] = type(exc).__name__
    if failure is not None:
        try:
            failure.resource_audit = audit
        except Exception:
            pass
        raise failure
    result.resource_audit = audit
    return result


def check_task_continuation(task_dir, worker_status=None, task_status=None):
    """Inspect only named resource JSON in this new task; never clean/restart.

    Caller persists the task's original outcome first, then persists this
    receipt and stops on rejection. Missing audits are reported as absent,
    never a claim that an unobserved child was cleaned up successfully.
    """
    import hashlib
    import json
    from collections.abc import Mapping

    reasons, records, worker_errors = [], [], []
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate resource audit JSON key")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite resource audit JSON")
    if task_dir is not None:
        try:
            root = Path(task_dir).absolute()
            if root.resolve(strict=True) != root or not root.is_dir():
                raise ValueError("unsafe task directory")
            # This is a bounded, named metadata inventory of this task only.
            # Frame-level subprocess logs can live below frame_results/*/logs.
            for path in sorted(root.rglob("*_subprocess.resources.json")):
                relative = path.relative_to(root)
                if "logs" not in relative.parts:
                    continue
                record = {"path": relative.as_posix()}
                records.append(record)
                try:
                    if path.resolve(strict=True) != path.absolute() or path.is_symlink():
                        raise ValueError("linked resource audit")
                    before = path.stat()
                    if before.st_size > 16 * 1024 * 1024:
                        raise ValueError("resource audit exceeds metadata budget")
                    raw = path.read_bytes()
                    after = path.stat()
                    if ((before.st_size, before.st_mtime_ns, before.st_ino) !=
                            (after.st_size, after.st_mtime_ns, after.st_ino)
                            or path.resolve(strict=True) != path.absolute()):
                        raise ValueError("resource audit changed during read")
                    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=nonfinite)
                    if (not isinstance(value, Mapping) or value.get("schema_version") != "subprocess_resources.v1"
                            or value.get("status") not in {"completed", "nonzero_exit", "timeout", "execution_interrupted", "launch_failed"}
                            or not isinstance(value.get("tree_cleanup"), Mapping)
                            or value["tree_cleanup"].get("status") not in {"not_requested", "tree_termination_requested", "unknown"}
                            or value.get("pipe_drain_status") not in {None, "completed", "incomplete"}):
                        raise ValueError("invalid resource audit schema/status")
                    record.update(sha256=hashlib.sha256(raw).hexdigest(), status=value["status"],
                                  tree_cleanup=value["tree_cleanup"], pipe_drain_status=value.get("pipe_drain_status"))
                    if value["tree_cleanup"]["status"] == "unknown":
                        reasons.append("subprocess_tree_cleanup_unknown:" + relative.as_posix())
                    if value.get("pipe_drain_status") == "incomplete":
                        reasons.append("subprocess_pipe_drain_incomplete:" + relative.as_posix())
                    if value["status"] in {"timeout", "execution_interrupted"} and (
                            value["tree_cleanup"]["status"] == "not_requested" or value.get("pipe_drain_status") != "completed"):
                        reasons.append("subprocess_interruption_cleanup_unverified:" + relative.as_posix())
                except (OSError, ValueError, TypeError, UnicodeError) as exc:
                    record.update(status="invalid", error_type=type(exc).__name__, error_message=str(exc))
                    reasons.append("invalid_subprocess_resource_audit:" + relative.as_posix())
        except (OSError, ValueError, TypeError) as exc:
            reasons.append("task_resource_audit_inventory_failed:" + type(exc).__name__)
    if worker_status is not None:
        if not isinstance(worker_status, Mapping):
            reasons.append("worker_status_invalid")
        else:
            unhealthy = worker_status.get("health") != "ok" or worker_status.get("ready") is not True
            for name, value in worker_status.items():
                if isinstance(value, Mapping) and (value.get("state") in {"error", "timeout", "failed", "unavailable"}
                                                   or value.get("health") in {"error", "timeout"}):
                    worker_errors.append({"worker": name, "state": value.get("state"), "health": value.get("health"),
                                          "error": value.get("error"), "healthy": value.get("healthy")})
            if unhealthy and (worker_errors or worker_status.get("health") in {"error", "timeout"}):
                reasons.append("persistent_worker_error_and_manager_unhealthy")
    return {"schema_version": "task.continuation_check.v1", "status": "rejected" if reasons else "passed",
            "reasons": reasons, "task_status": task_status, "resource_audits": records,
            "resource_audit_count": len(records), "worker_errors": worker_errors,
            "audit_scope": "named *_subprocess.resources.json under logs within this task only; no media read",
            "missing_audits_semantics": "not_observed_not_proof_of_cleanup",
            "automatic_cleanup": False, "automatic_worker_restart": False}
