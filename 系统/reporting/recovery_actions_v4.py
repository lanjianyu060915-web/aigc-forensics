"""Explicit recovery actions. No batch resumption or implicit judgment retries.

Each action checks its own evidence and authorization. Cloud claims are never
removed/refunded: a crash with uncertain delivery consumes the claim and blocks
further cloud retries until independently resolved. Local report repair remains
available through analysis_recovery.recover_analysis_report.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import stat
import uuid

from artifact_io import atomic_write_bytes, atomic_write_json, write_stage_receipt
from contracts.task_contract_v4 import canonical_digest, validate_saved_decision
from contracts.execution_profile_v4 import is_independent_profile
from .analysis_recovery import _load_checkpoint, report_config_digest, save_report_checkpoint
from .evidence_figures import local_file
from .llm_client import assert_upload_safe, now_iso
from .llm_service import run_writer

MAX_CLOUD_RETRIES = 2
WRITER_ATTEMPTS_DIR = "wr_v4"


class RecoveryActionError(ValueError):
    pass


def _authorized(value):
    if value is not True:
        raise RecoveryActionError("current_explicit_action_authorization_required")


def _root(task_dir):
    path = Path(task_dir).absolute()
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        raise RecoveryActionError("task_directory_links_forbidden")
    return path.resolve(strict=True)


def _stop(root, stop_requested=False, stop_paths=()):
    stopped = stop_requested() if callable(stop_requested) else stop_requested
    if stopped or any((root / name).exists() for name in ("STOP", "CANCEL_SHUTDOWN")) or any(Path(p).exists() for p in stop_paths):
        raise RecoveryActionError("recovery_stopped")


def _directory(root, relative):
    path = root
    for part in Path(relative).parts:
        if part in {"", ".", ".."}:
            raise RecoveryActionError("unsafe_recovery_directory")
        path = path / part
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise RecoveryActionError("recovery_directory_links_forbidden")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read(root, name):
    path = local_file(root, name)
    if path is None:
        raise RecoveryActionError("required_recovery_evidence_missing:" + str(name))
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise RecoveryActionError("invalid_recovery_json:" + str(name)) from exc
    return value, {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}


def _frozen(root, report, config, settings):
    source = {k: deepcopy(v) for k, v in report.items() if k not in {"analysis", "evidence_package"}}
    source_digest = canonical_digest(source)
    config_digest = report_config_digest(config, settings)
    stage = "report_ready" if (root / "recovery_v4/report_ready.json").exists() else "decision"
    checkpoint = _load_checkpoint(root, stage, source_digest=source_digest, config_digest=config_digest)
    data = checkpoint["payload"]
    final = data["final_decision"]
    validate_saved_decision(final, data["decision_input"], requested_mode=settings.options["decision_mode"], source_hashes=final["rule_source_hashes"])
    actual, _ = _read(root, "final_decision.json")
    if actual != final:
        raise RecoveryActionError("frozen_decision_changed")
    return source, source_digest, config_digest, stage, checkpoint


def _record(root, name):
    path = local_file(root, name)
    if path is None:
        raise RecoveryActionError("recovery_artifact_missing:" + str(name))
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}


def _grouping(source, checkpoint):
    material = next((item.get("data", {}) for item in checkpoint["payload"]["package"].get("items", [])
                     if item.get("item_id") == "E09"), {})
    digest = material.get("computed_input_sha256") or material.get("input_sha256")
    valid = isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest)
    return {"original_task_id": source.get("task", {}).get("task_id"),
            "input_content_sha256": digest.lower() if valid else None,
            "content_group_status": "bound_input_digest" if valid else "unavailable_do_not_count_as_independent_sample",
            "fallback_source_report_sha256": canonical_digest(source), "independent_sample": False}


def write_recovery_receipt(task_dir, report, *, config, settings, authorized=False,
                           stop_requested=False, stop_paths=()):
    """Append a receipt only for a fully verified already-generated report."""
    _authorized(authorized)
    root = _root(task_dir)
    _stop(root, stop_requested, stop_paths)
    source, source_digest, config_digest, stage, checkpoint = _frozen(root, report, config, settings)
    if stage != "report_ready":
        raise RecoveryActionError("report_ready_checkpoint_required")
    if (root / "stage_receipts/analysis.initial.report.json").exists() or (root / "recovery_receipts_v4/completed.json").exists():
        raise RecoveryActionError("report_receipt_already_present_or_requires_separate_integrity_review")
    from .analysis_pipeline import verify_analysis_artifacts
    if verify_analysis_artifacts(root):
        raise RecoveryActionError("complete_report_artifact_manifest_required")
    saved, _ = _read(root, "analysis_report.json")
    final, _ = _read(root, "final_decision.json")
    package, _ = _read(root, "evidence_package.json")
    if (saved.get("source_report_sha256") != source_digest or saved.get("status") != "success"
            or saved.get("final_decision") != final or saved.get("evidence_package") != package
            or final != checkpoint["payload"]["final_decision"] or package != checkpoint["payload"]["package"]):
        raise RecoveryActionError("report_final_package_checkpoint_disagreement")
    records = [_record(root, name) for name in ("analysis_artifact_manifest.json", "analysis_report.json", "final_decision.json", "evidence_package.json", "recovery_v4/report_ready.json")]
    _stop(root, stop_requested, stop_paths)
    attempt = uuid.uuid4().hex
    directory = _directory(root, "recovery_receipts_v4")
    receipt = write_stage_receipt(directory, task_id="analysis", attempt_id=attempt, stage="receipt", status="completed",
        input_digest=source_digest, config_digest=config_digest, artifacts=records)
    result = {"schema_version": "recovery_action.v1", "action": "write_receipt", "status": "completed", "attempt_id": attempt,
              "source_report_sha256": source_digest, "config_digest": config_digest, "receipt_file": receipt.relative_to(root).as_posix(),
              "model_calls": 0, "cloud_calls": 0, "pdf_renders": 0, "old_records_overwritten": False,
              "independent_sample": False, "created_at": now_iso()}
    result["grouping"] = _grouping(source, checkpoint)
    atomic_write_json(directory / "completed.json", result, role="recovery_receipt_completion", overwrite=False)
    return result


def _initial_retry_count(root, checkpoint):
    names = {item["path"] for item in checkpoint["artifacts"]}
    if "llm_writer_request.json" not in names:
        raise RecoveryActionError("original_writer_attempt_audit_unknown_local_report_repair_remains_available")
    request, _ = _read(root, "llm_writer_request.json")
    metadata, attempts = request.get("metadata"), request.get("attempts")
    if not isinstance(metadata, list) or not isinstance(attempts, list) or len(metadata) != len(attempts):
        raise RecoveryActionError("original_writer_attempt_audit_invalid")
    if any(not isinstance(entry, dict) or type(entry.get("request_attempted")) is not bool for entry in metadata):
        raise RecoveryActionError("original_writer_delivery_unknown_local_report_repair_remains_available")
    if any(entry.get("transmission_status") == "delivery_unknown" or
           (entry["request_attempted"] and entry.get("transmission_status") != "response_received") for entry in metadata):
        raise RecoveryActionError("original_writer_delivery_unknown_local_report_repair_remains_available")
    return max(0, sum(entry["request_attempted"] for entry in metadata) - 1)


def _claim(root, *, source_digest, config_digest, prior_retries, attempt):
    directory = _directory(root, "writer_retry_claims_v4")
    for slot in range(1, MAX_CLOUD_RETRIES - prior_retries + 1):
        path = directory / f"slot-{slot}.json"
        if not path.exists():
            value = {"schema_version": "writer_retry_claim.v1", "slot": slot, "attempt_id": attempt,
                     "source_report_sha256": source_digest, "config_digest": config_digest,
                     "initial_internal_retries": prior_retries, "total_retry_limit": MAX_CLOUD_RETRIES,
                     "attempt_directory": f"{WRITER_ATTEMPTS_DIR}/{attempt[:16]}", "created_at": now_iso()}
            try:
                atomic_write_json(path, value, role="immutable_cloud_retry_claim", overwrite=False)
                return value
            except FileExistsError:
                pass  # Another caller claimed it first; inspect, never replace.
        existing, _ = _read(root, path.relative_to(root))
        if (existing.get("schema_version") != "writer_retry_claim.v1" or existing.get("slot") != slot
                or existing.get("source_report_sha256") != source_digest or existing.get("config_digest") != config_digest
                or existing.get("initial_internal_retries") != prior_retries
                or not re.fullmatch(r"[0-9a-f]{32}", str(existing.get("attempt_id", "")))
                or existing.get("attempt_directory") != f"{WRITER_ATTEMPTS_DIR}/{existing['attempt_id'][:16]}"):
            raise RecoveryActionError("existing_retry_claim_invalid")
        outcome_path = f"{existing['attempt_directory']}/recovery_result.json"
        if local_file(root, outcome_path) is None:
            raise RecoveryActionError("previous_writer_retry_pending_or_delivery_unknown")
        outcome, _ = _read(root, outcome_path)
        if outcome.get("attempt_id") != existing["attempt_id"] or outcome.get("source_report_sha256") != source_digest:
            raise RecoveryActionError("retry_outcome_binding_invalid")
        if outcome.get("writer_status") == "success":
            raise RecoveryActionError("writer_already_succeeded_no_further_retry")
        if outcome.get("cloud_calls") is None or outcome.get("delivery_unknown") is True or outcome.get("status") == "stopped":
            raise RecoveryActionError("previous_writer_delivery_unknown_or_stopped")
    raise RecoveryActionError("cloud_retry_budget_exhausted")


def retry_report_writer(task_dir, report, *, config, settings, expected_writer_prompt_sha256,
                        authorized=False, cloud_authorized=False, stop_requested=False, stop_paths=()):
    """One explicit writer request, at most two total retries per original task.

    The expected prompt digest must come from the frozen release/authorization,
    not be guessed from current code. Existing internal retries count against
    the limit. A new render-ready checkpoint may subsequently be rendered by the
    existing local-only recovery function; this action never renders a PDF.
    """
    _authorized(authorized)
    _authorized(cloud_authorized)
    if is_independent_profile(config):
        raise RecoveryActionError("independent_profiles_forbid_cloud")
    if (not settings.configured or settings.options.get("upload_scope") != "text_only"
            or settings.options.get("judge_send_images") is not False
            or settings.base_url.rstrip("/") != "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        raise RecoveryActionError("existing_authorized_text_only_cloud_protocol_required")
    root = _root(task_dir)
    _stop(root, stop_requested, stop_paths)
    if (root / "recovery_origin_v4.json").exists():
        raise RecoveryActionError("retry_must_target_original_task_budget")
    source, source_digest, config_digest, stage, checkpoint = _frozen(root, report, config, settings)
    data = checkpoint["payload"]
    if stage != "report_ready" or data.get("writer", {}).get("status") == "success":
        raise RecoveryActionError("verified_incomplete_writer_stage_required")
    if data["final_decision"].get("source") == "rule_fallback":
        raise RecoveryActionError("frozen_decision_not_eligible_for_writer_retry")
    prompt_path = Path(__file__).with_name("prompts") / "writer_v1.md"
    prompt_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    if not isinstance(expected_writer_prompt_sha256, str) or expected_writer_prompt_sha256 != prompt_sha:
        raise RecoveryActionError("frozen_writer_prompt_digest_mismatch")
    previous_prompt = data.get("writer", {}).get("prompt_sha256")
    if previous_prompt and previous_prompt != prompt_sha:
        raise RecoveryActionError("original_writer_prompt_changed")
    validator = Path(__file__).with_name("llm_validation.py")
    if data["final_decision"]["rule_source_hashes"].get("reporting/llm_validation.py") != hashlib.sha256(validator.read_bytes()).hexdigest():
        raise RecoveryActionError("frozen_writer_validator_changed")
    prior_retries = _initial_retry_count(root, checkpoint)
    from .evidence_package import to_llm_payload
    safe_view = to_llm_payload(data["decision_input"], "text_only")
    safe_final = to_llm_payload(data["final_decision"], "text_only")
    assert_upload_safe(safe_view)
    assert_upload_safe(safe_final)
    attempt = uuid.uuid4().hex
    claim = _claim(root, source_digest=source_digest, config_digest=config_digest, prior_retries=prior_retries, attempt=attempt)
    destination = _directory(root, claim["attempt_directory"])
    result = {"schema_version": "recovery_action.v1", "action": "retry_cloud_writer_only", "attempt_id": attempt,
              "source_report_sha256": source_digest, "config_digest": config_digest, "writer_status": None,
              "status": "failed", "cloud_calls": 0, "judge_calls": 0, "model_calls": 0, "pdf_renders": 0,
              "independent_sample": False, "claim": claim, "old_records_overwritten": False}
    result["grouping"] = _grouping(source, checkpoint)
    writer_invoked = False
    try:
        atomic_write_json(destination / "recovery_origin_v4.json", {"original_task_dir": str(root), "source_report_sha256": source_digest,
            "original_task_id": source.get("task", {}).get("task_id"), "claim": claim}, overwrite=False)
        copied = []
        for item in checkpoint["artifacts"]:
            # Old writer audits stay in the original attempt. New writer files
            # have the same conventional names only inside the fresh directory.
            if Path(item["path"]).name.startswith("llm_writer_"):
                continue
            original = local_file(root, item["path"])
            raw = original.read_bytes() if original else b""
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise RecoveryActionError("evidence_changed_during_writer_snapshot")
            atomic_write_bytes(destination / item["path"], raw, overwrite=False)
            copied.append(item["path"])
        package = deepcopy(data["package"])
        for figure in package.get("figures", []):
            for key in ("original_path", "path", "preview_path"):
                original = local_file(root, figure.get(key))
                if original is not None and (destination / original.relative_to(root)).is_file():
                    figure[key] = str(destination / original.relative_to(root))
        save_report_checkpoint(destination, stage="decision", source_digest=source_digest, config_digest=config_digest,
            payload={"package": package, "decision_input": data["decision_input"], "final_decision": data["final_decision"]}, artifact_names=copied)
        _stop(root, stop_requested, stop_paths)
        limited_settings = replace(settings, options={**deepcopy(settings.options), "max_retries": 0})
        writer_invoked = True
        writer = run_writer(safe_view, safe_final, settings=limited_settings, task_dir=destination, figures=(), bypass_cache=True)
        request, _ = _read(destination, "llm_writer_request.json")
        metadata = request.get("metadata", [])
        if not isinstance(metadata, list) or any(type(item.get("request_attempted")) is not bool for item in metadata):
            raise RecoveryActionError("new_writer_delivery_audit_unknown")
        result["cloud_calls"] = sum(item["request_attempted"] for item in metadata)
        result["delivery_unknown"] = any(item.get("transmission_status") == "delivery_unknown" or
            (item["request_attempted"] and item.get("transmission_status") != "response_received") for item in metadata)
        if result["cloud_calls"] > 1:
            raise RecoveryActionError("writer_exceeded_single_request_claim")
        result["writer_status"] = writer.get("status")
        _stop(root, stop_requested, stop_paths)
        from .analysis_templates import build_template_content, merge_writer_content
        from .coverage_summary_v4 import append_coverage_notes
        content = append_coverage_notes(merge_writer_content(build_template_content(package, data["final_decision"],
            band_labels=settings.options["band_labels"]), writer), package)
        save_report_checkpoint(destination, stage="report_ready", source_digest=source_digest, config_digest=config_digest,
            payload={"package": package, "decision_input": data["decision_input"], "final_decision": data["final_decision"], "writer": writer, "content": content},
            artifact_names=copied + writer.get("audit_files", []))
        result.update(status="completed" if writer.get("status") == "success" else "partial" if writer.get("status") == "partial" else "failed",
                      render_ready=True, frozen_final_decision_sha256=canonical_digest(data["final_decision"]))
    except Exception as exc:
        # Never serialize raw exception messages: they may contain credentials,
        # request bodies, private local paths or vendor response text.
        result.update(status="stopped" if isinstance(exc, RecoveryActionError) and str(exc) == "recovery_stopped" else "failed",
                      error_type=type(exc).__name__, error_code=str(exc) if isinstance(exc, RecoveryActionError) else "writer_attempt_failed")
        if writer_invoked and result["writer_status"] is None:
            result["cloud_calls"] = None
    atomic_write_json(destination / "recovery_result.json", result, role="writer_recovery_outcome", overwrite=False)
    return {"task_dir": str(destination), "result": result}


def rerun_as_new_task(original_task_dir, query, *, config_path, authorized=False,
                     selected_modules=None, stop_requested=False, stop_paths=(), progress_callback=None):
    """Dispatch one new independently admitted task; never resume a batch.

    Only originals with an independent execution binding qualify. Team test-only
    and unrecorded historical jobs cannot be resumed through this API. The normal
    run_task entry revalidates current policy and consumes admitted snapshot bytes.
    """
    _authorized(authorized)
    root = _root(original_task_dir)
    _stop(root, stop_requested, stop_paths)
    old, _ = _read(root, "execution_binding.json")
    if not str(old.get("usage", "")).startswith("independent_"):
        raise RecoveryActionError("historical_or_team_batch_rerun_forbidden")
    from orchestrator import load_config, run_task
    from contracts.execution_profile_v4 import (audit_execution_profile, admit_query_snapshot, trusted_policy_authority)
    config, resolved = load_config(config_path)
    if not is_independent_profile(config):
        raise RecoveryActionError("current_independent_admission_profile_required")
    authority = trusted_policy_authority()
    audit = audit_execution_profile(config, authority=authority, config_path=resolved, for_inference=True)
    snapshot = admit_query_snapshot(config, query, authority=authority, expected_frozen_release=config["execution_profile"].get("frozen_release"))
    if old.get("input_digest") != snapshot.sha256:
        raise RecoveryActionError("new_query_does_not_match_original_content_group")
    tasks_root = Path(config["runtime"]["tasks_root"]).resolve()
    if tasks_root == root or tasks_root.is_relative_to(root):
        raise RecoveryActionError("new_task_root_cannot_be_inside_original_task")
    _stop(root, stop_requested, stop_paths)
    attempt = uuid.uuid4().hex
    directory = _directory(root, "inference_rerun_requests_v4")
    independent_root = _root(config["execution_profile"]["independent_root"])
    config_snapshot = _directory(independent_root, "profiles/recovery_requests") / (attempt + ".json")
    # Do not reopen a mutable original configuration at dispatch time. All
    # independent configuration paths are explicit absolute paths already.
    atomic_write_json(config_snapshot, config, role="new_task_configuration_snapshot", overwrite=False)
    config_snapshot.chmod(stat.S_IREAD)
    request = {"schema_version": "inference_rerun_request.v1", "attempt_id": attempt, "original_task_id": old.get("task_id"),
               "input_digest": snapshot.sha256, "config_digest": audit["config_digest"], "policy_binding": audit["policy_binding"],
               "action": "new_task_only", "independent_sample": False, "batch_resume": False, "created_at": now_iso(),
               "configuration_snapshot_sha256": hashlib.sha256(config_snapshot.read_bytes()).hexdigest()}
    atomic_write_json(directory / (attempt + ".request.json"), request, overwrite=False)
    try:
        outcome = run_task(query, config_path=config_snapshot, selected_modules=selected_modules, services=None, progress_callback=progress_callback)
        new_root = Path(outcome["task_dir"]).resolve(strict=True)
        if new_root == root or not new_root.is_relative_to(tasks_root):
            raise RecoveryActionError("dispatcher_did_not_create_a_new_task")
        atomic_write_json(new_root / "rerun_lineage_v4.json", request, overwrite=False)
        _stop(root, stop_requested, stop_paths)
        record = {**request, "status": "dispatched_new_task", "new_task_dir": str(new_root), "result_status": outcome.get("status")}
    except Exception as exc:
        record = {**request, "status": "stopped" if isinstance(exc, RecoveryActionError) and str(exc) == "recovery_stopped" else "failed", "error_type": type(exc).__name__}
        atomic_write_json(directory / (attempt + ".outcome.json"), record, overwrite=False)
        raise
    atomic_write_json(directory / (attempt + ".outcome.json"), record, overwrite=False)
    return {"task_dir": str(new_root), "result": outcome, "recovery": record}
