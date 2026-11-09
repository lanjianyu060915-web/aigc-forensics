"""Add provenance and status semantics without reclassifying machine evidence.

The digest binds the exact decision input, including absence of evidence. It is
not an image hash or a claim that all supplied evidence supports the decision.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "4.0"
RULE_VERSION = "legacy-overall-and-judge-constraints-v1"
VALID_BANDS = frozenset({"none", "review", "obvious", "undetermined"})
UNAVAILABLE_STATUSES = frozenset({"error", "failed", "timeout", "skipped", "unavailable", "unrecorded"})


class ContractError(ValueError):
    """Evidence cannot be bound safely to a final decision."""


def canonical_digest(value: Any) -> str:
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ContractError("Decision evidence must be finite JSON data") from error
    return hashlib.sha256(raw).hexdigest()


def rule_source_hashes(system_root: str | Path | None = None) -> dict[str, str]:
    root = Path(system_root) if system_root is not None else Path(__file__).resolve().parents[1]
    names = ("decision/aigc_overall.py", "reporting/evidence_package.py", "reporting/llm_validation.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def status_snapshot(evidence: Mapping[str, Any]) -> dict[str, Any]:
    missing, not_applicable, statuses = [], [], {}
    for item in evidence.get("items", []):
        if not isinstance(item, Mapping) or not isinstance(item.get("item_id"), str):
            continue
        identifier = item["item_id"]
        if identifier in statuses:
            raise ContractError("Evidence identifiers must be unique")
        data = item.get("data", item)
        data = data if isinstance(data, Mapping) else {}
        status = data.get("status", "unrecorded")
        if not isinstance(status, str):
            raise ContractError("Evidence status must be an explicit string")
        statuses[identifier] = status
        record = {"item_id": identifier, "status": status}
        if status in {"not_applicable", "no_face"}:
            not_applicable.append(record)
        elif status in UNAVAILABLE_STATUSES:
            missing.append(record)
    return {"provided_evidence_ids": list(statuses), "evidence_statuses": statuses,
            "missing_evidence": missing, "not_applicable_evidence": not_applicable}


def normalize_final_decision(decision: Mapping[str, Any], evidence: Mapping[str, Any], *,
                             requested_mode: str, source_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Describe the already executed mode; never apply a new floor or threshold."""
    result = deepcopy(dict(decision))
    if result.get("final_band") not in VALID_BANDS:
        raise ContractError("Invalid final band")
    score = result.get("score")
    if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
        raise ContractError("Final score must be finite or null")
    constraints = evidence.get("constraints", {})
    constraints = constraints if isinstance(constraints, Mapping) else {}
    applied, unapplied = [], []
    for name in ("floor_band", "allow_obvious", "max_band_distance"):
        if name not in constraints:
            continue
        record = {"name": name, "value": deepcopy(constraints[name]), "source": "evidence_package.constraints"}
        if result.get("source") == "llm":
            if name == "max_band_distance":
                # Existing validator enforces distance 1, not an arbitrary
                # caller-provided number. Describe the executed rule exactly.
                record["requested_value"] = record["value"]
                record["value"] = 1
            record["execution"] = "validated_by_existing_judge_validator"
            applied.append(record)
        else:
            record["reason"] = "legacy_rule_or_fallback_keeps_rule_band"
            unapplied.append(record)
    result.update(schema_version=SCHEMA_VERSION, rule_version=RULE_VERSION,
                  rule_source_hashes=dict(source_hashes or {}),
                  evidence_digest=canonical_digest(evidence),
                  evidence_digest_scope="exact_decision_input",
                  decision_mode=requested_mode, effective_mode=result.get("source", "unrecorded"),
                  base_band=result.get("rule_band", "undetermined"),
                  applied_constraints=applied, unapplied_constraints=unapplied,
                  reasons=[{"text": result.get("reason", ""), "source": result.get("source", "unrecorded")}],
                  **status_snapshot(evidence))
    # Keep the old rule/score/band fields intact; state is independent of risk.
    result["decision_band_available"] = result["final_band"] != "undetermined"
    # A fallback band can exist when a required detector failed. Do not label
    # that as valid inference. Missing historical branch coverage is unknown.
    data = {row.get("item_id"): row.get("data", {}) for row in evidence.get("items", [])
            if isinstance(row, Mapping) and isinstance(row.get("data"), Mapping)}
    general, face = data.get("E03"), data.get("E04")
    valid = None
    reason = "required_branch_records_unavailable"
    if not result["decision_band_available"]:
        valid, reason = False, "no_available_final_band"
    elif general is not None and general.get("status") != "success":
        valid, reason = False, "general_branch_unavailable"
    elif general is not None and face is not None:
        audit = face.get("face_audit", {})
        coverage = audit.get("coverage", {}) if isinstance(audit, Mapping) else {}
        coverage = coverage if isinstance(coverage, Mapping) else {}
        failures = [coverage.get(key) for key in ("failed_count", "frame_failed_count", "model_failed_count")]
        if any(isinstance(value, (int, float)) and value > 0 for value in failures):
            valid, reason = False, "face_check_failure_recorded"
        elif face.get("status") in {"error", "failed", "timeout"}:
            valid, reason = False, "face_branch_failed"
        elif (coverage.get("count_status") == "recorded"
              and all(type(value) is int and value == 0 for value in failures)
              and isinstance(audit.get("frame_records"), list) and audit["frame_records"]
              and all(isinstance(row, Mapping) and row.get("status") in {"success", "no_face"} for row in audit["frame_records"])):
            total = coverage.get("detected_total")
            scores = [row.get("score") for row in face.get("per_face", []) if isinstance(row, Mapping)]
            finite_scores = bool(scores) and all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in scores)
            general_score = general.get("score")
            general_finite = isinstance(general_score, (int, float)) and not isinstance(general_score, bool) and math.isfinite(general_score)
            # Legacy aggregation can say no_face while smaller observed faces
            # have finite informational scores and none participates in the band.
            applicable = (type(total) is int and total == 0 and face.get("status") == "no_face") or (type(total) is int and total > 0 and face.get("status") in {"success", "no_face", "no_eligible_face"} and finite_scores)
            valid = bool(general_finite and applicable)
            reason = "recorded_required_branches" if valid else "required_branch_output_invalid"
    result["inference_valid"] = valid
    result["inference_validity_reason"] = reason
    return result


def validate_saved_decision(saved: Mapping[str, Any], evidence: Mapping[str, Any], *,
                            requested_mode: str, source_hashes: Mapping[str, str]) -> None:
    """A recovery caller must verify this binding before reusing a decision."""
    expected = {"schema_version": SCHEMA_VERSION, "rule_version": RULE_VERSION,
                "decision_mode": requested_mode, "evidence_digest": canonical_digest(evidence),
                "rule_source_hashes": dict(source_hashes)}
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ContractError("Saved decision does not match evidence, mode or frozen rule sources")
    if saved.get("final_band") not in VALID_BANDS:
        raise ContractError("Saved decision has an invalid final band")
    score = saved.get("score")
    if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
        raise ContractError("Saved decision score is invalid")

