"""Opt-in, pure strategy candidates; production rules do not import this module.

The existing review floor is applied once, not as votes from correlated model
branches. Identity similarity is never an input to this candidate. A digest is
an integrity binding, not proof of authorization or independent validation.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import re


VERSION = "v4-rule-candidate-1-experimental"
POLICIES = {"baseline": 0, "face_hints_only": 1, "existing_review_floor": 2}
BANDS = frozenset({"none", "review", "obvious", "undetermined"})


def digest(value):
    """Hash finite JSON, without loading files or accepting non-JSON objects."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def require_sha256(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def seal(value):
    output = deepcopy(dict(value))
    if "digest" in output:
        raise ValueError("Already sealed")
    output["digest"] = digest(output)
    return output


def verify_seal(value):
    if not isinstance(value, Mapping) or value.get("digest") != digest({k: v for k, v in value.items() if k != "digest"}):
        raise ValueError("Frozen descriptor was changed")


def freeze_candidate(policy, *, baseline_release_sha256, baseline_rule_version):
    if policy not in POLICIES:
        raise ValueError("Candidate must be predeclared; arbitrary thresholds are not supported")
    require_sha256(baseline_release_sha256, "baseline_release_sha256")
    if not isinstance(baseline_rule_version, str) or not baseline_rule_version:
        raise ValueError("A frozen baseline rule version is required")
    return seal({"version": VERSION, "policy": policy,
        "baseline_release_sha256": baseline_release_sha256,
        "baseline_rule_version": baseline_rule_version,
        "complexity_rank": POLICIES[policy], "independent_validation": "not_established",
        "changes_model_thresholds": False, "uses_identity_similarity": False,
        "shared_model_evidence": "not_independent_votes"})


def verify_candidate(candidate):
    verify_seal(candidate)
    expected = freeze_candidate(candidate.get("policy"),
        baseline_release_sha256=candidate.get("baseline_release_sha256"),
        baseline_rule_version=candidate.get("baseline_rule_version"))
    if dict(candidate) != expected:
        raise ValueError("Candidate differs from the versioned policy definition")


def _rows(value):
    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def _floor_basis(evidence):
    items = _rows(evidence.get("items"))
    if len({row.get("item_id") for row in items}) != len(items):
        raise ValueError("Duplicate evidence identifiers")
    data = {row.get("item_id"): row.get("data", {}) for row in items
            if isinstance(row.get("data", {}), Mapping)}
    # Preserve exactly the existing kinds of floor support. No raw logits,
    # similarity or independent-branch count can create a new positive vote.
    support = [key for key in ("E03", "E04") if data.get(key, {}).get("band") == "high"]
    if data.get("E01", {}).get("declared_generated") is True:
        support.append("E01")
    if data.get("E02", {}).get("visible_label_found") is True:
        support.append("E02")
    source_ranks = {row.get("rank") for row in _rows(data.get("E06", {}).get("candidates"))
                    if row.get("open_set_state") == "high_confidence"}
    if any(row.get("rank") in source_ranks and row.get("code") == "face_replaced"
           for row in _rows(data.get("E08", {}).get("candidates"))):
        support.extend(["E06", "E08"])
    return support


def apply_rule_candidate(baseline, evidence, candidate, *, enabled=False, experimental=False):
    """Return an experimental decision envelope; disabled means exact copy.

    Only the existing rule/rule-fallback result is eligible. This function is
    deliberately not called by run_judge, decide_overall or the report writer.
    The input score remains a baseline score and is never relabelled as a new
    calibrated score when the candidate raises none to review.
    """
    if enabled is False and experimental is False:
        return deepcopy(baseline)
    if enabled is not True or experimental is not True:
        raise ValueError("Both experimental rule switches must explicitly be true")
    verify_candidate(candidate)
    if not isinstance(baseline, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError("Candidate requires explicit baseline and evidence objects")
    band = baseline.get("final_band")
    if band not in BANDS or baseline.get("source") not in {"rule", "rule_fallback"}:
        raise ValueError("Candidate only accepts a frozen legacy rule result")
    if baseline.get("rule_version") != candidate["baseline_rule_version"]:
        raise ValueError("Baseline rule version differs from frozen candidate")
    if baseline.get("release_sha256") != candidate["baseline_release_sha256"]:
        raise ValueError("Baseline release differs from frozen candidate")
    if baseline.get("rule_band") != evidence.get("rule_band") or band != baseline.get("rule_band"):
        raise ValueError("Baseline was already reclassified or is not bound to this rule band")
    if baseline.get("evidence_digest") != digest(evidence):
        raise ValueError("Baseline evidence binding does not match")
    policy = candidate["policy"]
    constraints = evidence.get("constraints", {})
    if not isinstance(constraints, Mapping):
        raise ValueError("Constraints must be an object")
    floor = constraints.get("floor_band", "none")
    if floor not in {"none", "review"}:
        raise ValueError("This candidate only implements the existing none/review floor")
    basis = _floor_basis(evidence) if policy == "existing_review_floor" else []
    if policy == "existing_review_floor" and floor == "review" and not basis:
        raise ValueError("Review floor lacks supported non-identity evidence")
    changed = policy == "existing_review_floor" and floor == "review" and band == "none"
    result = deepcopy(dict(baseline))
    result.update(final_band="review" if changed else band, source="rule_candidate_experimental",
                  rule_version=VERSION, candidate_digest=candidate["digest"],
                  decision_mode="rule_candidate_experimental", effective_mode="rule_candidate_experimental",
                  baseline_decision=deepcopy(dict(baseline)), baseline_decision_digest=digest(baseline),
                  score_semantics="unchanged_baseline_score_not_recalibrated",
                  experimental_trace={"policy": policy, "changed": changed,
                      "floor_band": floor, "floor_applied_once": changed,
                      "supporting_evidence_ids": basis, "independent_vote_count": None,
                      "identity_similarity_used": False,
                      "independent_validation": "not_established"})
    if changed:
        result["reason"] = "Experimental candidate applies the existing review floor; baseline retained in baseline_decision."
        result["reasons"] = [{"text": result["reason"], "source": "rule_candidate_experimental"}]
        result["unapplied_constraints"] = [deepcopy(row) for row in _rows(result.get("unapplied_constraints"))
                                             if row.get("name") != "floor_band"]
        result["applied_constraints"] = [*_rows(result.get("applied_constraints")),
            {"name": "floor_band", "value": "review", "source": "evidence_package.constraints",
             "execution": "experimental_candidate_applied_once"}]
        result.pop("judgment", None)  # Do not reuse prose naming the old final band.
    if policy == "face_hints_only":
        face_data = next((row.get("data", {}) for row in _rows(evidence.get("items"))
                          if row.get("item_id") == "E04"), {})
        result["experimental_trace"]["face_hints"] = [
            {key: deepcopy(row.get(key)) for key in ("face_uid", "band", "score_status", "used_for_band")}
            for row in _rows(face_data.get("per_face"))] if isinstance(face_data, Mapping) else []
        result["experimental_trace"]["face_hints_semantics"] = "informational_only_no_extra_votes_or_band_change"
    return result
