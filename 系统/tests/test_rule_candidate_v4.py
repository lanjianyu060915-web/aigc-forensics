"""Pure synthetic rule fixtures; no saved reports, model files or media."""
from copy import deepcopy

import pytest

from decision.rule_candidate_v4 import apply_rule_candidate, digest, freeze_candidate

RELEASE = "a" * 64


def case(policy="existing_review_floor", band="none", floor="review"):
    evidence = {"rule_band": band, "constraints": {"floor_band": floor},
                "items": [{"item_id": "E01", "data": {"declared_generated": True}},
                          {"item_id": "E07", "data": {"score": 0.9999}}]}
    baseline = {"rule_band": band, "final_band": band, "source": "rule", "score": 12,
                "reason": "synthetic baseline", "judgment": "synthetic old-band prose",
                "rule_version": "synthetic-legacy", "release_sha256": RELEASE,
                "evidence_digest": digest(evidence)}
    candidate = freeze_candidate(policy, baseline_release_sha256=RELEASE, baseline_rule_version="synthetic-legacy")
    return baseline, evidence, candidate


def apply(baseline, evidence, candidate):
    return apply_rule_candidate(baseline, evidence, candidate, enabled=True, experimental=True)


def rebind(baseline, evidence):
    baseline["evidence_digest"] = digest(evidence)


def test_disabled_returns_exact_independent_copy_without_candidate_execution():
    baseline, _, _ = case()
    result = apply_rule_candidate(baseline, None, None)
    assert result == baseline and result is not baseline


@pytest.mark.parametrize("flags", [(True, False), (False, True), (1, True), (True, 1), (None, True)])
def test_dual_switches_require_literal_true(flags):
    with pytest.raises(ValueError, match="Both"):
        apply_rule_candidate(*case(), enabled=flags[0], experimental=flags[1])


def test_existing_floor_only_raises_to_review_and_keeps_frozen_baseline():
    baseline, evidence, candidate = case()
    baseline["unapplied_constraints"] = [{"name": "floor_band", "value": "review", "reason": "legacy_unapplied"}]
    before = deepcopy((baseline, evidence, candidate))
    result = apply(baseline, evidence, candidate)
    assert result["final_band"] == "review"
    assert result["score"] == baseline["score"]
    assert result["baseline_decision"] == baseline
    assert result["score_semantics"] == "unchanged_baseline_score_not_recalibrated"
    assert result["applied_constraints"][-1]["name"] == "floor_band"
    assert not result["unapplied_constraints"]
    assert "judgment" not in result
    assert (baseline, evidence, candidate) == before


@pytest.mark.parametrize("band", ["review", "obvious", "undetermined"])
def test_existing_floor_never_lowers_or_resolves_undetermined(band):
    result = apply(*case(band=band))
    assert result["final_band"] == band
    assert result["experimental_trace"]["changed"] is False


def test_identity_similarity_alone_cannot_create_floor_support():
    baseline, evidence, candidate = case()
    evidence["items"] = evidence["items"][1:]
    evidence["clues"] = [{"rule_id": "R1+", "matched": True}]
    rebind(baseline, evidence)
    with pytest.raises(ValueError, match="non-identity"):
        apply(baseline, evidence, candidate)


def test_positive_logits_and_shared_model_branches_are_not_independent_votes():
    baseline, evidence, candidate = case()
    evidence["items"] = [{"item_id": key, "data": {"band": "low", "score": 1.5, "model": {"id": "shared-cf"}}} for key in ("E03", "E04")]
    rebind(baseline, evidence)
    with pytest.raises(ValueError, match="support"):
        apply(baseline, evidence, candidate)
    for row in evidence["items"]:
        row["data"]["band"] = "high"
    rebind(baseline, evidence)
    result = apply(baseline, evidence, candidate)
    assert result["final_band"] == "review"
    assert result["experimental_trace"]["independent_vote_count"] is None
    assert result["experimental_trace"]["floor_applied_once"] is True


def test_existing_source_pair_floor_has_no_identity_similarity_dependency():
    baseline, evidence, candidate = case()
    evidence["items"] = [{"item_id": "E06", "data": {"candidates": [{"rank": 2, "open_set_state": "high_confidence"}]}},
                         {"item_id": "E08", "data": {"candidates": [{"rank": 2, "code": "face_replaced"}]}}]
    rebind(baseline, evidence)
    assert apply(baseline, evidence, candidate)["final_band"] == "review"


@pytest.mark.parametrize("policy", ["baseline", "face_hints_only"])
def test_informational_candidates_preserve_band(policy):
    baseline, evidence, candidate = case(policy=policy)
    evidence["items"].append({"item_id": "E04", "data": {"per_face": [
        {"face_uid": "frame/face2", "band": "high", "used_for_band": False, "score_status": "success"}]}})
    rebind(baseline, evidence)
    result = apply(baseline, evidence, candidate)
    assert result["final_band"] == "none"
    if policy == "face_hints_only":
        assert result["experimental_trace"]["face_hints"][0]["band"] == "high"


@pytest.mark.parametrize("mutation", ["candidate", "evidence", "release", "rule_version", "llm", "reclassified", "floor"])
def test_mismatched_frozen_inputs_fail_closed(mutation):
    baseline, evidence, candidate = case()
    if mutation == "candidate":
        candidate["complexity_rank"] = 0
    elif mutation == "evidence":
        evidence["new"] = True
    elif mutation in {"release", "rule_version"}:
        baseline["release_sha256" if mutation == "release" else "rule_version"] = "wrong"
    elif mutation == "llm":
        baseline["source"] = "llm"
    elif mutation == "reclassified":
        baseline["final_band"] = "review"
    else:
        evidence["constraints"]["floor_band"] = "obvious"
        rebind(baseline, evidence)
    with pytest.raises(ValueError):
        apply(baseline, evidence, candidate)


def test_no_arbitrary_weight_threshold_policy_or_nonfinite_digest():
    with pytest.raises(ValueError):
        freeze_candidate("tune_threshold", baseline_release_sha256=RELEASE, baseline_rule_version="x")
    with pytest.raises(ValueError):
        digest({"score": float("nan")})
