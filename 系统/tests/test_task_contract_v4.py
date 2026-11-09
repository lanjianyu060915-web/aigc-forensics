"""Independently constructed evidence; no media, model, library or external API."""
from copy import deepcopy
import json

import pytest

from contracts.task_contract_v4 import (
    ContractError, canonical_digest, normalize_final_decision, validate_saved_decision,
)
from reporting.llm_client import DEFAULT_LLM_CONFIG, LLMSettings
from reporting.llm_service import run_judge


def evidence():
    return {"rule_band": "none", "constraints": {"floor_band": "review", "allow_obvious": False},
            "items": [{"item_id": "X1", "data": {"status": "success", "score": 1.25}},
                      {"item_id": "X2", "data": {"status": "error"}},
                      {"item_id": "X3", "data": {"status": "no_face"}}]}


def decision(source="rule", band="none"):
    return {"final_band": band, "rule_band": "none", "score": None,
            "source": source, "reason": "independently constructed fixture"}


@pytest.mark.parametrize("source", ["rule", "rule_fallback"])
def test_compatibility_records_unapplied_floor_without_upgrading(source):
    view, old = evidence(), decision(source)
    original = deepcopy((view, old))
    actual = normalize_final_decision(old, view, requested_mode="rule")
    assert {k: actual[k] for k in old} == old
    assert actual["applied_constraints"] == []
    assert actual["unapplied_constraints"][0]["name"] == "floor_band"
    assert (view, old) == original
    assert actual["decision_band_available"] is True
    assert actual["inference_valid"] is None  # unrelated X IDs cannot certify required detectors
    assert actual["missing_evidence"] == [{"item_id": "X2", "status": "error"}]
    assert actual["not_applicable_evidence"] == [{"item_id": "X3", "status": "no_face"}]


def test_llm_constraints_describe_existing_validator_not_a_second_vote():
    actual = normalize_final_decision(decision("llm", "review"), evidence(), requested_mode="llm")
    assert actual["unapplied_constraints"] == []
    assert [x["name"] for x in actual["applied_constraints"]] == ["floor_band", "allow_obvious"]


def test_unavailable_detection_is_not_a_low_band():
    actual = normalize_final_decision(decision(band="undetermined"), evidence(), requested_mode="rule")
    assert actual["final_band"] == "undetermined" and actual["inference_valid"] is False


def test_fallback_band_cannot_certify_failed_general_inference():
    view = {"rule_band": "none", "items": [{"item_id": "E03", "data": {"status": "error", "score": None}}]}
    actual = normalize_final_decision(decision(), view, requested_mode="rule")
    assert actual["final_band"] == "none" and actual["decision_band_available"]
    assert actual["inference_valid"] is False
    assert actual["inference_validity_reason"] == "general_branch_unavailable"


def test_no_face_is_valid_only_with_completed_recorded_zero_detection():
    view = {"rule_band": "none", "items": [
        {"item_id": "E03", "data": {"status": "success", "score": 0.1}},
        {"item_id": "E04", "data": {"status": "no_face", "per_face": [], "face_audit": {
            "coverage": {"count_status": "recorded", "detected_total": 0, "failed_count": 0,
                         "frame_failed_count": 0, "model_failed_count": 0},
            "frame_records": [{"status": "no_face"}]}}}]}
    assert normalize_final_decision(decision(), view, requested_mode="rule")["inference_valid"] is True
    view["items"][1]["data"]["face_audit"]["coverage"]["detected_total"] = 2
    assert normalize_final_decision(decision(), view, requested_mode="rule")["inference_valid"] is False
    view["items"][1]["data"]["per_face"] = [{"score": .1}, {"score": .2}]
    assert normalize_final_decision(decision(), view, requested_mode="rule")["inference_valid"] is True
    del view["items"][1]["data"]["face_audit"]
    assert normalize_final_decision(decision(), view, requested_mode="rule")["inference_valid"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_evidence_is_rejected(value):
    with pytest.raises(ContractError): canonical_digest({"value": value})


def test_recovery_requires_exact_evidence_mode_and_rule_sources():
    view = evidence()
    saved = normalize_final_decision(decision(), view, requested_mode="rule", source_hashes={"rules": "x"})
    validate_saved_decision(saved, view, requested_mode="rule", source_hashes={"rules": "x"})
    changed = deepcopy(view); changed["items"][0]["data"]["score"] = 8.5
    for item, mode, hashes in [(changed, "rule", {"rules": "x"}), (view, "llm", {"rules": "x"}),
                                (view, "rule", {"rules": "y"})]:
        with pytest.raises(ContractError):
            validate_saved_decision(saved, item, requested_mode=mode, source_hashes=hashes)


def test_duplicate_evidence_identifiers_are_not_silently_merged():
    view = evidence(); view["items"].append(deepcopy(view["items"][0]))
    with pytest.raises(ContractError): normalize_final_decision(decision(), view, requested_mode="rule")


def test_rule_entry_persists_same_canonical_object_without_network(monkeypatch, tmp_path):
    monkeypatch.setattr("reporting.llm_service.complete_json", lambda *a, **kw: pytest.fail("network forbidden"))
    options=deepcopy(DEFAULT_LLM_CONFIG); options["decision_mode"]="rule"
    result=run_judge(evidence(), settings=LLMSettings(options=options), task_dir=tmp_path)
    assert json.loads((tmp_path/"final_decision.json").read_text("utf-8")) == result
    assert result["final_band"] == "none" and len(result["rule_source_hashes"]) == 3


def test_invalid_status_is_rejected_before_network_or_audit(monkeypatch, tmp_path):
    monkeypatch.setattr("reporting.llm_service.complete_json", lambda *a, **kw: pytest.fail("network forbidden"))
    view=evidence(); view["items"][0]["data"]["status"]={"invalid": True}
    options=deepcopy(DEFAULT_LLM_CONFIG); options["decision_mode"]="llm"
    with pytest.raises(ContractError):
        run_judge(view, settings=LLMSettings(options=options), task_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_existing_decision_cannot_be_rejudged_or_overwritten(monkeypatch, tmp_path):
    options = deepcopy(DEFAULT_LLM_CONFIG)
    options["decision_mode"] = "rule"
    settings = LLMSettings(options=options)
    run_judge(evidence(), settings=settings, task_dir=tmp_path)
    before = (tmp_path / "final_decision.json").read_bytes()
    monkeypatch.setattr("reporting.llm_service.complete_json", lambda *a, **kw: pytest.fail("network forbidden"))
    options["decision_mode"] = "llm"
    with pytest.raises(ValueError, match="已有研判"):
        run_judge(evidence(), settings=settings, task_dir=tmp_path, bypass_cache=True)
    assert (tmp_path / "final_decision.json").read_bytes() == before
