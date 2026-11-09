"""Newly constructed metadata/results only; no media/model/historical reports."""
from copy import deepcopy
import math

import pytest

from decision.candidate_evaluation_v4 import (evaluate_candidate, freeze_experiment, one_sided_cp_upper,
                                            paired_differences, select_development_candidate)
from decision.rule_candidate_v4 import digest, freeze_candidate

RELEASE = "a" * 64
PROTOCOL_FILE = "b" * 64
MODULES = ["general_aigc", "face_branch_check", "canonical_decision", "evidence_report"]


def protocol():
    # Intentionally permissive SYNTHETIC test protocol, never a business gate.
    return {"protocol_id": "synthetic-unit-test-only", "metrics": {
        "required_modules_for_detection_acceptance": MODULES,
        "serious_false_positive": {"max_observed_rate": 1, "max_one_sided_95_upper": 1},
        "serious_false_negative": {"max_observed_rate": 1, "max_one_sided_95_upper": 1},
        "valid_risk_output": {"minimum_rate": 0.95},
        "automatic_triage_coverage": {"minimum_rate": 0.8}, "failure": {"maximum_rate": 0.05}},
        "strata": {"required_media": ["image", "video"], "truth_types": ["real", "fully_generated", "face_manipulated"],
                   "minimum_acceptance_content_groups_per_media_truth": 1,
                   "minimum_key_subgroup_content_groups": 1, "key_subgroups": ["multiple_faces", "small_face", "source_dataset"]},
        "candidate_protocol": {"development_selection": {"serious_false_positive_group_increase_allowed": 0,
            "minimum_serious_false_negative_group_reduction": 1, "must_meet_automatic_coverage_and_failure_limits": True}}}


def row(index, truth="real", content=None, source="synthetic-source", media="image", stage="development", tags=None):
    return {"sample_id": f"sample-{index}", "content_id": content or f"original-{index}",
            "input_sha256": digest({"synthetic-input-marker": index}), "source_dataset_id": source,
            "media_type": media, "truth": truth, "split": stage, "required_module_ids": list(MODULES),
            "subgroups": tags if tags is not None else ["multiple_faces", "small_face"]}


def frozen(manifest=None, stage="development", config=None):
    manifest = deepcopy(manifest if manifest is not None else [row(0), row(1, "fully_generated"), row(2, "face_manipulated")])
    for item in manifest:
        item["split"] = stage
    candidates = [freeze_candidate(policy, baseline_release_sha256=RELEASE, baseline_rule_version="synthetic-legacy")
                  for policy in ("baseline", "existing_review_floor")]
    receipt = {"status": "passed", "receipt_sha256": "c" * 64, "stage": stage,
               "manifest_digest": digest(sorted(manifest, key=lambda item: item["sample_id"])),
               "protocol_file_sha256": PROTOCOL_FILE, "release_sha256": RELEASE}
    return freeze_experiment(manifest, config or protocol(), candidates, stage=stage,
        protocol_file_sha256=PROTOCOL_FILE, release_sha256=RELEASE, admission=receipt)


def candidate(experiment, policy="baseline"):
    return next(value["digest"] for value in experiment["candidates"] if value["policy"] == policy)


def results(experiment, policy="baseline", bands=None):
    output = []
    for item in experiment["manifest"]:
        band = (bands or {}).get(item["sample_id"], "none" if item["truth"] == "real" else "obvious")
        key = candidate(experiment, policy)
        output.append({"sample_id": item["sample_id"], "input_sha256": item["input_sha256"],
            "experiment_digest": experiment["digest"], "candidate_digest": key, "release_sha256": RELEASE,
            "final_band": band, "modules": {
                "general_aigc": {"status": "success", "score": -1.234},
                "face_branch_check": {"status": "no_face", "detected_total": 0, "detection_completed": True,
                                      "coverage_status": "recorded", "scores": [],
                                      "failure_counts": {"detection": 0, "crop": 0, "model": 0}},
                "canonical_decision": {"status": "success", "final_band": band, "candidate_digest": key,
                                       "release_sha256": RELEASE, "evidence_digest": digest({"synthetic": item["sample_id"]})},
                "evidence_report": {"status": "success", "persisted": True, "verified": True,
                    "evidence_digest": digest({"synthetic": item["sample_id"]}), "candidate_digest": key}}})
    return output


def evaluate(experiment, values=None, policy="baseline"):
    return evaluate_candidate(experiment, candidate(experiment, policy), values if values is not None else results(experiment, policy))


@pytest.mark.parametrize("errors,count,expected", [(0, 1, .95), (1, 2, math.sqrt(.95)),
    (4, 5, .95 ** .2), (0, 60, 1 - .05 ** (1 / 60)), (3, 3, 1.0)])
def test_exact_one_sided_upper_analytical_cases(errors, count, expected):
    assert one_sided_cp_upper(errors, count) == pytest.approx(expected, abs=1e-12)


def test_intervals_are_monotonic_and_empty_is_unknown():
    assert one_sided_cp_upper(0, 0) is None
    assert one_sided_cp_upper(0, 6000) > 0
    values = [one_sided_cp_upper(k, 60) for k in range(61)]
    assert values == sorted(values)
    for invalid in ((True, 2), (-1, 2), (3, 2), (1.0, 2)):
        with pytest.raises(ValueError):
            one_sided_cp_upper(*invalid)


def test_missing_results_stay_in_group_and_truth_denominators():
    exp = frozen()
    report = evaluate(exp, results(exp)[:1])
    assert report["overall"]["original_content_groups"] == 3
    assert report["missing_results_retained"] == 2
    assert report["overall"]["failure"]["numerator"] == 2
    assert report["overall"]["valid_risk_output"]["rate"] == pytest.approx(1 / 3)
    assert report["overall"]["serious_false_negative"]["fully_generated"]["denominator"] == 1
    assert report["overall"]["serious_false_negative"]["fully_generated"]["numerator"] == 0


def test_derivatives_do_not_inflate_n_and_worst_error_is_retained():
    exp = frozen([row(0, content="one-original"), row(1, content="one-original"), row(2, "unknown")])
    values = results(exp, bands={"sample-0": "none", "sample-1": "obvious"})
    report = evaluate(exp, values)
    assert report["overall"]["original_content_groups"] == 2
    assert report["overall"]["serious_false_positive"]["denominator"] == 1
    assert report["overall"]["serious_false_positive"]["numerator"] == 1
    assert report["overall"]["unknown_truth_groups"] == 1
    assert report["overall"]["valid_risk_output"]["denominator"] == 2


def test_all_versions_required_even_in_a_truth_or_subgroup_filter():
    exp = frozen([row(0, content="one", tags=["small_face"]),
                  row(1, "face_manipulated", content="one", tags=["multiple_faces"])])
    report = evaluate(exp, results(exp)[:1])
    assert report["overall"]["valid_risk_output"]["rate"] == 0
    assert report["by_media_truth"]["image/real"]["metrics"]["failure"]["rate"] == 1
    assert report["by_subgroup"]["small_face"]["metrics"]["valid_risk_output"]["rate"] == 0


def test_verified_no_face_is_valid_without_assigning_a_zero_score():
    exp = frozen()
    values = results(exp)
    report = evaluate(exp, values)
    assert report["overall"]["valid_risk_output"]["rate"] == 1
    assert values[0]["modules"]["face_branch_check"]["scores"] == []


@pytest.mark.parametrize("patch", [{"detection_completed": False}, {"coverage_status": "unknown"},
    {"detected_total": None}, {"detected_total": False}, {"detected_total": 2}, {"scores": [0]}, {"status": "failed"}])
def test_no_face_requires_actual_completed_detection_and_recorded_zero(patch):
    exp = frozen([row(0)])
    values = results(exp)
    values[0]["modules"]["face_branch_check"].update(patch)
    report = evaluate(exp, values)
    assert report["overall"]["valid_risk_output"]["rate"] == 0
    assert report["overall"]["failure"]["rate"] == 1


def test_detected_faces_need_a_valid_score_and_auxiliary_failure_does_not_erase_it():
    exp = frozen([row(0)])
    values = results(exp)
    values[0]["modules"]["face_branch_check"].update(status="success", detected_total=3, scores=[-2.345])
    values[0]["modules"]["whole_image_auxiliary"] = {"status": "failed", "score": None}
    report = evaluate(exp, values)
    assert report["overall"]["valid_risk_output"]["rate"] == 1
    assert report["overall"]["failure"]["rate"] == 0
    assert report["sample_outcomes"][0]["auxiliary_statuses"]["whole_image_auxiliary"]["status"] == "failed"
    values[0]["modules"]["face_branch_check"]["scores"] = []
    assert evaluate(exp, values)["overall"]["valid_risk_output"]["rate"] == 0


def test_review_valid_but_does_not_complete_automatic_triage():
    exp = frozen([row(0)])
    report = evaluate(exp, results(exp, bands={"sample-0": "review"}))
    assert report["overall"]["valid_risk_output"]["rate"] == 1
    assert report["overall"]["automatic_triage_coverage"]["rate"] == 0
    assert report["overall"]["failure"]["rate"] == 0


def test_invalid_output_does_not_erase_a_reported_serious_error():
    exp = frozen([row(0)])
    values = results(exp, bands={"sample-0": "obvious"})
    values[0]["modules"]["evidence_report"]["verified"] = False
    report = evaluate(exp, values)
    assert report["overall"]["serious_false_positive"]["numerator"] == 1
    assert report["overall"]["failure"]["numerator"] == 1


def test_actual_sources_and_missing_media_are_separate():
    exp = frozen([row(0, source="source-A"), row(1, source="source-B")])
    report = evaluate(exp)
    assert set(report["by_source_dataset"]) == {"source-A", "source-B"}
    assert report["by_media_truth"]["video/real"]["status"] == "insufficient_evidence"
    assert report["status"] == "insufficient_evidence"


@pytest.mark.parametrize("kind", ["extra", "duplicate", "wrong_input", "wrong_candidate", "wrong_experiment", "wrong_release"])
def test_result_integrity_failures_cannot_silently_drop_or_reassign_inputs(kind):
    exp = frozen()
    values = results(exp)
    if kind == "extra":
        values.append({"sample_id": "outside-frozen-universe"})
    elif kind == "duplicate":
        values.append(deepcopy(values[0]))
    else:
        key = {"wrong_input": "input_sha256", "wrong_candidate": "candidate_digest",
               "wrong_experiment": "experiment_digest", "wrong_release": "release_sha256"}[kind]
        values[0][key] = "d" * 64
    with pytest.raises(ValueError):
        evaluate(exp, values)


def test_nonfinite_success_is_retained_as_failed_without_removing_denominator():
    exp = frozen([row(0), row(1)])
    values = results(exp)
    values[0]["modules"]["general_aigc"]["score"] = float("nan")
    report = evaluate(exp, values)
    assert report["overall"]["failure"]["rate"] == .5
    assert report["overall"]["valid_risk_output"]["denominator"] == 2
    assert report["sample_outcomes"][0]["nonfinite_fields"] == ["general_aigc.score"]


def test_complete_undetermined_is_not_an_execution_failure():
    exp = frozen([row(0)])
    report = evaluate(exp, results(exp, bands={"sample-0": "undetermined"}))
    assert report["overall"]["valid_risk_output"]["rate"] == 0
    assert report["overall"]["automatic_triage_coverage"]["rate"] == 0
    assert report["overall"]["failure"]["rate"] == 0
    assert report["status_distribution"]["sample_status"] == {"undetermined": 1}


def test_undetermined_can_coexist_with_a_separate_execution_error():
    exp = frozen([row(0)])
    values = results(exp, bands={"sample-0": "undetermined"})
    values[0]["modules"]["general_aigc"] = {"status": "timeout"}
    report = evaluate(exp, values)
    assert report["overall"]["failure"]["rate"] == 1
    assert report["status_distribution"]["final_band"] == {"undetermined": 1}
    assert report["sample_outcomes"][0]["failed_modules"] == ["general_aigc"]


def test_explicit_partial_success_status_is_preserved_and_missing_contract_is_failure():
    exp = frozen([row(0)])
    values = results(exp)
    values[0]["modules"]["general_aigc"] = {"status": "partial_success", "reason": "synthetic documented limit",
        "execution_completed": True, "execution_failures": 0, "score": None}
    report = evaluate(exp, values)
    assert report["status_distribution"]["sample_status"] == {"partial_success": 1}
    assert report["overall"]["failure"]["rate"] == 0
    assert report["overall"]["valid_risk_output"]["rate"] == 0
    del values[0]["modules"]["general_aigc"]["execution_failures"]
    assert evaluate(exp, values)["overall"]["failure"]["rate"] == 1


def test_face_crop_or_model_failure_is_invalid_even_with_another_finite_score():
    exp = frozen([row(0)])
    values = results(exp)
    values[0]["modules"]["face_branch_check"].update(status="success", detected_total=3, scores=[-1.0])
    values[0]["modules"]["face_branch_check"]["failure_counts"]["crop"] = 1
    assert evaluate(exp, values)["overall"]["failure"]["rate"] == 1


def test_paired_fn_selection_counts_one_original_across_two_fake_truth_types():
    exp = frozen([row(0, "fully_generated", content="one"), row(1, "face_manipulated", content="one")])
    baseline = evaluate(exp, results(exp, bands={"sample-0": "none", "sample-1": "none"}))
    alternative = evaluate(exp, results(exp, "existing_review_floor"), "existing_review_floor")
    differences = paired_differences(baseline, alternative)
    assert differences["serious_false_negative_any"]["net_candidate_minus_baseline"] == -1
    selected = select_development_candidate(exp, [baseline, alternative])
    assert selected["candidate_checks"][candidate(exp, "existing_review_floor")]["serious_fn_group_reduction"] == 1


def test_changed_frozen_protocol_and_unregistered_candidate_are_rejected():
    exp = frozen()
    changed = deepcopy(exp)
    changed["protocol"]["metrics"]["failure"]["maximum_rate"] = 1
    with pytest.raises(ValueError, match="changed"):
        evaluate(changed, [])
    with pytest.raises(ValueError, match="declared"):
        evaluate_candidate(exp, "d" * 64, [])


@pytest.mark.parametrize("problem", ["byte_alias", "source_conflict", "missing_content", "duplicate_id", "missing_core"])
def test_manifest_requires_original_group_metadata(problem):
    manifest = [row(0), row(1)]
    if problem == "byte_alias":
        manifest[1]["input_sha256"] = manifest[0]["input_sha256"]
    elif problem == "source_conflict":
        manifest[1]["content_id"] = manifest[0]["content_id"]
        manifest[1]["source_dataset_id"] = "different-source"
    elif problem == "missing_content":
        del manifest[1]["content_id"]
    elif problem == "duplicate_id":
        manifest[1]["sample_id"] = manifest[0]["sample_id"]
    else:
        manifest[0]["required_module_ids"].pop()
    with pytest.raises(ValueError):
        frozen(manifest)


def test_admission_receipt_is_required_and_bound_before_results():
    with pytest.raises(ValueError, match="Admission"):
        freeze_experiment([row(0)], protocol(), [freeze_candidate("baseline", baseline_release_sha256=RELEASE, baseline_rule_version="s")],
            stage="development", protocol_file_sha256=PROTOCOL_FILE, release_sha256=RELEASE, admission={"status": "passed"})


def test_pairing_uses_one_original_group_and_forbids_different_experiments():
    exp = frozen([row(0, "fully_generated", content="one"), row(1, "fully_generated", content="one")])
    baseline = evaluate(exp, results(exp, bands={"sample-0": "none", "sample-1": "none"}))
    alternative = evaluate(exp, results(exp, "existing_review_floor"), "existing_review_floor")
    pair = paired_differences(baseline, alternative)
    assert pair["paired_group_count"] == 1
    assert pair["serious_false_negative"]["fully_generated"]["net_candidate_minus_baseline"] == -1
    with pytest.raises(ValueError, match="same frozen"):
        paired_differences(baseline, evaluate(frozen([row(8)])))


def test_predeclared_development_selection_and_default_retention():
    exp = frozen([row(index, "fully_generated") for index in range(10)])
    baseline = evaluate(exp, results(exp, bands={"sample-0": "none"}))
    alternative = evaluate(exp, results(exp, "existing_review_floor", bands={"sample-0": "review"}), "existing_review_floor")
    selected = select_development_candidate(exp, [baseline, alternative])
    assert selected["selected_candidate_digest"] == candidate(exp, "existing_review_floor")
    assert selected["acceptance_authorized"] is False
    no_improvement = evaluate(exp, results(exp, "existing_review_floor", bands={"sample-0": "none"}), "existing_review_floor")
    assert select_development_candidate(exp, [baseline, no_improvement])["baseline_retained"] is True


def test_development_requires_coverage_failure_and_no_fp_increase():
    exp = frozen([row(0), row(1, "fully_generated")])
    baseline = evaluate(exp, results(exp, bands={"sample-1": "none"}))
    review = evaluate(exp, results(exp, "existing_review_floor", bands={"sample-1": "review"}), "existing_review_floor")
    assert select_development_candidate(exp, [baseline, review])["baseline_retained"] is True
    fp = evaluate(exp, results(exp, "existing_review_floor", bands={"sample-0": "obvious"}), "existing_review_floor")
    assert select_development_candidate(exp, [baseline, fp])["baseline_retained"] is True
    with pytest.raises(ValueError, match="All predeclared"):
        select_development_candidate(exp, [baseline])


def test_acceptance_comparison_is_descriptive_and_never_selects():
    exp = frozen(stage="acceptance")
    baseline = evaluate(exp)
    alternative = evaluate(exp, results(exp, "existing_review_floor"), "existing_review_floor")
    assert "no retuning" in paired_differences(baseline, alternative)["permitted_use"]
    with pytest.raises(ValueError, match="Acceptance"):
        select_development_candidate(exp, [baseline, alternative])


def test_unknown_truth_remains_functional_denominator_and_cannot_pass_a_capability():
    exp = frozen([row(0, "unknown")])
    report = evaluate(exp)
    assert report["overall"]["unknown_truth_groups"] == 1
    assert report["overall"]["serious_false_positive"]["denominator"] == 0
    assert report["by_source_dataset"]["synthetic-source"]["status"] == "insufficient_evidence"


def test_only_parameters_fixed_before_results_control_limits():
    config = protocol()
    config["metrics"]["serious_false_positive"] = {"max_observed_rate": .05, "max_one_sided_95_upper": .1}
    exp = frozen([row(index) for index in range(60)], config=config)
    report = evaluate(exp)
    assert report["overall"]["serious_false_positive"]["passes_limits"] is True
    assert report["overall"]["serious_false_positive"]["one_sided_95_upper"] == pytest.approx(.0487029, abs=1e-7)
