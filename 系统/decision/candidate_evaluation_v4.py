"""Fixed-universe, original-content-group evaluation without media/model access.

All input metadata, protocol, candidate descriptors and admission receipts must
be frozen by the coordinator before outcomes are provided. The caller verifies
receipt authenticity and file provenance; this pure module checks bindings,
not filesystem policy or trust. No parameter search or acceptance selection.
"""
from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy
import math

from .rule_candidate_v4 import digest, require_sha256, seal, verify_candidate, verify_seal

VERSION = "v4-content-group-evaluation-1"
CORE_MODULES = frozenset({"general_aigc", "face_branch_check", "canonical_decision", "evidence_report"})
ERROR_STATUSES = frozenset({"failed", "error", "timeout", "cancelled", "missing", "unrecorded", "invalid", "skipped", "unavailable"})
KNOWN_TRUTHS = frozenset({"real", "fully_generated", "face_manipulated"})
VALID_BANDS = frozenset({"none", "review", "obvious"})


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be explicit")
    return value


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _rate(value, label):
    if not _finite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid protocol rate: {label}")


def validate_protocol(protocol):
    # Numerical limits are parameters supplied by the frozen protocol, never
    # inferred from development or acceptance outcomes.
    digest(protocol)
    if not isinstance(protocol, Mapping) or not protocol.get("protocol_id"):
        raise ValueError("Protocol identifier is required")
    metrics, strata = protocol["metrics"], protocol["strata"]
    if set(metrics["required_modules_for_detection_acceptance"]) != CORE_MODULES:
        raise ValueError("Protocol must explicitly name the four detection modules")
    for key in ("serious_false_positive", "serious_false_negative"):
        for limit in ("max_observed_rate", "max_one_sided_95_upper"):
            _rate(metrics[key][limit], f"{key}.{limit}")
    for key, limit in (("valid_risk_output", "minimum_rate"), ("automatic_triage_coverage", "minimum_rate"), ("failure", "maximum_rate")):
        _rate(metrics[key][limit], f"{key}.{limit}")
    for key in ("minimum_acceptance_content_groups_per_media_truth", "minimum_key_subgroup_content_groups"):
        if not _integer(strata[key]) or strata[key] < 1:
            raise ValueError("Positive minimum original-group counts required")
    if set(strata["truth_types"]) != KNOWN_TRUTHS or not set(strata["required_media"]) <= {"image", "video"}:
        raise ValueError("Unsupported protocol strata")


def freeze_experiment(manifest, protocol, candidates, *, stage, protocol_file_sha256,
                      release_sha256, admission):
    """Freeze explicit metadata, never enumerate directories or load data.

    admission = {status: passed, receipt_sha256, manifest_digest,
                 protocol_file_sha256, release_sha256, stage}.
    These are bindings to an externally verified gate receipt, not a substitute
    for the exclusion and source/license checks performed by that gate.
    """
    if stage not in {"development", "acceptance"}:
        raise ValueError("Explicit development or acceptance stage required")
    validate_protocol(protocol)
    require_sha256(protocol_file_sha256, "protocol_file_sha256")
    require_sha256(release_sha256, "release_sha256")
    rows, ids = [], set()
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("An explicit nonempty fixed-universe manifest is required")
    for supplied in manifest:
        if not isinstance(supplied, Mapping):
            raise ValueError("Invalid manifest row")
        row = deepcopy(dict(supplied))
        for key in ("sample_id", "content_id", "source_dataset_id"):
            _text(row.get(key), key)
        if row["sample_id"] in ids:
            raise ValueError("Duplicate sample identifiers")
        ids.add(row["sample_id"])
        require_sha256(row.get("input_sha256"), "input_sha256")
        if row.get("truth") not in KNOWN_TRUTHS | {"unknown"}:
            raise ValueError("Unknown truth must use the explicit unknown label")
        if row.get("media_type") not in {"image", "video"} or row.get("split") != stage:
            raise ValueError("Media type or split differs from experiment")
        required = row.get("required_module_ids")
        if not isinstance(required, list) or len(required) != len(set(required)) or set(required) != CORE_MODULES:
            raise ValueError("Detection list must retain exactly the four core modules; retrieval is separate")
        tags = row.get("subgroups")
        if not isinstance(tags, list) or not all(isinstance(tag, str) and tag and tag != "source_dataset" for tag in tags) or len(set(tags)) != len(tags):
            raise ValueError("Subgroups must be explicit unique tags; source_dataset is derived from its actual ID")
        rows.append(row)
    rows.sort(key=lambda row: row["sample_id"])
    # Re-encoded aliases with the same content_id remain one statistical unit.
    # Identical input bytes may not be labelled as independent content groups.
    hash_groups, content_sources = {}, {}
    for row in rows:
        for table, key, value in ((hash_groups, row["input_sha256"], row["content_id"]),
                                  (content_sources, row["content_id"], row["source_dataset_id"])):
            if key in table and table[key] != value:
                raise ValueError("Byte aliases/content ancestry have inconsistent group or source metadata")
            table[key] = value
    descriptors = []
    for candidate in candidates:
        verify_candidate(candidate)
        descriptors.append(deepcopy(dict(candidate)))
    if len({value["digest"] for value in descriptors}) != len(descriptors):
        raise ValueError("Duplicate candidate descriptor")
    if len([value for value in descriptors if value["policy"] == "baseline"]) != 1:
        raise ValueError("Exactly one frozen baseline is required")
    if len({(value["baseline_release_sha256"], value["baseline_rule_version"]) for value in descriptors}) != 1:
        raise ValueError("Candidates must share the same frozen baseline")
    bindings = {"stage": stage, "manifest_digest": digest(rows),
                "protocol_file_sha256": protocol_file_sha256, "release_sha256": release_sha256}
    if not isinstance(admission, Mapping) or admission.get("status") != "passed" or any(admission.get(k) != v for k, v in bindings.items()):
        raise ValueError("Admission receipt must bind stage, exact manifest, protocol and frozen release")
    require_sha256(admission.get("receipt_sha256"), "admission.receipt_sha256")
    return seal({"version": VERSION, **bindings, "protocol": deepcopy(dict(protocol)),
                 "protocol_digest": digest(protocol), "manifest": rows,
                 "candidates": sorted(descriptors, key=lambda value: value["digest"]),
                 "admission": deepcopy(dict(admission)),
                 "receipt_authenticity": "must_be_verified_by_coordinator"})


def one_sided_cp_upper(errors, count, confidence=0.95):
    """Exact binomial Clopper-Pearson upper bound (one sided), stable at k=0."""
    if not _integer(errors) or not _integer(count) or errors > count:
        raise ValueError("Binomial counts require 0 <= errors <= count")
    if not _finite(confidence) or not 0 < confidence < 1:
        raise ValueError("Confidence must be between zero and one")
    if count == 0:
        return None
    if errors == count:
        return 1.0
    alpha = 1 - confidence
    if errors == 0:
        return -math.expm1(math.log(alpha) / count)
    coefficients = [math.lgamma(count + 1) - math.lgamma(i + 1) - math.lgamma(count - i + 1) for i in range(errors + 1)]
    low, high = errors / count, 1.0
    for _ in range(80):
        probability = (low + high) / 2
        if probability in (low, high):
            break
        logp, logq = math.log(probability), math.log1p(-probability)
        terms = [value + i * logp + (count - i) * logq for i, value in enumerate(coefficients)]
        maximum = max(terms)
        cdf = math.exp(maximum) * sum(math.exp(value - maximum) for value in terms)
        if cdf > alpha:
            low = probability
        else:
            high = probability
    return (low + high) / 2


def _module_valid(name, value, result):
    if not isinstance(value, Mapping):
        return False
    if name == "face_branch_check":
        total = value.get("detected_total")
        if value.get("detection_completed") is not True or value.get("coverage_status") != "recorded" or not _integer(total):
            return False
        failures = value.get("failure_counts")
        if not isinstance(failures, Mapping) or any(not _integer(failures.get(key)) or failures[key] != 0 for key in ("detection", "crop", "model")):
            return False
        if total == 0:
            return value.get("status") == "no_face" and value.get("scores") == []
        scores = value.get("scores")
        return (value.get("status") == "success" and isinstance(scores, list) and 0 < len(scores) <= total
                and all(_finite(score) for score in scores))
    if value.get("status") != "success":
        return False
    if name == "general_aigc":
        return _finite(value.get("score"))
    if name == "canonical_decision":
        return (result.get("final_band") in VALID_BANDS and value.get("final_band") == result.get("final_band")
                and value.get("candidate_digest") == result.get("candidate_digest")
                and value.get("release_sha256") == result.get("release_sha256")
                and isinstance(value.get("evidence_digest"), str) and len(value["evidence_digest"]) == 64
                and all(char in "0123456789abcdef" for char in value["evidence_digest"]))
    if name == "evidence_report":
        canonical = result.get("modules", {}).get("canonical_decision", {})
        return (value.get("persisted") is True and value.get("verified") is True
                and isinstance(canonical, Mapping)
                and value.get("evidence_digest") == canonical.get("evidence_digest")
                and value.get("candidate_digest") == result.get("candidate_digest"))
    return False


def _finite_json(value):
    try:
        digest(value)
        return True
    except (ValueError, TypeError):
        return False


def _module_failed(name, value, result, valid):
    if not isinstance(value, Mapping) or not _finite_json(value):
        return True
    status = value.get("status", "unrecorded")
    if status in ERROR_STATUSES:
        return True
    if valid:
        return False
    if name == "canonical_decision" and result.get("final_band") == "undetermined":
        # A complete explicit inability to judge is not an execution failure.
        probe, module = deepcopy(dict(result)), deepcopy(dict(value))
        probe["final_band"] = module["final_band"] = "none"
        module["status"] = "success"
        return not (status in {"success", "undetermined"} and _module_valid(name, module, probe))
    if name != "evidence_report" and status in {"undetermined", "partial_success"}:
        complete = (value.get("execution_completed") is True and isinstance(value.get("reason"), str)
                    and bool(value["reason"]) and _integer(value.get("execution_failures"))
                    and value["execution_failures"] == 0)
        if name == "general_aigc":
            complete = complete and "score" in value and (value["score"] is None or _finite(value["score"]))
        if name == "face_branch_check":
            complete = complete and all(key in value for key in ("detected_total", "detection_completed", "coverage_status", "scores", "failure_counts"))
            failures = value.get("failure_counts", {})
            complete = complete and isinstance(failures, Mapping) and all(_integer(failures.get(key)) and failures[key] == 0 for key in ("detection", "crop", "model"))
        return not complete
    # Missing contract fields and invalid values labelled success are failures.
    return True


def _record_nonfinite(value, path="", paths=None):
    paths = paths if paths is not None else []
    if isinstance(value, float) and not math.isfinite(value):
        paths.append(path)
        return None
    if isinstance(value, Mapping):
        return {key: _record_nonfinite(child, f"{path}.{key}" if path else str(key), paths) for key, child in value.items()}
    if isinstance(value, list):
        return [_record_nonfinite(child, f"{path}[{index}]", paths) for index, child in enumerate(value)]
    return deepcopy(value)


def _sample_outcome(row, result, experiment, candidate_digest):
    if result is None:
        return {**row, "final_band": "undetermined", "valid": False, "automatic": False,
                "failed": True, "status": "missing_result", "invalid_modules": sorted(CORE_MODULES),
                "failed_modules": sorted(CORE_MODULES), "module_statuses": {}, "auxiliary_statuses": {},
                "nonfinite_fields": []}
    for key, expected in (("input_sha256", row["input_sha256"]), ("candidate_digest", candidate_digest),
                          ("release_sha256", experiment["release_sha256"]), ("experiment_digest", experiment["digest"])):
        if result.get(key) != expected:
            raise ValueError(f"Result {row['sample_id']} has a mismatched {key}")
    modules = result.get("modules", {})
    if not isinstance(modules, Mapping):
        raise ValueError("Module outcomes must be explicit objects")
    valid = {key: _module_valid(key, modules.get(key), result) for key in row["required_module_ids"]}
    for key in valid:
        valid[key] = valid[key] and _finite_json(modules.get(key))
    failed_modules = [key for key in valid if _module_failed(key, modules.get(key), result, valid[key])]
    statuses = {key: modules[key].get("status", "unrecorded") if isinstance(modules.get(key), Mapping) else "missing"
                for key in row["required_module_ids"]}
    band = result.get("final_band")
    if band not in VALID_BANDS | {"undetermined", None}:
        raise ValueError("Non-canonical final band; adapter must explicitly map legacy bands")
    # Explicit, complete undetermined/partial_success outcomes stay distinct
    # from execution errors. Review is valid but not automatic completion.
    failed = bool(failed_modules)
    status = ("failed" if failed else "undetermined" if band in {None, "undetermined"} or "undetermined" in statuses.values()
              else "partial_success" if "partial_success" in statuses.values() else "success")
    nonfinite = []
    recorded = _record_nonfinite(modules, paths=nonfinite)
    return {**row, "final_band": band or "undetermined", "valid": all(valid.values()),
            "automatic": all(valid.values()) and band in {"none", "obvious"}, "failed": failed,
            "status": status,
            "invalid_modules": sorted(key for key, passed in valid.items() if not passed),
            "failed_modules": sorted(failed_modules), "module_statuses": statuses,
            "nonfinite_fields": nonfinite,
            "auxiliary_statuses": {key: value for key, value in recorded.items() if key not in CORE_MODULES}}


def _groups(samples):
    grouped = defaultdict(list)
    for row in samples:
        grouped[row["content_id"]].append(row)
    return [{"content_id": key, "sample_ids": sorted(row["sample_id"] for row in rows),
             "valid": all(row["valid"] for row in rows), "automatic": all(row["automatic"] for row in rows),
             "failed": any(row["failed"] for row in rows),
             "serious_false_positive": any(row["truth"] == "real" and row["final_band"] == "obvious" for row in rows),
             "serious_false_negative": {truth: any(row["truth"] == truth and row["final_band"] == "none" for row in rows)
                                        for truth in ("fully_generated", "face_manipulated")},
             "truths": sorted({row["truth"] for row in rows})}
            for key, rows in sorted(grouped.items())]


def _fraction(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def _error_metric(groups, truth, policy):
    selected = [row for row in groups if truth in row["truths"]]
    count = sum(row["serious_false_positive"] if truth == "real" else row["serious_false_negative"][truth] for row in selected)
    value = _fraction(count, len(selected))
    value["one_sided_95_upper"] = one_sided_cp_upper(count, len(selected))
    value["passes_limits"] = bool(selected) and value["rate"] <= policy["max_observed_rate"] and value["one_sided_95_upper"] <= policy["max_one_sided_95_upper"]
    return value


def _metrics(groups, protocol):
    limits, count = protocol["metrics"], len(groups)
    value = {"original_content_groups": count,
             "serious_false_positive": _error_metric(groups, "real", limits["serious_false_positive"]),
             "serious_false_negative": {truth: _error_metric(groups, truth, limits["serious_false_negative"])
                                        for truth in ("fully_generated", "face_manipulated")}}
    fake_groups = [row for row in groups if set(row["truths"]) & {"fully_generated", "face_manipulated"}]
    value["serious_false_negative_any"] = _fraction(sum(any(row["serious_false_negative"].values()) for row in fake_groups), len(fake_groups))
    for key, field, limit in (("valid_risk_output", "valid", "minimum_rate"),
                               ("automatic_triage_coverage", "automatic", "minimum_rate"), ("failure", "failed", "maximum_rate")):
        item = _fraction(sum(row[field] for row in groups), count)
        item["passes_limits"] = bool(count) and (item["rate"] >= limits[key][limit] if limit == "minimum_rate" else item["rate"] <= limits[key][limit])
        value[key] = item
    value["unknown_truth_groups"] = sum("unknown" in row["truths"] for row in groups)
    return value


def _stratum(samples, protocol, minimum, all_groups):
    groups = _groups(samples)
    # A failed sibling variant cannot disappear when a subgroup or truth filter
    # is applied. Error types remain specific to the selected truth stratum.
    full = {row["content_id"]: row for row in all_groups}
    for group in groups:
        for key in ("valid", "automatic", "failed"):
            group[key] = full[group["content_id"]][key]
    metrics = _metrics(groups, protocol)
    applicable = [metrics["serious_false_positive"]] if any(row["truth"] == "real" for row in samples) else []
    applicable += [metrics["serious_false_negative"][truth] for truth in ("fully_generated", "face_manipulated")
                   if any(row["truth"] == truth for row in samples)]
    passed = all(value["passes_limits"] for value in applicable) and all(metrics[key]["passes_limits"] for key in ("valid_risk_output", "automatic_triage_coverage", "failure"))
    sufficient = len(groups) >= minimum and bool(applicable)
    return {"sample_count": len(samples), "minimum_original_content_groups": minimum,
            "status": "insufficient_evidence" if not sufficient else "passed" if passed else "failed",
            "metrics": metrics}


def evaluate_candidate(experiment, candidate_digest, results):
    """Evaluate every frozen input, including absent/failed/unknown outputs."""
    verify_seal(experiment)
    if experiment.get("version") != VERSION:
        raise ValueError("Unsupported evaluator version")
    descriptors = {row["digest"]: row for row in experiment["candidates"]}
    if candidate_digest not in descriptors:
        raise ValueError("Candidate was not declared before results")
    known_ids = {row["sample_id"] for row in experiment["manifest"]}
    supplied = {}
    for result in results:
        if not isinstance(result, Mapping) or result.get("sample_id") not in known_ids or result["sample_id"] in supplied:
            raise ValueError("Unknown or duplicate result input")
        supplied[result["sample_id"]] = result
    outcomes = [_sample_outcome(row, supplied.get(row["sample_id"]), experiment, candidate_digest) for row in experiment["manifest"]]
    protocol = experiment["protocol"]
    strata = protocol["strata"]
    groups = _groups(outcomes)
    by_media_truth = {}
    for media in strata["required_media"]:
        for truth in strata["truth_types"]:
            subset = [row for row in outcomes if row["media_type"] == media and row["truth"] == truth]
            by_media_truth[f"{media}/{truth}"] = _stratum(subset, protocol, strata["minimum_acceptance_content_groups_per_media_truth"], groups)
    by_subgroup = {tag: _stratum([row for row in outcomes if tag in row["subgroups"]], protocol,
                               strata["minimum_key_subgroup_content_groups"], groups)
                   for tag in strata["key_subgroups"] if tag != "source_dataset"}
    by_source = {source: _stratum([row for row in outcomes if row["source_dataset_id"] == source], protocol,
                                strata["minimum_key_subgroup_content_groups"], groups)
                 for source in sorted({row["source_dataset_id"] for row in outcomes})}
    overall = _metrics(groups, protocol)
    checks = list(by_media_truth.values()) + list(by_subgroup.values()) + list(by_source.values())
    overall_pass = all(overall[key]["passes_limits"] for key in ("serious_false_positive", "valid_risk_output", "automatic_triage_coverage", "failure")) and all(value["passes_limits"] for value in overall["serious_false_negative"].values())
    status = ("insufficient_evidence" if any(value["status"] == "insufficient_evidence" for value in checks)
              else "passed" if overall_pass and all(value["status"] == "passed" for value in checks) else "failed")
    return seal({"version": VERSION, "experiment_digest": experiment["digest"], "stage": experiment["stage"],
        "candidate_digest": candidate_digest, "manifest_digest": experiment["manifest_digest"],
        "protocol_digest": experiment["protocol_digest"], "status": status,
        "overall": overall, "by_media_truth": by_media_truth, "by_subgroup": by_subgroup, "by_source_dataset": by_source,
        "groups": groups, "sample_outcomes": outcomes,
        "status_distribution": {"sample_status": dict(Counter(row["status"] for row in outcomes)),
            "final_band": dict(Counter(row["final_band"] for row in outcomes)),
            "module_status": {module: dict(Counter(row["module_statuses"].get(module, "missing") for row in outcomes))
                              for module in sorted(CORE_MODULES)}},
        "unit": "original_content_group", "missing_results_retained": len(known_ids - supplied.keys()),
        "limits": "Numerical protocol checks only; not authorization to deploy or claim model effectiveness."})


def paired_differences(baseline, candidate):
    """Paired original-group discordances, never independent per-variant n."""
    verify_seal(baseline)
    verify_seal(candidate)
    for key in ("experiment_digest", "manifest_digest", "protocol_digest", "stage"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError("Paired comparison requires the same frozen universe and protocol")
    before, after = ({row["content_id"]: row for row in report["groups"]} for report in (baseline, candidate))
    if before.keys() != after.keys():
        raise ValueError("Paired content groups differ")
    def compare(selector):
        removed = sorted(key for key in before if selector(before[key]) and not selector(after[key]))
        added = sorted(key for key in before if not selector(before[key]) and selector(after[key]))
        return {"baseline_only": removed, "candidate_only": added, "net_candidate_minus_baseline": len(added) - len(removed)}
    return {"unit": "original_content_group", "paired_group_count": len(before),
            "serious_false_positive": compare(lambda row: row["serious_false_positive"]),
            "serious_false_negative": {truth: compare(lambda row: row["serious_false_negative"][truth]) for truth in ("fully_generated", "face_manipulated")},
            "serious_false_negative_any": compare(lambda row: any(row["serious_false_negative"].values())),
            "automatic_completion": compare(lambda row: row["automatic"]),
            "failure": compare(lambda row: row["failed"]),
            "permitted_use": "descriptive_only_on_acceptance; no retuning or candidate selection"}


def select_development_candidate(experiment, reports):
    """Select only among the predeclared development candidates, else baseline."""
    verify_seal(experiment)
    if experiment.get("stage") != "development":
        raise ValueError("Acceptance outcomes must never select or retune candidates")
    candidates = {row["digest"]: row for row in experiment["candidates"]}
    by_candidate = {}
    for report in reports:
        verify_seal(report)
        key = report.get("candidate_digest")
        if report.get("experiment_digest") != experiment["digest"] or report.get("stage") != "development" or key not in candidates or key in by_candidate:
            raise ValueError("Report is not bound to this development experiment")
        by_candidate[key] = report
    if set(by_candidate) != set(candidates):
        raise ValueError("All predeclared candidates, including failed candidates, must be reported")
    baseline_key = next(key for key, row in candidates.items() if row["policy"] == "baseline")
    baseline = by_candidate[baseline_key]
    selection = experiment["protocol"]["candidate_protocol"]["development_selection"]
    eligible, decisions = [], {}
    for key, report in by_candidate.items():
        if key == baseline_key:
            continue
        difference = paired_differences(baseline, report)
        fp_delta = difference["serious_false_positive"]["net_candidate_minus_baseline"]
        fn_reduction = -difference["serious_false_negative_any"]["net_candidate_minus_baseline"]
        limits_pass = all(report["overall"][name]["passes_limits"] for name in ("automatic_triage_coverage", "failure"))
        qualifies = (fp_delta <= selection["serious_false_positive_group_increase_allowed"]
                     and fn_reduction >= selection["minimum_serious_false_negative_group_reduction"]
                     and (limits_pass or not selection["must_meet_automatic_coverage_and_failure_limits"]))
        decisions[key] = {"eligible": qualifies, "serious_fp_group_delta": fp_delta,
                          "serious_fn_group_reduction": fn_reduction, "coverage_and_failure_limits_pass": limits_pass}
        if qualifies:
            errors = report["overall"]["serious_false_negative_any"]["numerator"]
            coverage = report["overall"]["automatic_triage_coverage"]["rate"]
            eligible.append((errors, -coverage, candidates[key]["complexity_rank"], key))
    selected = min(eligible)[3] if eligible else baseline_key
    return {"experiment_digest": experiment["digest"], "stage": "development", "selected_candidate_digest": selected,
            "baseline_retained": selected == baseline_key, "candidate_checks": decisions,
            "acceptance_authorized": False, "reason": "predeclared_development_criteria" if eligible else "no_candidate_qualifies"}
