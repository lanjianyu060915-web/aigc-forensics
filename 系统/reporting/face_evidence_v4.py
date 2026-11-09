"""Join query observations conservatively; never change scores or decisions."""
from collections.abc import Mapping
from copy import deepcopy
import math

from .face_bindings import bind_faces


def _rows(value):
    return [dict(v) for v in value if isinstance(v, Mapping)] if isinstance(value, list) else []


def _mapping(value):
    return dict(value) if isinstance(value, Mapping) else {}


def _scope(value):
    scope = _mapping(value)
    return scope if (scope.get("coordinate_space") == "original_image"
                     and all(isinstance(scope.get(key), str) and scope[key] for key in ("source_id", "frame_id"))) else {}


def _status(value):
    return value if isinstance(value, str) else "unrecorded"


def _binding_state(binding, scope, observations):
    state = binding["binding_status"]
    if state in {"matched", "ambiguous"}:
        return state
    if not scope or not observations or binding.get("reason") == "missing_original_coordinates_or_scope":
        return "unknown"
    for row in observations:
        box = row.get("bbox")
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box)
                or box[2] <= box[0] or box[3] <= box[1]):
            return "unknown"
    return state


def _pair_measurement_state(candidate, compared, row):
    """Only count an actual comparison, never convert its result into a verdict."""
    states = {_status(candidate.get("status")), _status(row.get("processing_status")), _status(row.get("status"))}
    if states & {"error", "failed", "timeout"} or compared.get("alignment_status") == "failed":
        return "failed"
    if states & {"skipped", "not_applicable"}:
        return "skipped"
    if row.get("match_status") == "unmatched" or row.get("status") == "unmatched":
        return "unmatched"
    cosine = row.get("cosine_similarity")
    measured = isinstance(cosine, (int, float)) and not isinstance(cosine, bool) and math.isfinite(cosine)
    if (candidate.get("status") == "success" and compared.get("alignment_status") == "completed"
            and row.get("match_status") == "matched" and row.get("reference_face_id") is not None and measured):
        return "processed"
    return "unknown"


def _combined_pair_state(states):
    if not states:
        return "unknown"
    if "processed" in states:
        return "processed" if all(value == "processed" for value in states) else "partial"
    for value in ("ambiguous", "unknown", "failed", "unmatched", "skipped"):
        if value in states:
            return value
    return "unknown"


def build_face_coverage_evidence(face_audit, identity, pair):
    identity, pair = _mapping(identity), _mapping(pair)
    audit = deepcopy(face_audit) if isinstance(face_audit, Mapping) else {
        "status": "unrecorded", "face_checks": [], "frame_records": [],
        "coverage": {"count_status": "unrecorded", "detected_total": None}}
    checks = _rows(audit.get("face_checks"))
    scope = _scope(identity.get("query_face_scope"))
    identity_faces = [{**scope, "face_uid": f"identity-query-{index}",
                       # First selection may differ from the face actually used
                       # for the embedding after masking and redetection.
                       "bbox": row.get("embedding_face_bbox_original"),
                       "keypoints_5": row.get("embedding_face_keypoints_original"),
                       "selection_trace": deepcopy(_mapping(row.get("selection_trace"))),
                       "status": _status(row.get("status"))}
                      for index, row in enumerate(_rows(identity.get("faces")))]
    identity_binding = bind_faces(checks, identity_faces, evidence_module="identity")
    pair_bindings = []
    pair_checks = [[] for _ in checks]
    for candidate in _rows(pair.get("candidates")):
        compared = _mapping(candidate.get("comparison"))
        rows = _rows(compared.get("faces"))
        scoped = [{**_scope(pair.get("query_face_scope")), "face_uid": f"pair-query-{index}",
                   "bbox": row.get("query_bbox_original"), "status": row.get("status", "unrecorded"),
                   "measurement_status": _pair_measurement_state(candidate, compared, row)}
                  for index, row in enumerate(rows)]
        bound = bind_faces(checks, scoped, evidence_module="pair_query")
        for binding in bound["rows"]:
            state = _binding_state(binding, _scope(pair.get("query_face_scope")), scoped)
            if state == "matched":
                state = scoped[binding["evidence_record_index"]]["measurement_status"]
            pair_checks[binding["query_record_index"]].append({"candidate_rank": candidate.get("rank"),
                "binding_status": binding["binding_status"], "processing_status": state,
                "binding_reason": binding.get("reason"), "candidate_status": _status(candidate.get("status"))})
        pair_bindings.append({"candidate_rank": candidate.get("rank"), "candidate_status": _status(candidate.get("status")),
                              "binding": bound, "observations": scoped,
                              "reference_correspondence": [{key: row.get(key) for key in (
                                  "query_face_id", "reference_face_id", "reference_bbox_original", "status", "match_status", "reason", "match_iou")}
                                                           for row in rows]})
    # Matching geometry establishes only which observation was checked, never
    # that two people share an identity or that the image is manipulated.
    for row in identity_binding["rows"]:
        check = checks[row["query_record_index"]]
        check["identity_binding_status"] = row["binding_status"]
        check["identity_processed"] = None
        check["identity_processing_status"] = _binding_state(row, scope, identity_faces)
        if row["binding_status"] == "matched":
            status = identity_faces[row["evidence_record_index"]]["status"]
            check["identity_status"] = status
            check["identity_processed"] = True if status == "success" else False if status in {"error", "failed", "timeout", "skipped"} else None
            check["identity_processing_status"] = ("processed" if status == "success" else "failed" if status in {"error", "failed", "timeout"}
                                                    else "skipped" if status == "skipped" else "unknown")
    for index, check in enumerate(checks):
        attempts = pair_checks[index]
        state = _combined_pair_state([entry["processing_status"] for entry in attempts])
        check["pair_processing_status"] = state
        check["pair_processed"] = True if state in {"processed", "partial"} else False if state in {"failed", "skipped", "unmatched"} else None
        check["pair_checks"] = attempts
    audit["face_checks"] = checks
    audit["identity_coverage"] = {"status": "recorded" if scope else "unrecorded",
                                  "recorded_query_observations": len(identity_faces),
                                  "bound_count": sum(r["binding_status"] == "matched" for r in identity_binding["rows"]),
                                  "processed_bound_count": sum(r.get("identity_processed") is True for r in checks),
                                  "module_status": _status(identity.get("status")),
                                  "query_observation_count": len(checks), "count_unit": "query_face_observations",
                                  "failed_bound_count": sum(r.get("identity_processing_status") == "failed" for r in checks),
                                  "ambiguous_count": sum(r.get("identity_processing_status") == "ambiguous" for r in checks),
                                  "unmatched_count": sum(r.get("identity_processing_status") == "unmatched" for r in checks),
                                  "unknown_count": sum(r.get("identity_processing_status") == "unknown" for r in checks),
                                  "skipped_count": sum(r.get("identity_processing_status") == "skipped" for r in checks),
                                  "module_reported_coverage": deepcopy(_mapping(_mapping(identity.get("face_search")).get("coverage"))),
                                  "module_face_checks": deepcopy(_rows(_mapping(identity.get("face_search")).get("face_checks")))}
    audit["pair_coverage"] = {"status": "recorded" if _scope(pair.get("query_face_scope")) else "unrecorded",
                              "module_status": _status(pair.get("status")),
                              "candidate_count": len(pair_bindings), "scope": pair.get("scope", "unrecorded"),
                              "module_detected_total": pair.get("query_face_count"),
                              "query_observation_count": len(checks), "count_unit": "query_face_observations",
                              "bound_count": sum(any(v["binding_status"] == "matched" for v in entries) for entries in pair_checks),
                              "processed_bound_count": sum(r.get("pair_processed") is True for r in checks),
                              "partial_count": sum(r.get("pair_processing_status") == "partial" for r in checks),
                              "failed_bound_count": sum(r.get("pair_processing_status") == "failed" for r in checks),
                              "ambiguous_count": sum(r.get("pair_processing_status") == "ambiguous" for r in checks),
                              "unmatched_count": sum(r.get("pair_processing_status") == "unmatched" for r in checks),
                              "unknown_count": sum(r.get("pair_processing_status") == "unknown" for r in checks),
                              "skipped_count": sum(r.get("pair_processing_status") == "skipped" for r in checks),
                              "candidate_failed_count": sum(v.get("candidate_status") in {"error", "failed", "timeout"} for v in pair_bindings),
                              "candidate_attempt_count": sum(len(entries) for entries in pair_checks),
                              "candidate_attempt_status_counts": {state: sum(v["processing_status"] == state for entries in pair_checks for v in entries)
                                  for state in ("processed", "failed", "ambiguous", "unmatched", "unknown", "skipped")},
                              "limitation": "已处理仅表示本查询脸有明确对应且完成测量；不等于身份相同、结论可靠或伪造类型已确定。"}
    audit["bindings"] = {"identity": identity_binding, "identity_observations": identity_faces,
                          "pair_candidates": pair_bindings,
                          "method": "same_query_scope_and_exact_one_to_one_geometry",
                          "limitation": "对应关系仅限同一查询画面，不证明身份、制作人或伪造类型。"}
    return audit
