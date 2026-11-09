"""Conservative correspondence of observations of the SAME query image face.

No IoU tolerance, identity inference, reference-to-query geometric matching, or
cross-image clustering. Callers must provide original-image coordinates and a
shared, task-local source_id + frame_id. A numeric face_id is never sufficient.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in value):
        return None
    left, top, right, bottom = (float(v) for v in value)
    return (left, top, right, bottom) if right > left and bottom > top else None


def _scope(row: Mapping[str, Any]) -> tuple[str, str] | None:
    source, frame = row.get("source_id"), row.get("frame_id")
    if (row.get("coordinate_space") != "original_image" or not isinstance(source, str) or not source
            or not isinstance(frame, str) or not frame):
        return None
    return source, frame


def _detection(row: Mapping[str, Any]) -> tuple[str, str] | None:
    namespace, identifier = row.get("detection_scope_id"), row.get("detection_id")
    return (namespace, identifier) if all(isinstance(v, str) and v for v in (namespace, identifier)) else None


def bind_faces(query_faces: Sequence[Mapping[str, Any]], evidence_faces: Sequence[Mapping[str, Any]], *,
               query_module: str = "face_aigc", evidence_module: str = "identity") -> dict[str, Any]:
    """Return only uniquely determined one-to-one joins, preserving all unknowns.

    Both lists describe faces in the *query* image. Pair modules must expose their
    query-side observation here; reference-side faces require their own explicit
    pair correspondence and cannot be joined merely because coordinates agree.
    """
    candidates: dict[int, list[tuple[int, str]]] = {}
    reasons = {}
    for index, left in enumerate(query_faces):
        left_scope, left_box = _scope(left), _bbox(left.get("bbox"))
        if left_scope is None or left_box is None:
            reasons[index] = "missing_original_coordinates_or_scope"
            candidates[index] = []
            continue
        possible = []
        conflicting_id = False
        for other_index, right in enumerate(evidence_faces):
            if left_scope != _scope(right):
                continue
            right_box = _bbox(right.get("bbox"))
            if right_box is None:
                continue
            left_id, right_id = _detection(left), _detection(right)
            if left_id is not None and left_id == right_id:
                if left_box != right_box:
                    conflicting_id = True
                else:
                    possible.append((other_index, "shared_detection_id"))
            elif left_box == right_box:
                # A shared detector explicitly assigned two different observations.
                if left_id and right_id and left_id[0] == right_id[0]:
                    conflicting_id = True
                else:
                    possible.append((other_index, "exact_original_geometry"))
        candidates[index] = possible
        if conflicting_id:
            reasons[index] = "conflicting_detection_provenance"
    reverse: dict[int, list[int]] = {}
    for index, matches in candidates.items():
        for other_index, _ in matches:
            reverse.setdefault(other_index, []).append(index)
    rows = []
    for index, left in enumerate(query_faces):
        matches = candidates[index]
        record = {"query_record_index": index, "query_face_uid": left.get("face_uid"),
                  "source_id": left.get("source_id"), "frame_id": left.get("frame_id"),
                  "binding_status": "unmatched", "binding_method": None,
                  "evidence_record_index": None, "evidence_face_uid": None,
                  "candidate_count": len(matches)}
        if index in reasons:
            record["reason"] = reasons[index]
            if reasons[index] == "conflicting_detection_provenance":
                record["binding_status"] = "ambiguous"
        elif len(matches) > 1 or (matches and len(reverse[matches[0][0]]) > 1):
            record.update(binding_status="ambiguous", reason="not_one_to_one")
        elif len(matches) == 1:
            other_index, method = matches[0]
            record.update(binding_status="matched", binding_method=method, reason="unique_same_query_observation",
                          evidence_record_index=other_index, evidence_face_uid=evidence_faces[other_index].get("face_uid"))
        else:
            record["reason"] = "no_exact_same_query_correspondence"
        rows.append(record)
    return {"schema_version": "1.0", "scope": "same_query_face_observation", "query_module": query_module,
            "evidence_module": evidence_module, "rows": rows,
            "matched_count": sum(row["binding_status"] == "matched" for row in rows),
            "ambiguous_count": sum(row["binding_status"] == "ambiguous" for row in rows),
            "unmatched_count": sum(row["binding_status"] == "unmatched" for row in rows)}
