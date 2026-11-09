"""Independent geometry/provenance literals only; no images or model access."""
from __future__ import annotations

from copy import deepcopy
import pytest

from reporting.face_bindings import bind_faces


def face(uid, box=(10, 20, 50, 60), source="synthetic-query", frame="frame-0", **extra):
    return {"face_uid": uid, "face_id": 1, "source_id": source, "frame_id": frame,
            "coordinate_space": "original_image", "bbox": list(box), **extra}


def test_exact_geometry_is_one_to_one_despite_order_and_numeric_id():
    left = [face("q1"), face("q2", box=(80, 20, 120, 60))]
    right = [face("i2", box=(80, 20, 120, 60)), face("i1")]
    frozen = deepcopy((left, right))
    result = bind_faces(left, right)
    assert result["matched_count"] == 2
    assert [row["evidence_face_uid"] for row in result["rows"]] == ["i1", "i2"]
    assert (left, right) == frozen


@pytest.mark.parametrize("changes", [
    {"bbox": [11, 20, 51, 60]}, {"bbox": None}, {"bbox": [10, 20, 10, 60]},
    {"bbox": [10, 20, float("nan"), 60]}, {"source_id": "reference-image"},
    {"frame_id": "frame-1"}, {"coordinate_space": "crop_pixels"}, {"source_id": None},
])
def test_same_numeric_id_does_not_bind_missing_different_or_untransformed_geometry(changes):
    result = bind_faces([face("q")], [{**face("i"), **changes}])
    assert result["matched_count"] == 0
    assert result["rows"][0]["evidence_face_uid"] is None


def test_explicit_shared_detection_id_requires_consistent_original_coordinates():
    left = face("q", detection_scope_id="detector-run", detection_id="observation-1")
    right = face("i", detection_scope_id="detector-run", detection_id="observation-1")
    result = bind_faces([left], [right])
    assert result["rows"][0]["binding_method"] == "shared_detection_id"
    right["bbox"] = [30, 40, 70, 80]
    result = bind_faces([left], [right])
    assert result["rows"][0]["binding_status"] == "ambiguous"


def test_multiple_candidates_and_multiple_queries_never_form_a_join():
    assert bind_faces([face("q")], [face("i1"), face("i2")])["ambiguous_count"] == 1
    result = bind_faces([face("q1"), face("q2")], [face("i")])
    assert result["ambiguous_count"] == 2 and result["matched_count"] == 0


def test_same_detector_different_observations_with_same_geometry_are_ambiguous():
    left = face("q", detection_scope_id="same-run", detection_id="one")
    right = face("i", detection_scope_id="same-run", detection_id="two")
    result = bind_faces([left], [right])
    assert result["rows"][0]["reason"] == "conflicting_detection_provenance"
    assert result["matched_count"] == 0


def test_missing_query_coordinates_and_empty_evidence_stay_unmatched():
    result = bind_faces([face("q", bbox=None)], [face("i")])
    assert result["unmatched_count"] == 1
    assert bind_faces([face("q")], [])["unmatched_count"] == 1
