"""Integration contracts using new synthetic coordinates, bytes and services."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

import orchestrator
from modules.face_crops.audit import frame_identifier
from reporting.face_evidence_v4 import build_face_coverage_evidence
from reporting.evidence_package import build_evidence_package
from reporting.coverage_summary_v4 import append_coverage_notes, face_coverage_note


def audit_faces():
    return {"status": "recorded", "coverage": {"count_status": "recorded", "detected_total": 2,
            "cropped_count": 2, "scored_count": 1, "used_for_band_count": 1, "truncated_count": 0, "failed_count": 0},
            "face_checks": [{"face_uid": f"sample/{i}", "source_id": "query.jpg", "frame_id": "frame-a",
                             "bbox": box, "coordinate_space": "original_image", "score": score}
                            for i, (box, score) in enumerate([([2, 3, 17, 18], 0.314), ([30, 8, 46, 25], None)])]}


def test_reordered_identity_joins_by_geometry_and_scope_not_index():
    audit = audit_faces()
    before = deepcopy(audit)
    identity = {"query_face_scope": {"source_id": "query.jpg", "frame_id": "frame-a", "coordinate_space": "original_image"},
                "faces": [{"face_index": 0, "embedding_face_bbox_original": [30, 8, 46, 25], "status": "error"},
                          {"face_index": 1, "embedding_face_bbox_original": [2, 3, 17, 18], "status": "success"}]}
    enriched = build_face_coverage_evidence(audit, identity, {})
    assert enriched["face_checks"][0]["identity_processed"] is True
    assert enriched["face_checks"][1]["identity_processed"] is False
    assert enriched["identity_coverage"]["bound_count"] == 2
    assert audit == before
    assert [r["score"] for r in enriched["face_checks"]] == [0.314, None]


@pytest.mark.parametrize("scope", [{}, {"source_id": "query.jpg", "frame_id": "another-frame", "coordinate_space": "original_image"}])
def test_unknown_or_other_frame_cannot_be_assigned_by_face_number(scope):
    result = build_face_coverage_evidence(audit_faces(), {"query_face_scope": scope,
            "faces": [{"face_index": 0, "embedding_face_bbox_original": [2, 3, 17, 18], "status": "success"}]}, {})
    assert result["identity_coverage"]["bound_count"] == 0
    assert all(row["identity_processed"] is None for row in result["face_checks"])


def test_first_selection_only_cannot_claim_actual_embedding_correspondence():
    identity = {"query_face_scope": {"source_id": "query.jpg", "frame_id": "frame-a", "coordinate_space": "original_image"},
                "faces": [{"face_index": 0, "query_bbox_original": [2, 3, 17, 18], "status": "success"}]}
    result = build_face_coverage_evidence(audit_faces(), identity, {})
    assert result["identity_coverage"]["bound_count"] == 0
    assert result["face_checks"][0]["identity_processed"] is None


def test_actual_redetected_face_is_used_and_original_selection_is_not_substituted():
    identity = {"query_face_scope": {"source_id": "query.jpg", "frame_id": "frame-a", "coordinate_space": "original_image"},
                "faces": [{"query_bbox_original": [2, 3, 17, 18], "embedding_face_bbox_original": [30, 8, 46, 25],
                           "selection_trace": {"correspondence_status": "geometry_changed_unverified"}, "status": "success"}],
                "face_search": {"coverage": {"detected_total": 7, "processed_count": 3, "truncated_count": 4},
                                "face_checks": [{"face_index": 6, "processed": False, "exclusion_reason": "over_limit"}]}}
    result = build_face_coverage_evidence(audit_faces(), identity, {})
    assert result["face_checks"][0]["identity_processed"] is None
    assert result["face_checks"][1]["identity_processed"] is True
    assert result["identity_coverage"]["module_reported_coverage"]["detected_total"] == 7
    assert result["coverage"]["detected_total"] == 2
    assert result["bindings"]["identity_observations"][0]["selection_trace"]["correspondence_status"] == "geometry_changed_unverified"


@pytest.mark.parametrize("bad", [None, [], "unrecorded", {"query_face_scope": "invalid"}])
def test_missing_or_malformed_module_scope_keeps_binding_unknown(bad):
    result = build_face_coverage_evidence(audit_faces(), bad, bad)
    assert result["identity_coverage"]["bound_count"] == 0
    assert result["pair_coverage"]["status"] == "unrecorded"


def test_reference_geometry_is_not_used_as_query_geometry():
    pair = {"query_face_scope": {"source_id": "query.jpg", "frame_id": "frame-a", "coordinate_space": "original_image"},
            "candidates": [{"rank": 1, "status": "success", "comparison": {"faces": [
                {"reference_bbox_original": [2, 3, 17, 18], "status": "matched"}]}}]}
    result = build_face_coverage_evidence(audit_faces(), {}, pair)
    assert result["bindings"]["pair_candidates"][0]["binding"]["matched_count"] == 0


def test_audit_is_forwarded_to_context_and_evidence_without_reclassification(tmp_path, monkeypatch):
    # Independently constructed fixtures must not load historical score tables.
    import reporting.evidence_package as evidence_module
    monkeypatch.setattr(evidence_module, "_read", lambda path: {})
    monkeypatch.setattr(evidence_module, "_known_issues", lambda: {})
    directory = tmp_path / "face_crops"
    directory.mkdir()
    audit = audit_faces()
    (directory / "crops.json").write_text(json.dumps({"per_face": [], "face_audit": audit}), encoding="utf-8")
    context = orchestrator._face_context(tmp_path, {"backend": "cf384", "band_aggregation": "largest"})
    assert context["face_audit"] == audit
    raw = {"modules": {"deepfake": {"result": {"overall": {"band": "low"}, "general_aigc": {
        "face_branch": {"status": "success", "logit": 0.314, "band": "low", "face_audit": audit}}}}}}
    package = build_evidence_package(raw)
    assert package["rule_band"] == "none"
    assert package["items"][3]["data"]["face_audit"]["coverage"] == audit["coverage"]


def test_local_coverage_is_appended_after_writer_text():
    package = {"items": [{"item_id": "E04", "data": {"face_audit": audit_faces()}},
                         {"item_id": "E10", "data": {"module_frame_coverage": {
                             "identity": {"requested_count": 3, "success_count": 1, "failure_count": 2}}}}]}
    result = append_coverage_notes({"process": {"detection": "synthetic writer paragraph"}}, package)
    text = result["process"]["detection"]
    assert "已评分 1" in text and "计划处理 3 帧" in text and "失败 2 帧" in text
    assert "未记录" in face_coverage_note({})


def test_identity_scope_binds_actual_query_bytes(tmp_path):
    query = tmp_path / "query.jpg"
    query.write_bytes(b"synthetic-marker-not-real-image")
    payload = {}
    orchestrator._annotate_identity_scope(payload, query)
    assert payload["query_face_scope"]["frame_id"] == frame_identifier("query.jpg")
    assert payload["query_face_scope"]["input_sha256"] == orchestrator._sha256_path(query)


def test_video_module_records_actual_success_and_requested_frames(tmp_path, monkeypatch):
    key_dir = tmp_path / "key_frames"
    key_dir.mkdir()
    frames = [{"path": f"frame_{i}.jpg", "frame_id": f"frame-{i}", "frame_index": i * 11,
               "timestamp_seconds": i * 0.44, "frame_rank": i + 1} for i in range(3)]
    for frame in frames:
        (key_dir / frame["path"]).write_bytes(b"new fake query marker")
    calls = []
    def fake_search(**kwargs):
        calls.append(kwargs["query_path"].name)
        return {"status": "error" if len(calls) == 2 else "success", "candidates": [], "faces": [],
                "runtime_seconds": 0.001, "error_message": "synthetic failure" if len(calls) == 2 else None}, {}
    monkeypatch.setattr(orchestrator, "_run_search_module", fake_search)
    monkeypatch.setattr(orchestrator, "_finish_module", lambda **kwargs: (kwargs["payload"], {}))
    class Progress:
        def emit(self, *args, **kwargs):
            pass
    result, _ = orchestrator._run_video_module(
        name="identity", section={}, config_path=tmp_path / "config.json", task_dir=tmp_path,
        logs_dir=tmp_path / "logs", output_path=tmp_path / "identity_result.json", identity_cache=tmp_path / "cache",
        library_override=None, progress=Progress(), services=None, frames=frames, primary_frame_path=key_dir / frames[0]["path"],
        video_selection={"key_frames": frames, "video": {"frame_count": 100, "fps": 25, "duration_seconds": 4}}, band_labels=None)
    coverage = result["module_frame_coverage"]
    assert coverage["requested_count"] == 3 and coverage["success_count"] == 2 and coverage["failure_count"] == 1
    assert coverage["continuous_detection"] is False
    assert coverage["identity_secondary_faces_scope"] == "first_requested_frame_only"
    assert calls == [f["path"] for f in frames]
