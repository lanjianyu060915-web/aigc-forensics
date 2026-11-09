"""E02 fixtures are independent arrays, literal coordinates and fake models only.

No protected media, scores, caches, model packages or historical reports are read.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "face_crops"))
sys.path.insert(0, str(ROOT / "modules" / "general_aigc"))
from audit import frame_identifier, new_audit
from cropper import FaceCropper
from multiface import write_frame_crops
import run_face_crops
from general_aigc_core import enrich_face_branch, face_branch_payload, merge_face_check_results


def fake_cropper(boxes):
    cropper = object.__new__(FaceCropper)
    calls = []
    def get(frame):
        calls.append(frame.shape)
        return [type("Face", (), {"bbox": np.array(box, dtype=float)})() for box in boxes]
    cropper.app = type("Detector", (), {"get": staticmethod(get)})()
    return cropper, calls


def frame_audit(source="independent.png", count=1):
    audit = new_audit()
    frame_id = frame_identifier(source)
    audit["frame_records"].append({"frame_id": frame_id, "source": source, "status": "success",
                                    "detected_total": count, "count_basis": "synthetic_detector"})
    return audit


def test_inspection_records_all_detections_with_stable_ties_and_one_detection():
    boxes = [[20, 20, 120, 120], [0, 0, 100, 100], *[[0, 0, side, side] for side in range(90, 0, -10)]]
    cropper, calls = fake_cropper(boxes)
    result = cropper.inspect_faces(np.zeros((140, 140, 3), dtype=np.uint8))
    assert len(calls) == 1
    assert result["detected_total"] == 11
    assert result["selected_count"] == 8
    assert result["truncated_count"] == 3
    assert result["boxes"][:2] == [tuple(boxes[0]), tuple(boxes[1])]
    assert [row["detector_index"] for row in result["detections"]] == list(range(11))
    assert all(row["exclusion_reason"] == "over_limit" for row in result["detections"][8:])


def test_image_route_retains_over_limit_without_scoring_or_changing_old_crops(tmp_path, monkeypatch):
    frame = np.full((160, 160, 3), 93, np.uint8)
    cropper, calls = fake_cropper([[0, 0, side, side] for side in range(100, 0, -10)])
    monkeypatch.setattr(run_face_crops, "_read_image", lambda path: frame)
    monkeypatch.setattr("cropper.FaceCropper", lambda root: cropper)
    audit = new_audit()
    legacy, rows = run_face_crops.crops_from_image(Path("independent.png"), tmp_path, Path("unused"),
                                                include_all=True, audit=audit)
    assert len(calls) == 1 and len(legacy) == 1 and len(rows) == 8
    assert audit["coverage"]["detected_total"] == 10
    assert audit["coverage"]["truncated_count"] == 2
    assert audit["coverage"]["cropped_count"] == 8
    assert [r["face_id"] for r in audit["face_checks"]] == list(range(1, 11))
    assert len({r["face_uid"] for r in audit["face_checks"]}) == 10
    assert not any("file" in r for r in audit["face_checks"][8:])


def test_no_face_and_detection_failure_have_distinct_audits(tmp_path, monkeypatch):
    monkeypatch.setattr(run_face_crops, "_read_image", lambda path: np.zeros((5, 5, 3), np.uint8))
    cropper, _ = fake_cropper([])
    monkeypatch.setattr("cropper.FaceCropper", lambda root: cropper)
    empty = new_audit()
    assert run_face_crops.crops_from_image(Path("blank.png"), tmp_path, Path("unused"), audit=empty) == []
    assert empty["status"] == "no_face" and empty["coverage"]["detected_total"] == 0
    def fail(frame):
        raise RuntimeError("synthetic detector failure")
    cropper.app.get = fail
    failed = new_audit()
    with pytest.raises(RuntimeError):
        run_face_crops.crops_from_image(Path("blank.png"), tmp_path, Path("unused"), audit=failed)
    assert failed["frame_records"][0]["status"] == "detection_failed"
    assert failed["coverage"]["detected_total"] is None


def test_tiny_faces_kept_transform_and_actual_encoded_input_are_recorded(tmp_path):
    frame = np.full((120, 120, 3), 73, np.uint8)
    audit = frame_audit(count=2)
    legacy, rows = write_frame_crops(frame, [(10, 10, 60, 60), (80, 80, 100, 100)], source="independent.png",
                                    prefix="image", out_dir=tmp_path, audit=audit)
    assert len(legacy) == 1 and len(rows) == 2
    assert rows[0]["legacy_selected"] and rows[0]["too_small"]
    assert not rows[1]["legacy_selected"] and rows[1]["crop_status"] == "success"
    assert rows[1]["short_edge_px"] == 20
    for row in rows:
        assert row["model_input"]["sha256"] == hashlib.sha256((tmp_path / row["file"]).read_bytes()).hexdigest()
        bounds = row["transform"]["crop_box_original"]
        assert row["transform"]["crop_to_original"][0][2] == bounds[0]
        assert row["transform"]["crop_to_original"][1][2] == bounds[1]
        assert row["quality_status"] == "unassessed"
        assert row["identity_processed"] is None


def test_invalid_crop_and_encoder_false_do_not_silently_disappear(tmp_path, monkeypatch):
    frame = np.zeros((200, 200, 3), np.uint8)
    audit = frame_audit(count=3)
    monkeypatch.setattr(cv2, "imencode", lambda *args: (False, None))
    legacy, rows = write_frame_crops(frame, [(0, 0, 80, 80), (1, 1, 0, 0), (500, 500, 550, 550)],
                                    source="independent.png", prefix="image", out_dir=tmp_path, audit=audit)
    assert legacy == [] and rows == []
    assert [r["crop_status"] for r in audit["face_checks"]] == ["encode_failed", "invalid_bbox", "crop_failed"]
    assert audit["coverage"]["failed_count"] == 3
    assert all(row["score"] is None for row in audit["face_checks"])


def test_write_failure_is_recorded_and_still_raises(tmp_path, monkeypatch):
    class Encoded:
        def tofile(self, path):
            raise OSError("synthetic disk failure")
    monkeypatch.setattr(cv2, "imencode", lambda *args: (True, Encoded()))
    audit = frame_audit()
    with pytest.raises(OSError):
        write_frame_crops(np.zeros((90, 90, 3), np.uint8), [(10, 10, 70, 70)],
                          source="independent.png", prefix="image", out_dir=tmp_path, audit=audit)
    assert audit["face_checks"][0]["crop_status"] == "write_failed"
    assert audit["coverage"]["failed_count"] == 1


def test_video_records_frame_limit_and_never_decodes_extra_frames(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"key_frames": [
        {"path": f"synthetic-{i}.png", "frame_index": i * 10, "face_count": 1,
         "faces": [{"bbox": [10, 10, 70, 70]}], "timestamp_seconds": i / 3} for i in range(7)]}), "utf-8")
    read = []
    def image(path):
        read.append(path.name)
        return np.zeros((80, 80, 3), np.uint8)
    monkeypatch.setattr(run_face_crops, "_read_image", image)
    audit = new_audit()
    legacy, rows = run_face_crops.crops_from_frames(selection, tmp_path, tmp_path / "out", include_all=True, audit=audit)
    assert len(read) == 5 and len(legacy) == 5 and len(rows) == 5
    assert len(audit["frame_records"]) == 7
    assert audit["coverage"]["frames_not_processed"] == 2
    assert audit["coverage"]["detected_total"] == 7
    assert len({row["face_uid"] for row in audit["face_checks"]}) == 7


def test_video_unknown_total_decode_failure_and_malformed_records_remain_distinct(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"key_frames": [
        {"path": "one.png", "largest_face_bbox": [10, 10, 70, 70]},
        {"path": "missing.png", "face_count": 1, "faces": [{"bbox": [10, 10, 70, 70]}]},
        {"path": "invalid.png", "faces": [{}, {"bbox": [10, 10, 70, 70]}]}]}), "utf-8")
    monkeypatch.setattr(run_face_crops, "_read_image", lambda path: None if path.name == "missing.png" else np.zeros((80, 80, 3), np.uint8))
    audit = new_audit()
    run_face_crops.crops_from_frames(selection, tmp_path, tmp_path / "out", audit=audit)
    assert audit["frame_records"][0]["detected_total"] is None
    assert audit["frame_records"][1]["status"] == "decode_failed"
    assert audit["coverage"]["detected_total"] is None
    assert {row["crop_status"] for row in audit["face_checks"]} == {"success", "frame_decode_failed", "invalid_bbox"}


def test_decode_failure_keeps_over_limit_distinct_and_ignored_metadata_cannot_break_legacy(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    frames = [{"path": "missing.png", "face_count": 10,
               "faces": [{"bbox": [0, 0, n, n]} for n in range(100, 0, -10)]}]
    frames.extend({"path": f"empty-{i}.png", "faces": [], "face_count": 0} for i in range(4))
    frames.append({"path": "never-read.png", "faces": [{"bbox": [0, 0, "bad", 20]}]})
    selection.write_text(json.dumps({"key_frames": frames}), "utf-8")
    monkeypatch.setattr(run_face_crops, "_read_image", lambda path: None)
    audit = new_audit()
    assert run_face_crops.crops_from_frames(selection, tmp_path, tmp_path / "out", audit=audit) == []
    assert audit["coverage"]["truncated_count"] == 2
    assert sum(row["crop_status"] == "frame_decode_failed" for row in audit["face_checks"]) == 8
    assert audit["frame_records"][-1]["status"] == "over_frame_limit"


def test_inconsistent_largest_face_metadata_never_claims_zero_detected():
    boxes, records, total, basis = run_face_crops._recorded_boxes(
        {"face_count": 0, "largest_face_bbox": [0, 0, 60, 60]}, 8)
    assert len(boxes) == 1 and len(records) == 1
    assert total is None and basis == "inconsistent_or_unresolved_fallback"


def test_merging_scores_keeps_old_decision_and_records_unused_faces(tmp_path):
    audit = frame_audit(count=2)
    legacy, rows = write_frame_crops(np.zeros((140, 140, 3), np.uint8), [(10, 10, 60, 60), (80, 80, 100, 100)],
                                    source="independent.png", prefix="image", out_dir=tmp_path, audit=audit)
    calibration = {"face_crop": {"low_max": 2.0, "medium_max": 4.0}}
    original = face_branch_payload([1.0], [legacy[0]["file"]], calibration)
    frozen = copy.deepcopy(audit)
    result = enrich_face_branch(original, rows, [1.0, 9.0], calibration, face_audit=audit)
    assert {key: result[key] for key in original} == original
    assert audit == frozen
    assert result["face_audit"]["coverage"]["bound_score_count"] == 2
    assert result["face_audit"]["coverage"]["used_for_band_count"] == 1
    checks = result["face_audit"]["face_checks"]
    assert checks[0]["used_for_band"] and checks[0]["too_small"]
    assert checks[1]["aggregation_exclusion_reason"] == "not_selected_by_frozen_rule"
    failed = merge_face_check_results(audit, [], inference_status="failed", error_type="SyntheticError")
    assert failed["status"] == "inference_failed"
    assert all(row["score"] is None and row["score_binding_reason"] == "model_failed" for row in failed["face_checks"])


def test_unknown_coverage_and_mismatched_score_identity_are_not_invented(tmp_path):
    unknown = merge_face_check_results(None, [{"face_id": 1, "score": 3, "used_for_band": True}])
    assert unknown["coverage"]["detected_total"] is None
    assert unknown["coverage"]["bound_score_count"] == 0
    audit = frame_audit()
    _, rows = write_frame_crops(np.zeros((90, 90, 3), np.uint8), [(10, 10, 70, 70)],
                               source="independent.png", prefix="image", out_dir=tmp_path, audit=audit)
    bad = {**rows[0], "source_id": "different.png", "score": 3.0, "score_status": "success"}
    result = merge_face_check_results(audit, [bad])
    assert result["face_checks"][0]["score_status"] == "unmatched"
    assert result["coverage"]["bound_score_count"] == 0
