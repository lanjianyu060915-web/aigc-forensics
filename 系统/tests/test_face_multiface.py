"""逐脸展示与定档策略分别验证，覆盖旧字段兼容和视频帧权重。"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "face_crops"))
sys.path.insert(0, str(ROOT / "modules" / "general_aigc"))
from cropper import FaceCropper, square_crop
from general_aigc_core import enrich_face_branch, face_branch_payload

CAL = {"face_crop": {"low_max": 4.0, "medium_max": 6.0}}

def face(source, index, side=100, legacy=False):
    return {"file": f"{source}_{index}.jpg", "source": source, "face_id": index,
            "bbox": [0, 0, side, side], "short_edge_px": side,
            "legacy_selected": legacy, "timestamp_seconds": None}

def test_default_preserves_every_legacy_field_with_extra_face_scores():
    original = face_branch_payload([1.0, 4.0], ["a_1.jpg", "b_1.jpg"], CAL)
    frozen = copy.deepcopy(original)
    enriched = enrich_face_branch(original, [face("a", 1, legacy=True), face("a", 2), face("b", 1, legacy=True)],
                                  [1.0, 9.0, 4.0], CAL)
    assert original == frozen
    assert {key: enriched[key] for key in frozen} == frozen
    assert enriched["highest_score_face"]["face_id"] == 2
    assert [row["used_for_band"] for row in enriched["per_face"]] == [True, False, True]

def test_video_uses_max_per_frame_then_equal_weight_mean():
    original = face_branch_payload([1.0, 4.0], ["a_1.jpg", "b_1.jpg"], CAL)
    result = enrich_face_branch(original, [face("a", 1, legacy=True), face("a", 2), face("b", 1, legacy=True)],
                               [1.0, 9.0, 4.0], CAL, band_aggregation="max_all")
    assert result["logit"] == 6.5
    assert result["band"] == "high"
    assert [row["score"] for row in result["frame_scores"]] == [9.0, 4.0]
    assert result["crops"] == original["crops"]

def test_small_face_is_displayed_but_cannot_raise_max_all():
    original = face_branch_payload([1.0], ["a_1.jpg"], CAL)
    result = enrich_face_branch(original, [face("a", 1, legacy=True), face("a", 2, side=30)],
                               [1.0, 99.0], CAL, band_aggregation="max_all")
    assert result["logit"] == 1.0
    assert result["highest_score_face"]["face_id"] == 2
    assert result["per_face"][1]["too_small"]
    assert not result["per_face"][1]["eligible_for_band"]

def test_cf_largest_keeps_the_explicit_40_to_63_pixel_compatibility_exception():
    original = face_branch_payload([9.0], ["a_1.jpg"], CAL)
    result = enrich_face_branch(original, [face("a", 1, side=50, legacy=True)], [9.0], CAL)
    assert result["band"] == "high"
    assert result["per_face"][0]["too_small"]
    assert result["per_face"][0]["used_for_band"]
    strict = enrich_face_branch(original, [face("a", 1, side=50, legacy=True)], [9.0], CAL,
                               band_aggregation="max_all")
    assert strict["band"] is None
    assert strict["logit"] is None
    assert not strict["per_face"][0]["used_for_band"]

def test_empty_faces_preserve_the_legacy_no_face_result():
    original = face_branch_payload([], [], CAL)
    result = enrich_face_branch(original, [], [], CAL)
    assert {key: result[key] for key in original} == original
    assert result["per_face"] == []
    assert result["highest_score_face"] is None

def test_nonfinite_or_incomplete_face_scores_are_rejected():
    original = face_branch_payload([1.0], ["a_1.jpg"], CAL)
    with pytest.raises(ValueError):
        enrich_face_branch(original, [face("a", 1)], [], CAL)
    with pytest.raises(ValueError):
        enrich_face_branch(original, [face("a", 1)], [float("nan")], CAL)

def test_stable_area_sort_retains_old_largest_tie_and_caps_eight():
    class Detection:
        def __init__(self, box):
            self.bbox = np.array(box, dtype=float)
    boxes = [[4, 5, 104, 105], [20, 30, 120, 130], *[[0, 0, n, n] for n in range(90, 0, -10)]]
    cropper = object.__new__(FaceCropper)
    cropper.app = type("Fake", (), {"get": lambda self, frame: [Detection(box) for box in boxes]})()
    selected = cropper.face_boxes(np.zeros((150, 150, 3)), max_faces=8)
    assert selected[:2] == [tuple(boxes[0]), tuple(boxes[1])]
    assert len(selected) == 8
    assert cropper.largest_face_box(np.zeros((150, 150, 3))) == tuple(boxes[0])

def test_tiny_crop_is_opt_in_and_does_not_change_old_crop_default():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert square_crop(frame, (0, 0, 20, 20)) is None
    assert square_crop(frame, (0, 0, 20, 20), min_pixels=1) is not None


def test_new_crops_preserve_the_legacy_jpeg_and_records(tmp_path):
    import cv2
    from multiface import write_frame_crops
    frame = np.arange(400 * 600 * 3, dtype=np.uint8).reshape(400, 600, 3)
    boxes = [(100, 50, 230, 200), (400, 50, 450, 100), (50, 220, 70, 240)]
    legacy, rows = write_frame_crops(frame, boxes, source="frame_001.jpg", prefix="frame_001",
                                    out_dir=tmp_path, is_video=True, timestamp_seconds=2.5)
    old = square_crop(frame, boxes[0])
    ok, encoded = cv2.imencode(".jpg", old, [cv2.IMWRITE_JPEG_QUALITY, 98])
    assert ok
    assert (tmp_path / "frame_001_face.jpg").read_bytes() == encoded.tobytes()
    assert legacy == [{"file": "frame_001_face.jpg", "source": "frame_001.jpg",
                       "bbox": list(boxes[0]), "timestamp_seconds": 2.5}]
    assert len(rows) == 3
    assert [row["too_small"] for row in rows] == [False, True, True]
    assert all((tmp_path / row["file"]).is_file() for row in rows)


def test_video_all_boxes_are_used_and_old_maximum_is_retained(tmp_path):
    import cv2
    import json
    from run_face_crops import crops_from_frames
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    encoded.tofile(str(tmp_path / "frame_001.jpg"))
    selection = {"key_frames": [{"path": "frame_001.jpg", "largest_face_bbox": [20, 20, 120, 140],
                                 "faces": [{"bbox": [20, 20, 120, 140]}, {"bbox": [200, 50, 280, 130]}],
                                 "timestamp_seconds": 3.0}]}
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), "utf-8")
    legacy, rows = crops_from_frames(path, tmp_path, tmp_path / "crops", include_all=True)
    assert len(legacy) == 1
    assert len(rows) == 2
    assert legacy[0]["file"] == "frame_001_face.jpg"
    assert rows[1]["timestamp_seconds"] == 3.0
