"""人脸裁剪和人脸分支的结果结构。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT / "modules" / "face_crops"))
sys.path.insert(0, str(SYSTEM_ROOT / "modules" / "general_aigc"))

from cropper import square_crop  # noqa: E402
from general_aigc_core import face_branch_payload  # noqa: E402

RUNNER = SYSTEM_ROOT / "modules" / "face_crops" / "run_face_crops.py"


def _write_frame(path: Path) -> None:
    ok, encoded = cv2.imencode(".jpg", np.full((400, 600, 3), 128, dtype=np.uint8))
    assert ok
    encoded.tofile(str(path))


def _run_frames(selection: dict, tmp_path: Path) -> dict:
    frames_dir = tmp_path / "关键帧"
    frames_dir.mkdir(exist_ok=True)
    _write_frame(frames_dir / "frame_001.jpg")
    selection_path = tmp_path / "frame_selection.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False), encoding="utf-8")
    out_dir = tmp_path / "输出"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(RUNNER),
            "--out-dir",
            str(out_dir),
            "--frame-selection",
            str(selection_path),
            "--key-frames-dir",
            str(frames_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads((out_dir / "crops.json").read_text(encoding="utf-8"))


def test_square_crop_is_square_with_margin_and_rejects_tiny_faces() -> None:
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    crop = square_crop(frame, (200, 100, 300, 200))
    assert crop is not None
    assert crop.shape[0] == crop.shape[1] == 130
    assert square_crop(frame, (200, 100, 230, 200)) is None


def test_square_crop_shrinks_at_the_frame_edge_instead_of_padding() -> None:
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    crop = square_crop(frame, (0, 0, 100, 100))
    assert crop is not None
    assert crop.shape[:2] == (115, 115)


def test_video_crops_use_recorded_boxes_without_detection(tmp_path: Path) -> None:
    payload = _run_frames(
        {
            "key_frames": [
                {
                    "path": "frame_001.jpg",
                    "largest_face_bbox": [100.4, 50.2, 220.9, 180.7],
                    "timestamp_seconds": 1.5,
                }
            ]
        },
        tmp_path,
    )
    assert payload["status"] == "success"
    assert [item["file"] for item in payload["crops"]] == ["frame_001_face.jpg"]
    assert payload["crops"][0]["timestamp_seconds"] == 1.5
    assert (tmp_path / "输出" / "frame_001_face.jpg").is_file()


@pytest.mark.parametrize(
    "frame",
    [
        {"path": "frame_001.jpg", "largest_face_bbox": None},
        {"path": "frame_001.jpg", "largest_face_bbox": [10, 10, 30, 30]},
        {"path": "不存在.jpg", "largest_face_bbox": [100, 50, 220, 180]},
    ],
)
def test_frames_without_a_usable_face_give_no_crops(frame: dict, tmp_path: Path) -> None:
    payload = _run_frames({"key_frames": [frame]}, tmp_path)
    assert payload == {**payload, "status": "success", "crops": []}


def test_face_branch_payload_averages_crops_and_applies_bands() -> None:
    calibration = {"face_crop": {"low_max": -5.0, "medium_max": -4.0, "calibration_set": {"n_real": 3}}}
    payload = face_branch_payload([-6.0, -3.0], ["a.jpg", "b.jpg"], calibration)
    assert payload["logit"] == -4.5
    assert payload["band"] == "medium"
    assert payload["band_thresholds"]["calibration_set"] == {"n_real": 3}
    assert payload["crops"] == [{"file": "a.jpg", "logit": -6.0}, {"file": "b.jpg", "logit": -3.0}]


def test_face_branch_payload_without_faces_or_thresholds() -> None:
    assert face_branch_payload([], [], {"face_crop": {"low_max": 0, "medium_max": 1}})["status"] == "no_face"
    uncalibrated = face_branch_payload([-1.0], ["a.jpg"], {})
    assert uncalibrated["status"] == "success"
    assert uncalibrated["band"] is None
    assert uncalibrated["band_thresholds"] is None
