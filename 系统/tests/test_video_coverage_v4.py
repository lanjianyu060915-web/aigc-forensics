"""Independent, constructed fixtures only: no real video, model, library or scores.

Fake capture frames are numeric markers; fake JPEG output is a marker string.
All fixture values below were constructed for contract checks, not derived from
any media, historical report, holdout or protected team's data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.video_frames_v2 import run_video_frames_v2 as runner
from video_pipeline.aggregate import aggregate_deepfake_frames, build_video_summary
from video_pipeline.selection import (
    candidate_indices, select_key_frames, select_segmented_key_frames,
    summarize_frame_coverage, LEGACY_SELECTION_STRATEGY,
)


def observation(index, timestamp=None, sharpness=100):
    return {"frame_index": index, "frame_id": f"frame_{index:09d}",
            "timestamp_seconds": index / 10 if timestamp is None else timestamp,
            "timestamp_source": "frame_index_over_nominal_fps", "timestamp_is_pts": False,
            "sharpness": sharpness, "brightness": 128, "face_count": 1}


class FakeImage:
    shape = (12, 16, 3)
    def __init__(self, marker):
        self.marker = marker


class FakeEncoded:
    def __init__(self, marker):
        self.marker = marker
    def tofile(self, target):
        Path(target).write_bytes(f"independently-constructed-jpeg-marker-{self.marker}".encode())


class FakeCapture:
    def __init__(self, *, count=100, fps=10, failed=(), seek_fail_once=()):
        self.count, self.fps = count, fps
        self.failed = set(failed)
        self.seek_fail_once = set(seek_fail_once)
        self.position = 0
        self.last_seek = None
        self.released = False
    def isOpened(self):
        return True
    def get(self, prop):
        return {1: self.fps, 2: self.count, 3: 16, 4: 12,
                5: self.position, 6: self.position * 100}.get(prop, 0)
    def set(self, prop, value):
        self.position = int(value)
        self.last_seek = int(value)
        return True
    def grab(self):
        self.position += 1
        return self.position <= self.count
    def read(self):
        index = self.position
        self.position += 1
        if index in self.failed:
            return False, None
        if index == self.last_seek and index in self.seek_fail_once:
            self.seek_fail_once.remove(index)
            return False, None
        return True, FakeImage(index)
    def release(self):
        self.released = True


@pytest.fixture
def fake_cv(monkeypatch):
    module = SimpleNamespace(
        CAP_PROP_FPS=1, CAP_PROP_FRAME_COUNT=2, CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_POS_FRAMES=5, CAP_PROP_POS_MSEC=6,
        IMWRITE_JPEG_QUALITY=7,
        imencode=lambda extension, image, options: (True, FakeEncoded(image.marker)),
    )
    monkeypatch.setitem(sys.modules, "cv2", module)
    monkeypatch.setattr(runner, "frame_metrics", lambda image: {
        "sharpness": 100.0 + image.marker, "brightness": 128.0, "contrast": 20.0})
    def forbidden(*args, **kwargs):
        raise AssertionError("A model was requested in a synthetic test")
    monkeypatch.setattr(runner, "load_face_detector", forbidden)
    return module


def args_for(tmp_path, extra=()):
    source = tmp_path / "independent-fake-video.mp4"
    source.write_bytes(b"constructed fixture, not encoded video")
    argv = ["--video", str(source), "--output-dir", str(tmp_path / "frames"),
            "--selection", str(tmp_path / "selection.json"), "--candidate-frames", "4",
            "--max-key-frames", "2", "--face-detector", "none", *extra]
    return runner._parser().parse_args(argv), argv


def test_coverage_describes_discrete_observations_and_unsampled_edges():
    items = [observation(20), observation(60), observation(60)]
    result = summarize_frame_coverage(items, frame_count=100, fps=10, duration_seconds=10)
    assert result["entries"] == 3
    assert result["unique_requested_frame_count"] == 2
    assert result["continuous_time_coverage"] is False
    assert result["edge_unsampled_frame_ranges_half_open"] == {"leading": [0, 20], "trailing": [61, 100]}
    assert result["max_frame_index_gap"] == 40
    assert result["max_timestamp_gap_seconds"] == 4
    empty = summarize_frame_coverage([], frame_count=100)
    assert empty["edge_unsampled_frame_ranges_half_open"] == {"leading": [0, 100], "trailing": None}


def test_segmented_candidate_is_deterministic_bounded_and_not_the_default():
    items = [observation(0, 0.0, 99), observation(2, 0.2, 98),
             observation(40, 4.0, 20), observation(80, 8.0, 10)]
    candidate = select_segmented_key_frames(items, max_frames=3, duration_seconds=9)
    assert [item["frame_index"] for item in candidate["key_frames"]] == [0, 40, 80]
    assert candidate == select_segmented_key_frames(items, max_frames=3, duration_seconds=9)
    assert candidate["policy"]["experimental"] is True
    assert candidate["policy"]["independent_effect_validation"] == "not_performed"
    assert runner._parser().get_default("selection_strategy") == LEGACY_SELECTION_STRATEGY
    baseline = select_key_frames(items, max_frames=3, duration_seconds=9)
    assert "experimental" not in baseline["policy"]  # legacy pure function unchanged


def test_segmented_candidate_records_empty_bins_and_rejects_unknown_time():
    items = [observation(index, 0.0) for index in (0, 1, 2)]
    result = select_segmented_key_frames(items, max_frames=3, duration_seconds=9)
    assert result["policy"]["empty_temporal_bins"] == [1, 2]
    assert len(result["key_frames"]) == 3
    items[0]["timestamp_seconds"] = None
    with pytest.raises(ValueError):
        select_segmented_key_frames(items, max_frames=3)


def test_experimental_strategy_requires_explicit_opt_in_before_hashing(tmp_path, fake_cv, monkeypatch):
    args, _ = args_for(tmp_path, ["--selection-strategy", "segmented_time_v1"])
    monkeypatch.setattr(runner, "sha256_file", lambda path: pytest.fail("must refuse before reading input"))
    with pytest.raises(runner.FrameSelectionError, match="显式"):
        runner.run(args)


def test_seek_fallback_is_recorded_without_changing_legacy_frame_choices(fake_cv):
    cap = FakeCapture(count=25000, seek_fail_once=(100,))
    records = {}
    frames, strategy = runner.sample_frames(cap, [100, 200], frame_count=25000, decode_records=records)
    assert strategy == "seek_by_index+fallback_sequential"
    assert frames[100].marker == 100 and frames[200].marker == 200
    assert [attempt["decoded"] for attempt in records[100]] == [False, True]
    assert [attempt["method"] for attempt in records[100]] == ["seek_by_index", "fallback_sequential"]
    assert records[100][-1]["backend_observations_verified"] is False
    assert records[100][-1]["timestamp_is_pts"] is False


def test_runner_retains_failed_candidates_and_exact_legacy_selection(tmp_path, fake_cv, monkeypatch):
    args, _ = args_for(tmp_path)
    planned = candidate_indices(100, 4)
    cap = FakeCapture(failed=(planned[1],))
    monkeypatch.setattr(runner, "_open_capture", lambda path: cap)
    payload = runner.run(args)
    assert cap.released
    assert payload["sampling"]["candidate_frames_requested"] == 4
    assert payload["sampling"]["candidate_frames_planned"] == 4
    assert payload["sampling"]["candidate_frames_decoded"] == 3
    assert payload["sampling"]["candidate_frames_failed"] == 1
    assert len(payload["candidates"]) == 4
    failed = payload["candidates"][1]
    assert failed["selected"] is False and failed["exclusion_reason"] == "decode_failed"
    assert failed["timestamp_is_pts"] is False
    usable = [item for item in payload["candidates"] if item["decoded"]]
    legacy = select_key_frames(usable, max_frames=2, duration_seconds=10,
                               min_sharpness_ratio=0.35, prefer_single_face=True)
    assert [(x["frame_index"], x["quality_score"]) for x in payload["key_frames"]] == [
        (x["frame_index"], x["quality_score"]) for x in legacy["key_frames"]]
    assert payload["input_parameters"]["jpeg_quality"] == 95
    assert payload["selected_coverage"]["entries"] == 2


def test_all_decode_failures_survive_serialized_error(tmp_path, fake_cv, monkeypatch):
    args, argv = args_for(tmp_path)
    cap = FakeCapture(failed=candidate_indices(100, 4))
    monkeypatch.setattr(runner, "_open_capture", lambda path: cap)
    assert runner.main(argv) == 2
    payload = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["sampling"]["candidate_frames_failed"] == 4
    assert len(payload["candidates"]) == 4
    assert payload["key_frames"] == []


def test_zero_fps_has_no_invented_pts(tmp_path, fake_cv, monkeypatch):
    args, _ = args_for(tmp_path)
    monkeypatch.setattr(runner, "_open_capture", lambda path: FakeCapture(fps=0))
    result = runner.run(args)
    assert all(item["timestamp_seconds"] is None for item in result["candidates"])
    assert all(item["timestamp_source"] == "unavailable" for item in result["candidates"])
    assert result["selected_coverage"]["timestamp_seconds"] == []


def test_non_ascii_staging_is_cleaned_on_probe_error(tmp_path, fake_cv, monkeypatch):
    args, _ = args_for(tmp_path)
    staging = tmp_path / "synthetic-staging"
    staging.mkdir()
    cap = FakeCapture(count=0)
    monkeypatch.setattr(runner, "_ascii_media_path", lambda path: (path, str(staging)))
    monkeypatch.setattr(runner, "_open_capture", lambda path: cap)
    with pytest.raises(runner.FrameSelectionError):
        runner.run(args)
    assert cap.released and not staging.exists()


def test_partial_processing_keeps_band_and_propagates_frame_identity():
    good = {"frame": observation(10), "payload": {"status": "success", "score": 0.2,
            "details": {"band": "low"}, "runtime_seconds": 0.0}}
    bad = {"frame": observation(50), "payload": None, "error": "synthetic timeout"}
    result = aggregate_deepfake_frames([good, bad], query_image="synthetic.jpg")
    assert result["score"] == 0.2 and result["details"]["band"] == "low"
    assert result["status"] == "success"  # compatibility; completeness is separate
    coverage = result["details"]["processing_coverage"]
    assert coverage["state"] == "partial"
    assert coverage["attempted_frames"] == 2 and coverage["valid_frames"] == 1
    assert result["details"]["per_frame"][0]["frame_id"] == "frame_000000010"
    assert result["details"]["frame_failures"][0]["timestamp_is_pts"] is False
    summary = build_video_summary(selection={"key_frames": [observation(10)],
                                  "sampling": {"candidate_frames_planned": 4}},
                                  module_payloads={"deepfake": result}, input_file="synthetic.mp4")
    assert summary["key_frames"][0]["frame_id"] == "frame_000000010"
    assert summary["sampling"]["candidate_frames_planned"] == 4
    assert summary["modules"]["deepfake"]["processing_coverage"]["state"] == "partial"
