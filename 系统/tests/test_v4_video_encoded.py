"""Real codec checks using only newly constructed pixels and synthetic events.

No external media, model, historical report, protected library or detector is
loaded. FFmpeg/ffprobe must be explicitly configured. Generated fixtures and
audit JSON live under the caller's test_only pytest basetemp.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.video_frames_v2 import run_video_frames_v2 as runner
from video_pipeline.selection import candidate_indices, summarize_frame_coverage

WIDTH, HEIGHT, BITS = 128, 48, 16


@pytest.fixture
def codec_tools(monkeypatch):
    paths = [Path(os.environ.get(name, "")) for name in ("V4_TEST_FFMPEG", "V4_TEST_FFPROBE")]
    if not all(p.is_file() for p in paths):
        pytest.skip("Explicit local FFmpeg/ffprobe paths required; no fake codec substitute")
    versions = [subprocess.run([str(p), "-version"], capture_output=True, check=True, timeout=10).stdout.decode().splitlines()[0]
                for p in paths]
    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic codec tests must not load face models")
    monkeypatch.setattr(runner, "load_face_detector", forbidden)
    return paths[0], paths[1], versions


def coded_frame(index, *, event=False):
    frame = np.full((HEIGHT, WIDTH, 3), 100, dtype=np.uint8)
    for bit in range(BITS):
        frame[12:40, bit*8:(bit+1)*8] = 235 if (index >> bit) & 1 else 20
    frame[:8] = (0, 0, 255) if event else (80, 80, 80)
    return frame


def decoded_number(frame):
    return sum((1 << bit) if frame[24, bit*8+4].mean() > 128 else 0 for bit in range(BITS))


def has_event(frame):
    b, g, r = map(int, frame[3, WIDTH//2])
    return r > 200 and b < 40 and g < 40


def audit(tmp_path, tools, case, files, observations):
    content = {
        "test_only": True, "fixture_origin": "newly_program_constructed_pixel_bars_and_events",
        "case": case, "models_loaded": False, "external_media_read": False,
        "ffmpeg": tools[2][0], "ffprobe": tools[2][1], "opencv": cv2.__version__,
        "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
        "observations": observations,
        "scope": "codec/index/coverage implementation only; not detection accuracy or universal codec support",
    }
    (tmp_path / "synthetic_codec_audit.json").write_text(json.dumps(content, indent=2), encoding="utf-8")


def encode_cfr(tmp_path, tools, *, count, fps=10, events=(), name="synthetic.mkv"):
    output = tmp_path / name
    error_log = tmp_path / (name + ".encode.log")
    command = [str(tools[0]), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s:v", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "pipe:0",
               "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", "-threads", "1", str(output)]
    with error_log.open("wb") as errors:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            for index in range(count):
                proc.stdin.write(coded_frame(index, event=index in events).tobytes())
            proc.stdin.close()
            assert proc.wait(timeout=90) == 0, error_log.read_text(errors="replace")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
    return output


def packet_pts(tools, video):
    result = subprocess.run([str(tools[1]), "-v", "error", "-select_streams", "v:0", "-show_packets",
                             "-show_entries", "packet=pts_time,duration_time", "-of", "json", str(video)],
                            capture_output=True, check=True, timeout=30)
    return [float(x["pts_time"]) for x in json.loads(result.stdout)["packets"]]


def run_selection(tmp_path, video):
    args = runner._parser().parse_args(["--video", str(video), "--output-dir", str(tmp_path / "selected"),
        "--selection", str(tmp_path / "selection.json"), "--face-detector", "none"])
    return runner.run(args)


def sample_exact(video, indices):
    capture = cv2.VideoCapture(str(video))
    try:
        assert capture.isOpened()
        info = runner.probe_video(capture)
        records = {}
        frames, method = runner.sample_frames(capture, indices, frame_count=info["frame_count"], decode_records=records)
        return info, frames, method, records
    finally:
        capture.release()


def test_encoded_cfr_frame_ids_timestamps_and_trimmed_edge_events(tmp_path, codec_tools):
    count, fps = 80, 10
    events = {0, 1, 78, 79}
    video = encode_cfr(tmp_path, codec_tools, count=count, fps=fps, events=events)
    targets = candidate_indices(count, 24)
    info, frames, method, records = sample_exact(video, targets)
    assert info["frame_count"] == count and method == "sequential_grab"
    assert all(decoded_number(frames[i]) == i for i in targets)
    assert all(not has_event(frames[i]) for i in targets)
    assert not set(targets) & events
    payload = run_selection(tmp_path, video)
    assert payload["sampling"]["candidate_frames_decoded"] == 24
    assert payload["sampling"]["candidate_frames_failed"] == 0
    for item in payload["candidates"]:
        assert item["frame_id"] == f"frame_{item['frame_index']:09d}"
        assert item["timestamp_source"] == "frame_index_over_nominal_fps"
        assert item["timestamp_is_pts"] is False
        assert item["timestamp_seconds"] == pytest.approx(item["frame_index"] / fps, abs=1e-6)
    for item in payload["key_frames"]:
        exported = cv2.imread(str(tmp_path / "selected" / item["path"]))
        assert exported is not None and decoded_number(exported) == item["frame_index"]
        assert not has_event(exported)
    coverage = payload["sampling"]["decoded_coverage"]
    assert coverage["continuous_time_coverage"] is False
    assert coverage["edge_unsampled_frame_ranges_half_open"] == {
        "leading": [0, targets[0]], "trailing": [targets[-1]+1, count]}
    audit(tmp_path, codec_tools, "CFR and deliberately missed edge events", [video], {
        "planned_indices": targets, "decoded_numbers": [decoded_number(frames[i]) for i in targets],
        "event_indices": sorted(events), "event_hits": 0, "coverage": coverage, "decode_records": records})


def test_encoded_single_frame_event_between_candidates_is_unobserved(tmp_path, codec_tools):
    targets = candidate_indices(120, 24)
    missed = next(i for i in range(targets[0]+1, targets[1]) if i not in targets)
    video = encode_cfr(tmp_path, codec_tools, count=120, events={missed})
    info, frames, _, _ = sample_exact(video, targets)
    assert all(decoded_number(frames[i]) == i for i in targets)
    assert all(not has_event(frames[i]) for i in targets)
    _, event_frame, _, _ = sample_exact(video, [missed])
    assert has_event(event_frame[missed]) and decoded_number(event_frame[missed]) == missed
    audit(tmp_path, codec_tools, "known short event outside candidate pool", [video], {
        "event_index": missed, "event_exists_by_independent_direct_read": True,
        "candidate_event_hits": 0, "planned_indices": targets, "info": info})


def test_encoded_very_short_clip_decodes_all_available_candidates(tmp_path, codec_tools):
    video = encode_cfr(tmp_path, codec_tools, count=3)
    payload = run_selection(tmp_path, video)
    assert payload["sampling"]["candidate_frames_requested"] == 24
    assert payload["sampling"]["candidate_frames_planned"] == 3
    assert payload["sampling"]["candidate_frames_decoded"] == 3
    assert {x["frame_index"] for x in payload["candidates"]} == {0, 1, 2}
    audit(tmp_path, codec_tools, "shorter than requested candidate count", [video], payload["sampling"])


def test_encoded_long_clip_exercises_real_seek_with_known_frame_numbers(tmp_path, codec_tools):
    count = runner.SEQUENTIAL_SCAN_LIMIT + 3
    video = encode_cfr(tmp_path, codec_tools, count=count, fps=100)
    targets = [0, 101, count//2, count-1]
    info, frames, method, records = sample_exact(video, targets)
    assert info["frame_count"] == count
    assert method.startswith("seek_by_index")
    assert all(frames[i] is not None and decoded_number(frames[i]) == i for i in targets)
    assert all(records[i][0]["method"] == "seek_by_index" for i in targets)
    assert all(records[i][0]["timestamp_is_pts"] is False for i in targets)
    audit(tmp_path, codec_tools, "over sequential scan limit, actual lossless seek", [video], {
        "frame_count": count, "requested_indices": targets, "decoded_numbers": [decoded_number(frames[i]) for i in targets],
        "strategy": method, "decode_records": records})


def test_encoded_vfr_packet_pts_differ_from_reported_nominal_fps_estimate(tmp_path, codec_tools):
    durations = [0.08, 0.24, 0.04, 0.40, 0.12, 0.28, 0.08, 0.36, 0.04, 0.20, 0.16]
    manifest_lines = ["ffconcat version 1.0"]
    for index in range(len(durations)+1):
        name = f"number_{index:02d}.png"
        assert cv2.imwrite(str(tmp_path / name), coded_frame(index, event=index in {0, len(durations)}))
        manifest_lines.append(f"file '{name}'")
        if index < len(durations):
            manifest_lines.append(f"duration {durations[index]:.6f}")
    concat = tmp_path / "constructed.ffconcat"
    concat.write_text("\n".join(manifest_lines)+"\n", encoding="utf-8")
    video = tmp_path / "synthetic_vfr.mkv"
    subprocess.run([str(codec_tools[0]), "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "1",
                    "-i", str(concat), "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", "-threads", "1",
                    "-fps_mode", "vfr", str(video)], capture_output=True, check=True, timeout=30)
    pts = packet_pts(codec_tools, video)
    assert len(pts) == len(durations)+1
    assert len({round(b-a, 6) for a,b in zip(pts, pts[1:])}) > 1
    actual_indices = list(range(len(pts)))
    info, frames, method, records = sample_exact(video, actual_indices)
    assert all(decoded_number(frames[i]) == i for i in actual_indices)
    nominal = [i/info["fps"] for i in actual_indices]
    errors = [abs(a-b) for a,b in zip(pts, nominal)]
    assert max(errors) > 0.03  # The fixture demonstrates why nominal FPS is not PTS.
    payload = run_selection(tmp_path, video)
    assert all(x["timestamp_is_pts"] is False for x in payload["candidates"])
    assert all(x["timestamp_source"] == "frame_index_over_nominal_fps" for x in payload["candidates"])
    assert all(x["timestamp_is_pts"] is False for attempts in records.values() for x in attempts)
    audit(tmp_path, codec_tools, "VFR estimate limitation measured against packet PTS", [video, concat], {
        "packet_pts_source": "ffprobe packet pts_time from synthetic FFV1 stream", "packet_pts": pts,
        "nominal_index_over_fps": nominal, "max_abs_timestamp_error_seconds": max(errors),
        "actual_encoded_frame_count": len(pts), "opencv_reported_info": info,
        "sampling": payload["sampling"], "read_strategy": method,
        "limitation": "Metadata frame count/FPS may be estimated for VFR; no reliable VFR or continuous-time support is claimed"})


def test_encoded_truncation_keeps_failure_or_partial_coverage_visible(tmp_path, codec_tools):
    complete = encode_cfr(tmp_path, codec_tools, count=80)
    broken = tmp_path / "truncated.mkv"
    original = complete.read_bytes()
    broken.write_bytes(original[:len(original)//2])
    argv = ["--video", str(broken), "--output-dir", str(tmp_path / "selected"),
            "--selection", str(tmp_path / "selection.json"), "--face-detector", "none"]
    result = runner.main(argv)
    payload = json.loads((tmp_path / "selection.json").read_text("utf-8"))
    if result == 0:
        assert payload["sampling"]["candidate_frames_failed"] > 0
        assert any(x.get("decoded") is False for x in payload["candidates"])
    else:
        assert result == 2 and payload["status"] == "error" and payload["error_message"]
    audit(tmp_path, codec_tools, "truncated synthetic stream", [complete, broken], {
        "exit_code": result, "status": payload["status"], "sampling": payload.get("sampling"),
        "error_message": payload.get("error_message")})
