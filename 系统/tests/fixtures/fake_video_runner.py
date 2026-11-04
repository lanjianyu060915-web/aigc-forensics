"""视频抽帧 fixture：不读真实视频，只生成确定性的关键帧与选择证据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--selection", required=True)
args, _unknown = parser.parse_known_args()

frames_dir = Path(args.output_dir)
frames_dir.mkdir(parents=True, exist_ok=True)
key_frames = []
for rank, timestamp in enumerate((0.5, 1.5, 2.5), start=1):
    name = f"frame_{rank:03d}.jpg"
    (frames_dir / name).write_bytes(b"fixture-jpeg-frame")
    key_frames.append(
        {
            "frame_index": rank * 10,
            "timestamp_seconds": timestamp,
            "decoded": True,
            "width": 64,
            "height": 64,
            "sharpness": 100.0,
            "brightness": 120.0,
            "contrast": 40.0,
            "face_count": 1,
            "quality_score": 0.9 - 0.01 * rank,
            "selected": True,
            "selection_rank": rank,
            "selection_reasons": ["单人脸帧优先（身份检索要求恰好一张人脸）"],
            "frame_rank": rank,
            "file_name": name,
            "path": name,
            "sha256": "0" * 64,
            "size_bytes": 19,
        }
    )

selection = {
    "schema_version": "3.0",
    "module": "video",
    "status": "success",
    "video_path": args.video,
    "video_sha256": "c" * 64,
    "video": {
        "file_name": Path(args.video).name,
        "size_bytes": 19,
        "fps": 25.0,
        "frame_count": 75,
        "width": 64,
        "height": 64,
        "duration_seconds": 3.0,
    },
    "sampling": {
        "strategy": "uniform_trimmed",
        "candidate_frames_requested": 6,
        "candidate_frames_decoded": 3,
        "candidate_indices": [2, 30, 60],
        "decode_strategy": "sequential_grab",
        "jpeg_quality": 95,
    },
    "face_detection": {
        "requested": False,
        "available": False,
        "method": "none",
        "frames_with_single_face": 3,
        "frames_with_multiple_faces": 0,
        "frames_without_face": 0,
    },
    "selection": {
        "rule": "按质量分降序贪心选帧",
        "max_key_frames": 3,
        "selected_frames": 3,
        "min_temporal_gap_seconds": 0.25,
        "min_sharpness_ratio": 0.35,
        "prefer_single_face": True,
        "weights": {"face": 0.55, "sharpness": 0.3, "brightness": 0.15},
        "sharpness_reference": 100.0,
        "tiers": [],
    },
    "key_frames": key_frames,
    "primary_frame": key_frames[0],
    "candidates": key_frames,
    "warnings": [],
    "limitations": ["关键帧是有损抽样：未被选中的帧没有参与任何模块。"],
}
Path(args.selection).write_text(json.dumps(selection, ensure_ascii=False), encoding="utf-8")
print("VIDEO_FIXTURE_RAN", flush=True)
