"""视频抽帧 fixture：模拟无法解码的视频，返回结构化失败。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--selection", required=True)
args, _unknown = parser.parse_known_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)
Path(args.selection).write_text(
    json.dumps(
        {
            "schema_version": "3.0",
            "module": "video",
            "status": "error",
            "video_path": args.video,
            "key_frames": [],
            "error_message": "fixture：视频无法解码",
        },
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
print("VIDEO_FIXTURE_FAILED_AS_EXPECTED", flush=True)
