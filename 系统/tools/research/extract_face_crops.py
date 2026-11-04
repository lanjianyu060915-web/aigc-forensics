# -*- coding: utf-8 -*-
"""从图片和视频里裁人脸，供人脸分支的特征提取和评测使用。

用界面环境运行（需要 insightface 和 opencv）。输入是 eval_general_aigc.py 那种 JSONL 清单，
输出的 crops.jsonl 每行一张裁剪图。没检出人脸的样本记录下来，不丢。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules.face_crops.cropper import FaceCropper, video_frames  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--assume-cropped", action="store_true", help="输入已是对齐人脸，不再检测")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cropper = None if args.assume_cropped else FaceCropper(ROOT / "packaging_v0.3" / "resources")
    index_path = args.out / "crops.jsonl"
    done: set[str] = set()
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as stream:
            done = {json.loads(line)["sample_id"] for line in stream if line.strip()}
    samples = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    with index_path.open("a", encoding="utf-8") as index:
        for position, sample in enumerate(samples, start=1):
            if sample["sample_id"] in done:
                continue
            path = Path(sample["path"])
            frames = (
                [read_image(path)]
                if path.suffix.lower() in IMAGE_SUFFIXES
                else video_frames(path, args.frames)
            )
            crops = []
            for number, frame in enumerate(frames):
                if frame is None:
                    continue
                face = frame if args.assume_cropped else cropper.largest_face_crop(frame)
                if face is None:
                    continue
                name = f"{abs(hash(sample['sample_id'])) % 10**12:012d}_{number}.jpg"
                target = args.out / "images" / name
                target.parent.mkdir(exist_ok=True)
                ok, encoded = cv2.imencode(".jpg", face, [cv2.IMWRITE_JPEG_QUALITY, 98])
                if ok:
                    encoded.tofile(str(target))
                    crops.append(f"images/{name}")
            index.write(
                json.dumps(
                    {
                        "sample_id": sample["sample_id"],
                        "label": sample["label"],
                        "source": sample.get("source"),
                        "split": sample.get("split"),
                        "group_id": sample.get("group_id"),
                        "crops": crops,
                        "frames_tried": len(frames),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            index.flush()
            if position % 50 == 0:
                print(f"{position}/{len(samples)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
