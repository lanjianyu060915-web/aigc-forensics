# -*- coding: utf-8 -*-
"""裁出任务里的人脸，供通用 AIGC 引擎的人脸分支打分。

图片：检测最大人脸后裁剪。视频：用抽帧阶段记录的人脸框直接裁，不再检测。
结果写到 <out-dir>/crops.json；没有人脸不算失败，crops 为空。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

MODULE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_ROOT))

MAX_FRAME_CROPS = 5


def _read_image(path: Path):
    import cv2
    import numpy as np

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _write_crop(crop, target: Path) -> bool:
    import cv2

    target.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 98])
    if ok:
        encoded.tofile(str(target))
    return bool(ok)


def crops_from_image(image: Path, out_dir: Path, model_root: Path) -> list[dict]:
    from cropper import FaceCropper, square_crop

    frame = _read_image(image)
    if frame is None:
        raise ValueError(f"图片无法解码：{image.name}")
    box = FaceCropper(model_root).largest_face_box(frame)
    if box is None:
        return []
    crop = square_crop(frame, box)
    target = out_dir / "image_face.jpg"
    if crop is None or not _write_crop(crop, target):
        return []
    return [{"file": target.name, "source": image.name, "bbox": list(box)}]


def crops_from_frames(selection: Path, frames_dir: Path, out_dir: Path) -> list[dict]:
    from cropper import square_crop

    payload = json.loads(selection.read_text(encoding="utf-8"))
    crops: list[dict] = []
    for item in (payload.get("key_frames") or [])[:MAX_FRAME_CROPS]:
        box = item.get("largest_face_bbox")
        name = str(item.get("path") or item.get("file_name") or "")
        if not name or not isinstance(box, list) or len(box) != 4:
            continue
        frame = _read_image(frames_dir / name)
        if frame is None:
            continue
        crop = square_crop(frame, tuple(int(value) for value in box))
        target = out_dir / f"{Path(name).stem}_face.jpg"
        if crop is not None and _write_crop(crop, target):
            crops.append(
                {
                    "file": target.name,
                    "source": name,
                    "bbox": [int(value) for value in box],
                    "timestamp_seconds": item.get("timestamp_seconds"),
                }
            )
    return crops


def main() -> int:
    parser = argparse.ArgumentParser(description="裁出任务里的人脸")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--frame-selection", type=Path)
    parser.add_argument("--key-frames-dir", type=Path)
    parser.add_argument("--model-root", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = {"status": "success", "error_message": None, "crops": []}
    try:
        if args.image is not None:
            result["crops"] = crops_from_image(args.image, args.out_dir, args.model_root)
        else:
            result["crops"] = crops_from_frames(args.frame_selection, args.key_frames_dir, args.out_dir)
    except Exception as exc:  # noqa: BLE001 - 裁脸失败只影响人脸分支，结果里说明原因
        result.update(status="error", error_message=f"{type(exc).__name__}: {exc}", crops=[])
    result["runtime_seconds"] = time.perf_counter() - started
    temporary = args.out_dir / f".crops.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out_dir / "crops.json")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
