# -*- coding: utf-8 -*-
"""裁出任务里的人脸，供通用 AIGC 引擎的人脸分支打分。

图片：逐脸检测并裁剪。视频：用抽帧阶段记录的逐脸框直接裁，不再检测。
结果写到 <out-dir>/crops.json；旧 crops 保留最大脸，新增 per_face 保存全部展示记录。
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


def crops_from_image(image: Path, out_dir: Path, model_root: Path, *, max_faces: int = 8,
                     min_face_side: int = 64, include_all: bool = False):
    from cropper import FaceCropper
    from multiface import write_frame_crops

    frame = _read_image(image)
    if frame is None:
        raise ValueError(f"图片无法解码：{image.name}")
    boxes = FaceCropper(model_root).face_boxes(frame, max_faces=max_faces)
    legacy, per_face = write_frame_crops(frame, boxes, source=image.name, prefix="image",
                                       out_dir=out_dir, min_face_side=min_face_side)
    return (legacy, per_face) if include_all else legacy


def crops_from_frames(selection: Path, frames_dir: Path, out_dir: Path, *, max_faces: int = 8,
                      min_face_side: int = 64, include_all: bool = False):
    from multiface import write_frame_crops

    payload = json.loads(selection.read_text(encoding="utf-8"))
    crops: list[dict] = []
    all_faces: list[dict] = []
    for item in (payload.get("key_frames") or [])[:MAX_FRAME_CROPS]:
        name = str(item.get("path") or item.get("file_name") or "")
        if not name:
            continue
        boxes = [face.get("bbox") for face in item.get("faces") or [] if isinstance(face, dict)]
        boxes = [box for box in boxes if isinstance(box, list) and len(box) == 4]
        if not boxes:
            box = item.get("largest_face_bbox")
            boxes = [box] if isinstance(box, list) and len(box) == 4 else []
        boxes = sorted(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)[:max_faces]
        frame = _read_image(frames_dir / name)
        if frame is None:
            continue
        legacy, per_face = write_frame_crops(frame, [tuple(int(value) for value in box) for box in boxes],
                                           source=name, prefix=Path(name).stem, out_dir=out_dir,
                                           min_face_side=min_face_side, is_video=True,
                                           timestamp_seconds=item.get("timestamp_seconds"))
        crops.extend(legacy)
        all_faces.extend(per_face)
    return (crops, all_faces) if include_all else crops


def main() -> int:
    parser = argparse.ArgumentParser(description="裁出任务里的人脸")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--frame-selection", type=Path)
    parser.add_argument("--key-frames-dir", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--max-faces", type=int, default=8)
    parser.add_argument("--min-face-side", type=int, default=64)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = {"status": "success", "error_message": None, "crops": [], "per_face": []}
    try:
        if args.max_faces < 1 or args.min_face_side < 1:
            raise ValueError("人脸数量上限和最小短边必须为正整数")
        if args.image is not None:
            result["crops"], result["per_face"] = crops_from_image(
                args.image, args.out_dir, args.model_root, max_faces=args.max_faces,
                min_face_side=args.min_face_side, include_all=True)
        else:
            result["crops"], result["per_face"] = crops_from_frames(
                args.frame_selection, args.key_frames_dir, args.out_dir, max_faces=args.max_faces,
                min_face_side=args.min_face_side, include_all=True)
    except Exception as exc:  # noqa: BLE001 - 裁脸失败只影响人脸分支，结果里说明原因
        result.update(status="error", error_message=f"{type(exc).__name__}: {exc}", crops=[])
    result["runtime_seconds"] = time.perf_counter() - started
    temporary = args.out_dir / f".crops.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out_dir / "crops.json")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
