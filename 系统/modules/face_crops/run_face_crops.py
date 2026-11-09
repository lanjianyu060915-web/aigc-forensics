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

from audit import base_check, frame_identifier, new_audit, summarize


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
                     min_face_side: int = 64, include_all: bool = False, audit: dict | None = None):
    from cropper import FaceCropper
    from multiface import write_frame_crops

    audit = audit if audit is not None else new_audit()
    frame_id = frame_identifier(image.name)
    record = {"frame_id": frame_id, "source": image.name, "status": "pending", "detected_total": None,
              "timestamp_seconds": None, "count_basis": "unrecorded"}
    audit["frame_records"].append(record)
    frame = _read_image(image)
    if frame is None:
        record["status"] = "decode_failed"
        summarize(audit)
        raise ValueError(f"图片无法解码：{image.name}")
    try:
        inspection = FaceCropper(model_root).inspect_faces(frame, max_faces=max_faces)
    except Exception as exc:
        record.update(status="detection_failed", error_type=type(exc).__name__)
        summarize(audit)
        raise
    record.update({key: value for key, value in inspection.items() if key not in {"boxes", "detections"}})
    try:
        legacy, per_face = write_frame_crops(frame, inspection["boxes"], source=image.name, prefix="image",
                                           out_dir=out_dir, min_face_side=min_face_side, audit=audit,
                                           frame_id=frame_id, detection_records=inspection["detections"])
    except Exception:
        record["status"] = "crop_failed"
        summarize(audit)
        raise
    return (legacy, per_face) if include_all else legacy


def _recorded_boxes(item: dict, max_faces: int) -> tuple[list[tuple], list[dict], int | None, str]:
    """Preserve legacy filtering/sorting; retain formerly skipped malformed records in the audit."""
    raw_faces = item.get("faces")
    listed = raw_faces if isinstance(raw_faces, list) else []
    valid, invalid = [], []
    for detector_index, face in enumerate(listed):
        box = face.get("bbox") if isinstance(face, dict) else None
        if isinstance(box, list) and len(box) == 4:
            valid.append((detector_index, box))
        else:
            invalid.append({"detector_index": detector_index, "bbox": None, "selected_for_crop": False,
                            "exclusion_reason": "invalid_bbox"})
    basis = "recorded_faces" if isinstance(raw_faces, list) else "largest_face_only"
    if not valid:
        box = item.get("largest_face_bbox")
        if isinstance(box, list) and len(box) == 4:
            valid = [(None, box)]
            basis = "largest_face_fallback"
    ordered = sorted(valid, key=lambda pair: (pair[1][2] - pair[1][0]) * (pair[1][3] - pair[1][1]), reverse=True)
    boxes = [tuple(int(value) for value in box) for _, box in ordered[:max_faces]]
    records = []
    for rank, (detector_index, box) in enumerate(ordered, 1):
        try:
            recorded_box = [int(value) for value in box]
        except (TypeError, ValueError, OverflowError):
            recorded_box = None
        records.append({"face_id": rank, "detector_index": detector_index, "bbox": recorded_box,
                        "selected_for_crop": rank <= max_faces,
                        "exclusion_reason": None if rank <= max_faces else "over_limit"})
    for record in invalid:
        records.append({**record, "face_id": len(records) + 1})
    reported = item.get("face_count")
    if isinstance(reported, int) and not isinstance(reported, bool) and reported >= 0:
        total = reported
        if basis == "recorded_faces" and total != len(listed):
            total, basis = None, "inconsistent_recorded_count"
    else:
        total = len(listed) if basis == "recorded_faces" else None
    if (total is not None and total < len(ordered)) or (basis == "largest_face_fallback" and invalid):
        # An unresolved fallback observation may duplicate a malformed listed face.
        # Keep both records for diagnosis but do not claim a known unique total.
        total, basis = None, "inconsistent_or_unresolved_fallback"
    return boxes, records, total, basis


def crops_from_frames(selection: Path, frames_dir: Path, out_dir: Path, *, max_faces: int = 8,
                      min_face_side: int = 64, include_all: bool = False, audit: dict | None = None):
    from multiface import write_frame_crops

    payload = json.loads(selection.read_text(encoding="utf-8"))
    audit = audit if audit is not None else new_audit()
    crops: list[dict] = []
    all_faces: list[dict] = []
    for sequence, item in enumerate(payload.get("key_frames") or []):
        name = str(item.get("path") or item.get("file_name") or "")
        frame_index = item.get("frame_index")
        frame_id = frame_identifier(name, frame_index if isinstance(frame_index, int) and not isinstance(frame_index, bool) else sequence)
        record = {"frame_id": frame_id, "frame_index": frame_index, "source": name,
                  "timestamp_seconds": item.get("timestamp_seconds"), "timestamp_source": item.get("timestamp_source", "unrecorded"),
                  "status": "pending", "detected_total": None, "count_basis": "unrecorded"}
        audit["frame_records"].append(record)
        if not name:
            record["status"] = "missing_frame_path"
            continue
        try:
            boxes, records, total, basis = _recorded_boxes(item, max_faces)
        except (TypeError, ValueError, OverflowError, IndexError) as exc:
            if sequence < MAX_FRAME_CROPS:
                record.update(status="crop_failed", error_type=type(exc).__name__)
                summarize(audit)
                raise
            # These frames were never processed by the frozen route; audit-only
            # malformed metadata must not cause a new failure in that route.
            record.update(status="over_frame_limit", metadata_error_type=type(exc).__name__)
            continue
        record.update(detected_total=total, count_basis=basis, selected_count=len(boxes),
                      truncated_count=sum(row["exclusion_reason"] == "over_limit" for row in records))
        if sequence >= MAX_FRAME_CROPS:
            record["status"] = "over_frame_limit"
            for row in records:
                check = base_check(name, frame_id, row["face_id"], row["bbox"], timestamp_seconds=item.get("timestamp_seconds"))
                check.update(selected_for_crop=False, crop_status="frame_not_processed", exclusion_reason="over_frame_limit")
                audit["face_checks"].append(check)
            continue
        frame = _read_image(frames_dir / name)
        if frame is None:
            record["status"] = "decode_failed"
            for row in records:
                check = base_check(name, frame_id, row["face_id"], row["bbox"], timestamp_seconds=item.get("timestamp_seconds"))
                reason = "frame_decode_failed" if row["selected_for_crop"] else row["exclusion_reason"]
                check.update(selected_for_crop=row["selected_for_crop"], crop_status=reason,
                             exclusion_reason=reason)
                audit["face_checks"].append(check)
            continue
        record["status"] = "no_face" if total == 0 else "success"
        try:
            legacy, per_face = write_frame_crops(frame, boxes,
                                           source=name, prefix=Path(name).stem, out_dir=out_dir,
                                           min_face_side=min_face_side, is_video=True,
                                           timestamp_seconds=item.get("timestamp_seconds"),
                                           audit=audit, frame_id=frame_id, detection_records=records)
        except Exception:
            record["status"] = "crop_failed"
            summarize(audit)
            raise
        crops.extend(legacy)
        all_faces.extend(per_face)
    summarize(audit)
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
    result = {"status": "success", "error_message": None, "crops": [], "per_face": [], "face_audit": new_audit()}
    try:
        if args.max_faces < 1 or args.min_face_side < 1:
            raise ValueError("人脸数量上限和最小短边必须为正整数")
        if args.image is not None:
            result["crops"], result["per_face"] = crops_from_image(
                args.image, args.out_dir, args.model_root, max_faces=args.max_faces,
                min_face_side=args.min_face_side, include_all=True, audit=result["face_audit"])
        else:
            result["crops"], result["per_face"] = crops_from_frames(
                args.frame_selection, args.key_frames_dir, args.out_dir, max_faces=args.max_faces,
                min_face_side=args.min_face_side, include_all=True, audit=result["face_audit"])
    except Exception as exc:  # noqa: BLE001 - 裁脸失败只影响人脸分支，结果里说明原因
        result.update(status="error", error_message=f"{type(exc).__name__}: {exc}", crops=[])
    summarize(result["face_audit"])
    result["runtime_seconds"] = time.perf_counter() - started
    temporary = args.out_dir / f".crops.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out_dir / "crops.json")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
