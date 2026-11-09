"""Pure bookkeeping for task-local face coverage; no model or media access."""
from __future__ import annotations

import hashlib
from typing import Any


def frame_identifier(source: str, frame_index: int | None = None) -> str:
    suffix = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"frame-{frame_index}-{suffix}" if frame_index is not None else f"frame-{suffix}"


def new_audit() -> dict[str, Any]:
    return {"schema_version": "1.0", "scope": "task_local_face_checks", "frame_records": [],
            "face_checks": [], "coverage": {}, "identity_coverage": {"status": "unrecorded"},
            "pair_coverage": {"status": "unrecorded"}}


def base_check(source: str, frame_id: str, index: int, bbox, *, timestamp_seconds=None) -> dict[str, Any]:
    uid = f"{frame_id}/face-{index:04d}"
    return {"face_id": index, "face_uid": uid, "face_uid_scope": "task",
            "detection_id": uid, "detection_scope_id": f"face-crops/{frame_id}",
            "frame_id": frame_id, "source_id": source, "source": source,
            "coordinate_space": "original_image", "bbox": list(bbox) if bbox is not None else None,
            "timestamp_seconds": timestamp_seconds, "crop_status": "not_attempted",
            "score_status": "not_run", "score": None, "used_for_band": False,
            "exclusion_reason": None, "quality_status": "unassessed",
            "identity_processed": None, "pair_processed": None}


def summarize(audit: dict[str, Any]) -> None:
    frames, checks = audit["frame_records"], audit["face_checks"]
    known = [row.get("detected_total") for row in frames]
    complete = bool(frames) and all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in known)
    failed = {"invalid_bbox", "crop_failed", "encode_failed", "write_failed", "frame_decode_failed"}
    audit["coverage"] = {
        "module": "face_crops", "count_status": "recorded" if complete else "incomplete",
        "count_unit": "face_observations_per_frame",
        "detected_total": sum(known) if complete else None,
        "recorded_face_count": len(checks),
        "selected_count": sum(bool(row.get("selected_for_crop")) for row in checks),
        "processed_count": sum(row.get("crop_status") not in {"not_attempted", "over_limit", "frame_not_processed"} for row in checks),
        "cropped_count": sum(row.get("crop_status") == "success" for row in checks),
        "truncated_count": sum(row.get("crop_status") == "over_limit" for row in checks),
        "failed_count": sum(row.get("crop_status") in failed for row in checks),
        "frame_failed_count": sum(row.get("status") in {"decode_failed", "detection_failed", "crop_failed", "missing_frame_path"} for row in frames),
        "frames_recorded": len(frames),
        "frames_not_processed": sum(row.get("status") == "over_frame_limit" for row in frames),
        "scored_count": None, "used_for_band_count": None,
    }
    coverage = audit["coverage"]
    audit["status"] = ("partial" if coverage["failed_count"] or coverage["frame_failed_count"]
                       or coverage["truncated_count"] or coverage["frames_not_processed"] else
                       "no_face" if complete and coverage["detected_total"] == 0 else
                       "recorded" if complete else "incomplete")
