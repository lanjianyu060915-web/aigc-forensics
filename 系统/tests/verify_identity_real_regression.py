"""Strict verifier for a real identity-result evidence upgrade.

Usage:
    python tests/verify_identity_real_regression.py BASELINE CURRENT EVENTS
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any


TOLERANCE = 1e-6
REQUIRED_ARTIFACT_FIELDS = {
    "id",
    "kind",
    "path",
    "sha256",
    "size_bytes",
    "status",
    "description",
}
REQUIRED_ARTIFACT_KINDS = {
    "identity_face_detection",
    "identity_face_crop_before_alignment",
    "identity_face_aligned",
    "identity_best_reference_original",
    "identity_best_reference_face_detection",
    "identity_best_reference_aligned",
    "identity_query_best_reference_comparison",
    "identity_evidence_metadata",
}
REQUIRED_STAGES = (
    "detect_query_face",
    "locate_query_landmarks",
    "extract_identity_embedding",
    "load_identity_features",
    "compare_reference_images",
    "aggregate_identity_scores",
    "rank_identity_candidates",
    "align_query_face",
    "generate_identity_visuals",
    "identity_complete",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_close(label: str, old: Any, new: Any) -> float:
    delta = abs(float(new) - float(old))
    if delta > TOLERANCE:
        raise AssertionError(f"{label} delta {delta} exceeds {TOLERANCE}")
    return delta


def verify(baseline_path: Path, current_path: Path, events_path: Path) -> dict[str, Any]:
    baseline = _load_json(baseline_path)
    current = _load_json(current_path)
    if baseline.get("status") != "success" or current.get("status") != "success":
        raise AssertionError("baseline/current identity status must both be success")
    old_candidates = baseline.get("candidates")
    new_candidates = current.get("candidates")
    if not isinstance(old_candidates, list) or not isinstance(new_candidates, list):
        raise AssertionError("candidate arrays are missing")
    if len(old_candidates) != len(new_candidates):
        raise AssertionError("candidate count changed")

    maximum_delta = 0.0
    rows: list[dict[str, Any]] = []
    for index, (old, new) in enumerate(zip(old_candidates, new_candidates, strict=True), start=1):
        if old.get("identity_id") != new.get("identity_id"):
            raise AssertionError(f"rank {index} identity changed")
        if int(old.get("rank")) != int(new.get("rank")):
            raise AssertionError(f"rank {index} rank field changed")
        deltas = [
            _assert_close(f"rank {index} score", old["score"], new["score"]),
            _assert_close(
                f"rank {index} mean_similarity",
                old["mean_similarity"],
                new["mean_similarity"],
            ),
            _assert_close(
                f"rank {index} best_reference_score",
                old["best_reference_score"],
                new["best_reference_score"],
            ),
        ]
        old_refs = old.get("reference_scores", [])
        new_refs = new.get("reference_scores", [])
        if len(old_refs) != len(new_refs):
            raise AssertionError(f"rank {index} valid reference count changed")
        for ref_index, (old_ref, new_ref) in enumerate(
            zip(old_refs, new_refs, strict=True), start=1
        ):
            if Path(old_ref["reference_image"]).name != Path(
                new_ref["reference_image"]
            ).name:
                raise AssertionError(
                    f"rank {index} reference {ref_index} filename changed"
                )
            deltas.append(
                _assert_close(
                    f"rank {index} reference {ref_index}",
                    old_ref["similarity"],
                    new_ref["similarity"],
                )
            )
        maximum_delta = max(maximum_delta, *deltas)
        rows.append(
            {
                "rank": index,
                "identity_id": new["identity_id"],
                "score": new["score"],
                "best_reference_file_name": new.get("best_reference_file_name"),
                "valid_reference_count": new.get("valid_reference_count"),
                "skipped_reference_count": new.get("skipped_reference_count"),
                "gap_from_top1": new.get("gap_from_top1"),
            }
        )

    margin_delta = _assert_close(
        "top1_top2_margin",
        baseline["metrics"]["top1_top2_margin"],
        current["metrics"]["top1_top2_margin"],
    )
    maximum_delta = max(maximum_delta, margin_delta)

    evidence = current.get("visual_evidence")
    if not isinstance(evidence, dict) or evidence.get("status") != "complete":
        raise AssertionError("visual_evidence is not complete")
    if evidence.get("embedding_dimension") != 512:
        raise AssertionError("embedding_dimension is not 512")
    query_face = evidence.get("query_face")
    if not isinstance(query_face, dict) or len(query_face.get("keypoints_5", [])) != 5:
        raise AssertionError("five query landmarks are missing")

    task_root = current_path.parent.resolve()
    manifest = current.get("artifact_manifest")
    if not isinstance(manifest, list) or not manifest:
        raise AssertionError("artifact_manifest is empty")
    actual_kinds: set[str] = set()
    for item in manifest:
        if not isinstance(item, dict) or set(item) != REQUIRED_ARTIFACT_FIELDS:
            raise AssertionError("artifact manifest item has an invalid schema")
        relative = PurePosixPath(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(item["path"]):
            raise AssertionError(f"artifact path is not safe task-relative POSIX: {item['path']}")
        artifact_path = task_root.joinpath(*relative.parts)
        if not artifact_path.is_file():
            raise AssertionError(f"artifact does not exist: {item['path']}")
        if artifact_path.stat().st_size != int(item["size_bytes"]):
            raise AssertionError(f"artifact size mismatch: {item['path']}")
        if _sha256(artifact_path) != str(item["sha256"]):
            raise AssertionError(f"artifact SHA-256 mismatch: {item['path']}")
        actual_kinds.add(str(item["kind"]))
    if not REQUIRED_ARTIFACT_KINDS.issubset(actual_kinds):
        raise AssertionError(
            "missing required identity artifact kinds: "
            + ", ".join(sorted(REQUIRED_ARTIFACT_KINDS - actual_kinds))
        )

    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stages = [str(item.get("stage")) for item in events]
    cursor = -1
    for required in REQUIRED_STAGES:
        try:
            cursor = stages.index(required, cursor + 1)
        except ValueError as exc:
            raise AssertionError(f"required progress stage missing/out of order: {required}") from exc
    if events[-1].get("stage") != "identity_complete" or events[-1].get("status") != "success":
        raise AssertionError("identity progress did not end successfully")

    return {
        "status": "passed",
        "tolerance": TOLERANCE,
        "maximum_numeric_delta": maximum_delta,
        "ranking_unchanged": True,
        "candidate_rows": rows,
        "top1_top2_margin": current["metrics"]["top1_top2_margin"],
        "artifact_count": len(manifest),
        "artifact_kinds": sorted(actual_kinds),
        "progress_event_count": len(events),
        "required_progress_stages_in_order": list(REQUIRED_STAGES),
        "skipped_library_images": current.get("diagnostics", {}).get(
            "skipped_library_images", []
        ),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "usage: verify_identity_real_regression.py BASELINE CURRENT EVENTS",
            file=sys.stderr,
        )
        return 2
    summary = verify(Path(argv[1]), Path(argv[2]), Path(argv[3]))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
