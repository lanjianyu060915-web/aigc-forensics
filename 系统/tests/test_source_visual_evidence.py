from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "modules" / "source_search"
for import_root in (ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from progress_events import ProgressEventWriter
from source_search.global_features import (
    extract_global,
    global_similarity,
    global_similarity_with_components,
)
from source_search.image_io import write_image
from source_search.pipeline import search_source_images
import source_search.pipeline as source_pipeline
import run_source_search


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_global_component_export_is_numerically_identical() -> None:
    rng = np.random.default_rng(20260831)
    first = rng.integers(0, 256, (137, 193, 3), dtype=np.uint8)
    second = np.roll(first, 7, axis=1)
    mask = np.full(first.shape[:2], 255, dtype=np.uint8)
    a = extract_global(first, mask)
    b = extract_global(second, mask)

    legacy = global_similarity(a, b)
    score, components = global_similarity_with_components(a, b)

    assert score == legacy
    assert components["score"] == score
    assert sum(components["weighted_terms"].values()) == pytest.approx(
        score, abs=1e-15
    )
    assert components["weights"] == {
        "hsv_histogram_similarity": 0.30,
        "grayscale_thumbnail_similarity": 0.25,
        "edge_map_similarity": 0.15,
        "border_region_similarity": 0.30,
    }


def test_source_visual_evidence_manifest_and_true_stage_events(
    tmp_path: Path,
) -> None:
    query = tmp_path / "查询 图.jpg"
    library = tmp_path / "中文 底图库"
    task_dir = tmp_path / "任务 case_source_evidence"
    library.mkdir()
    rng = np.random.default_rng(7391)
    base = rng.integers(0, 256, (160, 192, 3), dtype=np.uint8)
    write_image(query, base)
    write_image(library / "相同.jpg", base)
    write_image(library / "平移.jpg", np.roll(base, 4, axis=1))
    write_image(library / "反色.jpg", 255 - base)
    (library / "损坏.jpg").write_bytes(b"not an image")

    events_path = task_dir / "logs" / "progress_events.jsonl"
    reporter = ProgressEventWriter(
        "case_source_evidence",
        "source",
        events_path,
        task_dir / "logs" / "progress_latest.json",
        min_interval_seconds=0,
        min_percent_delta=0,
    )
    candidates, diagnostics = search_source_images(
        query,
        library,
        top_k=3,
        output_dir=task_dir,
        visualization_stem="source_result_testhash",
        progress_reporter=reporter,
    )

    assert len(candidates) == 3
    assert [item["rank"] for item in candidates] == [1, 2, 3]
    first = candidates[0]
    # Frozen fields remain available and the new score decomposition recombines
    # to the exact unrounded value used by the sorter.
    assert {
        "file_path",
        "global_score",
        "local_score",
        "final_score",
        "visualization_path",
    } <= set(first)
    assert first["score_fusion"]["final_score"] == pytest.approx(
        0.45 * first["score_fusion"]["global_score"]
        + 0.55 * first["score_fusion"]["local_score"],
        abs=1e-15,
    )
    assert first["global_score_components"]["score"] == pytest.approx(
        first["score_fusion"]["global_score"], abs=1e-15
    )

    evidence = diagnostics["visual_evidence"]["source"]
    assert evidence["global_stage"] == {
        "library_file_count": 4,
        "library_decoded": 3,
        "library_skipped_count": 1,
        "library_skipped_paths": [str(library / "损坏.jpg")],
        "global_compared_count": 3,
        "shortlist_size": 3,
        "returned_top_k": 3,
    }
    assert evidence["query"]["face_masked_image"].startswith(
        "visualizations/source/"
    )
    assert len(evidence["global_scoring"]["candidates"]) == 3
    assert len(evidence["top_candidates"]) == 3
    assert evidence["local_matching"]["lowe_good_match_count"] == first[
        "good_match_count"
    ]
    assert evidence["fusion"]["displayed_final_score"] == first[
        "final_score"
    ]

    manifest = diagnostics["artifact_manifest"]
    assert manifest
    ids = {item["id"] for item in manifest}
    assert {
        "source_query_face_detection",
        "source_query_face_masked",
        "source_candidate_001_image",
        "source_candidate_001_matches_inliers",
        "source_top1_query_keypoints",
        "source_top1_candidate_keypoints",
        "source_top1_matches_all_lowe",
        "source_top1_matches_inliers_outliers",
        "source_top1_homography_projection",
    } <= ids
    for item in manifest:
        assert set(item) == {
            "id",
            "kind",
            "path",
            "sha256",
            "size_bytes",
            "status",
            "description",
        }
        if item["status"] == "success":
            assert item["path"]
            assert "\\" not in item["path"]
            artifact = task_dir / Path(item["path"])
            assert artifact.is_file()
            assert artifact.stat().st_size == item["size_bytes"] > 0
            assert _sha256(artifact) == item["sha256"]

    stages = [event["stage"] for event in _records(events_path)]
    required = [
        "read_source_query",
        "detect_source_query_face",
        "mask_query_face",
        "extract_query_global_features",
        "decode_source_library",
        "scan_source_library",
        "source_global_ranking",
        "source_geometric_review",
        "source_sift_features",
        "source_lowe_filter",
        "source_ransac_verification",
        "source_score_fusion",
        "source_final_ranking",
        "generate_source_visual_evidence",
        "source_search_complete",
    ]
    for stage in required:
        assert stage in stages
    assert [stages.index(stage) for stage in required] == sorted(
        stages.index(stage) for stage in required
    )
    assert stages[-1] == "source_search_complete"


def test_source_runner_publishes_evidence_at_top_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = tmp_path / "query.jpg"
    library = tmp_path / "library"
    output = tmp_path / "task" / "source_result.json"
    library.mkdir()
    image = np.random.default_rng(55).integers(
        0, 256, (128, 128, 3), dtype=np.uint8
    )
    write_image(query, image)
    write_image(library / "candidate.jpg", image)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_source_search.py",
            "--query",
            str(query),
            "--library",
            str(library),
            "--top-k",
            "1",
            "--output",
            str(output),
        ],
    )

    assert run_source_search.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["visual_evidence"]["source"]["top_candidates"]
    assert payload["artifact_manifest"]
    assert payload["diagnostics"]["visual_evidence"] == payload[
        "visual_evidence"
    ]


def test_optional_evidence_failure_does_not_change_successful_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = tmp_path / "query.jpg"
    library = tmp_path / "library"
    library.mkdir()
    image = np.random.default_rng(77).integers(
        0, 256, (128, 128, 3), dtype=np.uint8
    )
    write_image(query, image)
    write_image(library / "candidate.jpg", image)

    def fail_keypoint_evidence(*_args, **_kwargs):
        raise RuntimeError("injected optional visualization failure")

    monkeypatch.setattr(
        source_pipeline,
        "make_keypoints_visualization",
        fail_keypoint_evidence,
    )
    candidates, diagnostics = search_source_images(
        query,
        library,
        top_k=1,
        output_dir=tmp_path / "task",
        visualization_stem="failure_isolation",
    )

    assert candidates and candidates[0]["rank"] == 1
    assert candidates[0]["final_score"] > 0
    assert any(
        "injected optional visualization failure" in warning
        for warning in diagnostics["warnings"]
    )
    manifest = {
        item["id"]: item for item in diagnostics["artifact_manifest"]
    }
    assert manifest["source_top1_query_keypoints"]["status"] == (
        "not_generated"
    )
    assert manifest["source_top1_candidate_keypoints"]["status"] == (
        "not_generated"
    )
    assert manifest["source_candidate_001_matches_inliers"]["status"] == (
        "success"
    )
