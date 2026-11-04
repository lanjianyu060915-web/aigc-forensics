from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "modules" / "source_search"
for import_root in (ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from source_search.feature_cache import SourceFeatureCache
from source_search.image_io import write_image
from source_search.pipeline import search_source_images


def _scene(seed: int, height: int = 240, width: int = 320) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (height // 8, width // 8, 3), dtype=np.uint8)
    return np.kron(image, np.ones((8, 8, 1), dtype=np.uint8))


def _library(tmp_path: Path, count: int = 8) -> tuple[Path, Path]:
    library = tmp_path / "中文 底图库"
    for index in range(count):
        write_image(library / f"scene_{index:02d}.jpg", _scene(100 + index))
    query = tmp_path / "查询 图.jpg"
    write_image(query, _scene(103))
    return query, library


COMPARED_FIELDS = (
    "file_path",
    "global_score",
    "local_score",
    "final_score",
    "query_keypoint_count",
    "candidate_keypoint_count",
    "raw_knn_pair_count",
    "good_match_count",
    "ransac_inlier_count",
    "ransac_inlier_ratio",
    "local_score_components",
    "global_score_components",
)


def _comparable(results: list[dict]) -> list[dict]:
    return [{key: item[key] for key in COMPARED_FIELDS} for item in results]


def test_cached_search_is_numerically_identical_to_realtime_scan(tmp_path: Path) -> None:
    query, library = _library(tmp_path)
    cache = tmp_path / "cache"

    realtime, realtime_diag = search_source_images(query, library, 5, tmp_path / "realtime")
    first, first_diag = search_source_images(
        query, library, 5, tmp_path / "first", feature_cache_dir=cache
    )
    second, second_diag = search_source_images(
        query, library, 5, tmp_path / "second", feature_cache_dir=cache
    )

    assert _comparable(first) == _comparable(realtime)
    assert _comparable(second) == _comparable(realtime)
    assert realtime_diag["metrics"]["feature_cache"] == {"enabled": False}
    assert first_diag["metrics"]["feature_cache"]["index_misses"] == 8
    assert first_diag["metrics"]["feature_cache"]["index_hits"] == 0
    assert first_diag["metrics"]["feature_cache"]["sift_misses"] == 8
    assert second_diag["metrics"]["feature_cache"]["index_hits"] == 8
    assert second_diag["metrics"]["feature_cache"]["index_misses"] == 0
    assert second_diag["metrics"]["feature_cache"]["sift_hits"] == 8
    assert second_diag["metrics"]["feature_cache"]["sift_misses"] == 0


def test_cached_search_writes_the_same_visual_evidence_files(tmp_path: Path) -> None:
    query, library = _library(tmp_path)
    cache = tmp_path / "cache"
    search_source_images(query, library, 3, tmp_path / "warm", feature_cache_dir=cache)

    _, realtime = search_source_images(query, library, 3, tmp_path / "realtime")
    _, cached = search_source_images(
        query, library, 3, tmp_path / "cached", feature_cache_dir=cache
    )

    def artifact_ids(diagnostics: dict) -> list[tuple[str, str]]:
        return [(item["id"], item["status"]) for item in diagnostics["artifact_manifest"]]

    assert artifact_ids(cached) == artifact_ids(realtime)
    assert all(
        item["sha256"] for item in cached["artifact_manifest"] if item["status"] == "success"
    )


def test_changed_file_only_rebuilds_that_entry(tmp_path: Path) -> None:
    query, library = _library(tmp_path)
    cache = tmp_path / "cache"
    search_source_images(query, library, 5, tmp_path / "first", feature_cache_dir=cache)

    changed = library / "scene_02.jpg"
    write_image(changed, _scene(777))
    os.utime(changed, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    results, diagnostics = search_source_images(
        query, library, 5, tmp_path / "second", feature_cache_dir=cache
    )
    realtime, _ = search_source_images(query, library, 5, tmp_path / "third")

    stats = diagnostics["metrics"]["feature_cache"]
    assert stats["index_misses"] == 1
    assert stats["index_hits"] == 7
    assert _comparable(results) == _comparable(realtime)


def test_added_and_removed_files_stay_consistent_with_realtime_scan(tmp_path: Path) -> None:
    query, library = _library(tmp_path)
    cache = tmp_path / "cache"
    search_source_images(query, library, 5, tmp_path / "first", feature_cache_dir=cache)

    (library / "scene_05.jpg").unlink()
    write_image(library / "scene_new.jpg", _scene(900))
    results, diagnostics = search_source_images(
        query, library, 5, tmp_path / "second", feature_cache_dir=cache
    )
    realtime, _ = search_source_images(query, library, 5, tmp_path / "third")

    assert _comparable(results) == _comparable(realtime)
    assert diagnostics["metrics"]["library_decoded"] == 8
    assert diagnostics["metrics"]["feature_cache"]["index_misses"] == 1


def test_corrupt_index_is_rebuilt_without_changing_results(tmp_path: Path) -> None:
    query, library = _library(tmp_path)
    cache = tmp_path / "cache"
    feature_cache = SourceFeatureCache(cache, library)
    baseline, _ = search_source_images(
        query, library, 5, tmp_path / "first", feature_cache_dir=cache
    )
    feature_cache.index_path.write_bytes(b"not a zip archive")

    results, diagnostics = search_source_images(
        query, library, 5, tmp_path / "second", feature_cache_dir=cache
    )

    stats = diagnostics["metrics"]["feature_cache"]
    assert stats["rebuilt_reason"].startswith("index_unreadable")
    assert stats["index_misses"] == 8
    assert _comparable(results) == _comparable(baseline)


def test_corrupt_sift_entry_is_recomputed(tmp_path: Path) -> None:
    query, library = _library(tmp_path)
    cache = tmp_path / "cache"
    baseline, _ = search_source_images(
        query, library, 5, tmp_path / "first", feature_cache_dir=cache
    )
    feature_cache = SourceFeatureCache(cache, library)
    for path in feature_cache.sift_dir.glob("*.npz"):
        path.write_bytes(b"broken")

    results, diagnostics = search_source_images(
        query, library, 5, tmp_path / "second", feature_cache_dir=cache
    )

    assert diagnostics["metrics"]["feature_cache"]["sift_misses"] == 8
    assert _comparable(results) == _comparable(baseline)


def test_undecodable_library_file_is_reported_as_skipped_on_every_run(tmp_path: Path) -> None:
    query, library = _library(tmp_path)
    cache = tmp_path / "cache"
    (library / "broken.jpg").write_bytes(b"not an image")

    _, first = search_source_images(
        query, library, 5, tmp_path / "first", feature_cache_dir=cache
    )
    _, second = search_source_images(
        query, library, 5, tmp_path / "second", feature_cache_dir=cache
    )
    _, realtime = search_source_images(query, library, 5, tmp_path / "third")

    for diagnostics in (first, second):
        assert diagnostics["metrics"]["library_skipped_count"] == 1
        assert diagnostics["metrics"]["library_skipped_paths"] == realtime["metrics"]["library_skipped_paths"]
    assert second["metrics"]["feature_cache"]["index_misses"] == 0


def test_unusable_cache_directory_falls_back_to_realtime_scan(tmp_path: Path) -> None:
    query, library = _library(tmp_path)
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, not a directory", encoding="utf-8")

    results, diagnostics = search_source_images(
        query, library, 5, tmp_path / "out", feature_cache_dir=blocked
    )
    realtime, _ = search_source_images(query, library, 5, tmp_path / "realtime")

    assert _comparable(results) == _comparable(realtime)
    assert diagnostics["metrics"]["feature_cache"] == {"enabled": False}
    assert any("回退实时扫描" in item for item in diagnostics["warnings"])


def test_empty_library_error_matches_realtime_scan(tmp_path: Path) -> None:
    query = tmp_path / "q.jpg"
    write_image(query, _scene(1))
    library = tmp_path / "empty"
    library.mkdir()

    with pytest.raises(ValueError, match="no decodable"):
        search_source_images(
            query, library, 5, tmp_path / "out", feature_cache_dir=tmp_path / "cache"
        )


def test_runner_accepts_feature_cache_argument(tmp_path: Path) -> None:
    query, library = _library(tmp_path)
    output = tmp_path / "task" / "source_result.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SOURCE_ROOT / "run_source_search.py"),
            "--query",
            str(query),
            "--library",
            str(library),
            "--top-k",
            "3",
            "--output",
            str(output),
            "--feature-cache-dir",
            str(tmp_path / "cache"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(SOURCE_ROOT),
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["metrics"]["feature_cache"]["enabled"] is True
    assert len(payload["candidates"]) == 3
