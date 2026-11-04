from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_ROOT = ROOT / "modules" / "identity_search"
for import_root in (ROOT, IDENTITY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from identity_search.errors import ImageDecodeError
from identity_search.searcher import IdentitySearcher
from progress_events import ProgressEventWriter


class _Face:
    def __init__(self, embedding: np.ndarray):
        self.embedding = embedding
        self.bbox = np.asarray([20, 18, 82, 88], dtype=np.float32)
        self.det_score = 0.991
        self.kps = np.asarray(
            [[38, 43], [64, 43], [51, 56], [41, 70], [61, 70]],
            dtype=np.float32,
        )


class _EvidenceExtractor:
    def extract(self, path):
        path = Path(path)
        if path.name == "broken.jpg":
            raise ImageDecodeError("fixture damaged reference")
        if path.name == "query.jpg" or (
            path.parent.name == "person_a" and path.name == "01.jpg"
        ):
            vector = np.r_[1.0, np.zeros(511)].astype(np.float32)
        elif path.parent.name == "person_a":
            vector = np.r_[0.8, 0.6, np.zeros(510)].astype(np.float32)
        else:
            vector = np.r_[0.0, 1.0, np.zeros(510)].astype(np.float32)
        return vector, _Face(vector)


def _write_face_fixture(path: Path, value: int) -> None:
    image = np.full((106, 104, 3), value, dtype=np.uint8)
    cv2.circle(image, (39, 43), 5, (30, 30, 30), -1)
    cv2.circle(image, (64, 43), 5, (30, 30, 30), -1)
    cv2.ellipse(image, (51, 70), (16, 8), 0, 0, 180, (50, 50, 50), 2)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_identity_visual_evidence_is_real_relative_and_score_invariant(
    tmp_path: Path,
) -> None:
    library = tmp_path / "身份库"
    for relative, value in (
        (Path("person_a/01.jpg"), 215),
        (Path("person_a/02.jpg"), 205),
        (Path("person_a/broken.jpg"), 195),
        (Path("person_b/01.jpg"), 185),
    ):
        _write_face_fixture(library / relative, value)
    query = tmp_path / "任务输入" / "query.jpg"
    _write_face_fixture(query, 225)
    events_path = tmp_path / "task" / "logs" / "progress_events.jsonl"
    latest_path = tmp_path / "task" / "logs" / "progress_latest.json"
    reporter = ProgressEventWriter(
        "case_identity_evidence",
        "identity",
        events_path,
        latest_path,
        min_interval_seconds=0,
        min_percent_delta=0,
    )
    searcher = IdentitySearcher(
        library,
        cache_dir=tmp_path / "cache",
        extractor=_EvidenceExtractor(),
        progress_reporter=reporter,
        model_fingerprint="fixture-buffalo_l",
    )

    plain = searcher.search(query, top_k=2)
    rich = searcher.search(
        query,
        top_k=2,
        progress_reporter=reporter,
        artifact_root=tmp_path / "task",
        task_id="case_identity_evidence",
    )

    assert [item["identity_id"] for item in rich["candidates"]] == [
        "person_a",
        "person_b",
    ]
    assert rich["candidates"][0]["score"] == pytest.approx(0.9)
    assert rich["candidates"][0]["best_reference_score"] == pytest.approx(1.0)
    assert rich["candidates"][0]["gap_from_top1"] == pytest.approx(0.0)
    assert rich["candidates"][1]["gap_from_top1"] == pytest.approx(0.9)
    assert [item["score"] for item in rich["candidates"]] == pytest.approx(
        [item["score"] for item in plain["candidates"]], abs=0.0
    )
    assert [item["identity_id"] for item in rich["candidates"]] == [
        item["identity_id"] for item in plain["candidates"]
    ]

    top1 = rich["candidates"][0]
    assert top1["valid_reference_count"] == 2
    assert top1["skipped_reference_count"] == 1
    assert top1["skipped_reference_images"][0]["reference_file_name"] == "broken.jpg"
    assert top1["skipped_reference_images"][0]["reason"] == "ImageDecodeError"
    assert all(item["status"] == "valid" for item in top1["reference_scores"])

    evidence = rich["visual_evidence"]
    assert evidence["status"] == "complete"
    assert evidence["embedding_dimension"] == 512
    assert evidence["query_face"]["landmark_count"] == 5
    assert len(evidence["query_face"]["keypoints_5"]) == 5
    assert evidence["best_reference"]["evidence_similarity_delta"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert len(evidence["candidate_card_images"]) == 2
    assert all(item["status"] == "success" for item in evidence["candidate_card_images"])
    assert all(item["candidate_card_image"] for item in rich["candidates"])
    assert len(evidence["per_reference_scores"]) == 2

    manifest = rich["artifact_manifest"]
    assert len(manifest) >= 8
    required = {"id", "kind", "path", "sha256", "size_bytes", "status", "description"}
    for item in manifest:
        assert set(item) == required
        assert item["status"] == "success"
        assert not Path(item["path"]).is_absolute()
        assert "\\" not in item["path"]
        artifact = tmp_path / "task" / Path(item["path"])
        assert artifact.is_file()
        assert artifact.stat().st_size == item["size_bytes"] > 0
        assert _sha256(artifact) == item["sha256"]

    stages = [
        json.loads(line)["stage"]
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    for expected in (
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
    ):
        assert expected in stages
    assert stages.index("detect_query_face") < stages.index("compare_reference_images")
    assert stages.index("compare_reference_images") < stages.index("rank_identity_candidates")
    assert stages.index("rank_identity_candidates") < stages.index("generate_identity_visuals")


def test_cache_hit_preserves_real_skipped_reference_reasons(tmp_path: Path) -> None:
    library = tmp_path / "identity_library"
    _write_face_fixture(library / "person_a" / "01.jpg", 210)
    _write_face_fixture(library / "person_a" / "broken.jpg", 190)
    query = tmp_path / "query.jpg"
    _write_face_fixture(query, 220)
    cache = tmp_path / "cache"

    built = IdentitySearcher(
        library,
        cache_dir=cache,
        extractor=_EvidenceExtractor(),
        model_fingerprint="fixture-buffalo_l",
    )
    assert built.diagnostics["cache_status"] == "miss_built"
    assert built.diagnostics["skipped_images"][0]["reason"] == "ImageDecodeError"

    loaded = IdentitySearcher(
        library,
        cache_dir=cache,
        extractor=_EvidenceExtractor(),
        model_fingerprint="fixture-buffalo_l",
    )
    assert loaded.diagnostics["cache_status"] == "hit"
    assert loaded.diagnostics["skipped_images"] == built.diagnostics["skipped_images"]
    result = loaded.search(query, top_k=1)
    skipped = result["candidates"][0]["skipped_reference_images"]
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "ImageDecodeError"


class _ForbiddenExtractor:
    """搬动后不应重新提取任何参考图。"""

    def extract(self, path: Path):
        raise AssertionError(f"不应重新提取：{path}")


def test_moved_library_reuses_index_without_rebuilding(tmp_path: Path) -> None:
    library = tmp_path / "identity_library"
    _write_face_fixture(library / "person_a" / "01.jpg", 210)
    cache = tmp_path / "cache"
    IdentitySearcher(
        library,
        cache_dir=cache,
        extractor=_EvidenceExtractor(),
        model_fingerprint="fixture-buffalo_l",
    )

    moved = tmp_path / "搬动后 目录" / "identity_library"
    moved.parent.mkdir()
    library.rename(moved)
    loaded = IdentitySearcher(
        moved,
        cache_dir=cache,
        extractor=_ForbiddenExtractor(),
        model_fingerprint="fixture-buffalo_l",
    )
    assert loaded.diagnostics["cache_status"] == "hit"
    assert loaded.identities["person_a"]["paths"] == ["person_a/01.jpg"]
    assert len(list(cache.glob("identity_index_*.json"))) == 2


@pytest.mark.parametrize("change", ["content", "model"])
def test_relocated_cache_is_not_adopted_when_library_or_model_differs(
    tmp_path: Path, change: str
) -> None:
    library = tmp_path / "identity_library"
    _write_face_fixture(library / "person_a" / "01.jpg", 210)
    cache = tmp_path / "cache"
    IdentitySearcher(
        library,
        cache_dir=cache,
        extractor=_EvidenceExtractor(),
        model_fingerprint="fixture-buffalo_l",
    )

    moved = tmp_path / "moved" / "identity_library"
    moved.parent.mkdir()
    library.rename(moved)
    if change == "content":
        _write_face_fixture(moved / "person_a" / "01.jpg", 150)
    rebuilt = IdentitySearcher(
        moved,
        cache_dir=cache,
        extractor=_EvidenceExtractor(),
        model_fingerprint="other-model" if change == "model" else "fixture-buffalo_l",
    )
    assert rebuilt.diagnostics["cache_status"] == "miss_built"
