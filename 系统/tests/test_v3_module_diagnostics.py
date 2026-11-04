from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_ROOT = ROOT / "modules" / "identity_search"
SOURCE_ROOT = ROOT / "modules" / "source_search"
for import_root in (ROOT, IDENTITY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from progress_events import ProgressEventWriter, create_progress_writer
from identity_search.extractor import FeatureExtractor
from identity_search.errors import MultipleFacesError, NoFaceError
from identity_search.searcher import IdentitySearcher
from source_search.image_io import write_image
from source_search.local_matcher import calculate_local_score
from source_search.pipeline import search_source_images


REQUIRED_EVENT_FIELDS = {
    "event_id",
    "timestamp",
    "task_id",
    "module",
    "stage",
    "status",
    "message",
    "progress_current",
    "progress_total",
    "progress_percent",
    "elapsed_seconds",
    "details",
}


class _Face:
    def __init__(self, embedding, bbox=(1, 2, 30, 40), det_score=0.987):
        self.embedding = embedding
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.det_score = det_score


class _FakeExtractor:
    def __init__(self, query_vector: np.ndarray):
        self.query_vector = query_vector

    def extract(self, path):
        path = Path(path)
        if path.name == "query.jpg":
            vector = self.query_vector
        elif path.parent.name == "person_a" and path.name == "ref_1.jpg":
            vector = np.r_[1.0, np.zeros(511)].astype(np.float32)
        elif path.parent.name == "person_a":
            vector = np.r_[0.8, 0.6, np.zeros(510)].astype(np.float32)
        else:
            vector = np.r_[0.0, 1.0, np.zeros(510)].astype(np.float32)
        return vector, _Face(vector)


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_progress_events_are_complete_atomic_and_monotonic(tmp_path: Path) -> None:
    events_path = tmp_path / "progress_events.jsonl"
    latest_path = tmp_path / "progress_latest.json"
    writer = ProgressEventWriter(
        "task-progress",
        "source",
        events_path,
        latest_path,
        min_interval_seconds=0,
        min_percent_delta=0,
    )
    writer.emit("scan", "running", "0 / 2", progress_current=0, progress_total=2)
    writer.emit("scan", "running", "1 / 2", progress_current=1, progress_total=2)
    regressed = writer.emit(
        "scan",
        "running",
        "reported lower percentage",
        progress_percent=25,
        force=True,
    )
    writer.emit("done", "success", "done", progress_percent=100, force=True)

    records = _events(events_path)
    assert records
    assert all(REQUIRED_EVENT_FIELDS == set(record) for record in records)
    assert json.loads(latest_path.read_text(encoding="utf-8")) == records[-1]
    assert regressed["progress_percent"] is None
    assert regressed["details"]["progress_percent_omitted_reason"] == "would_decrease"

    timestamps = [datetime.fromisoformat(record["timestamp"]) for record in records]
    assert timestamps == sorted(timestamps)
    assert len(timestamps) == len(set(timestamps))
    elapsed = [record["elapsed_seconds"] for record in records]
    assert elapsed == sorted(elapsed)
    numeric_percent = [
        record["progress_percent"]
        for record in records
        if record["progress_percent"] is not None
    ]
    assert numeric_percent == sorted(numeric_percent)

    # A restarted writer cannot publish a lower module percentage either.
    restarted = ProgressEventWriter(
        "task-progress",
        "source",
        events_path,
        latest_path,
        min_interval_seconds=0,
        min_percent_delta=0,
    )
    restart_event = restarted.emit(
        "retry", "running", "retry", progress_percent=10, force=True
    )
    assert restart_event["progress_percent"] is None
    json.loads(latest_path.read_text(encoding="utf-8"))


def test_progress_writer_rejects_partial_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        create_progress_writer(
            task_id="task",
            module="identity",
            events_path=tmp_path / "events.jsonl",
            latest_path=None,
        )


def test_identity_v3_fields_preserve_person_mean_ranking(tmp_path: Path) -> None:
    library = tmp_path / "identity_library"
    (library / "person_a").mkdir(parents=True)
    (library / "person_b").mkdir(parents=True)
    for relative in (
        Path("person_a/ref_1.jpg"),
        Path("person_a/ref_2.jpg"),
        Path("person_b/ref_1.jpg"),
    ):
        (library / relative).write_bytes(relative.as_posix().encode("ascii"))
    query = tmp_path / "query.jpg"
    query.write_bytes(b"query")
    query_vector = np.r_[1.0, np.zeros(511)].astype(np.float32)
    events_path = tmp_path / "identity_events.jsonl"
    latest_path = tmp_path / "identity_latest.json"
    reporter = ProgressEventWriter(
        "identity-task",
        "identity",
        events_path,
        latest_path,
        min_interval_seconds=0,
        min_percent_delta=0,
    )

    searcher = IdentitySearcher(
        library,
        cache_dir=tmp_path / "cache",
        extractor=_FakeExtractor(query_vector),
        progress_reporter=reporter,
        model_fingerprint="test-model",
    )
    result = searcher.search(query, top_k=2, progress_reporter=reporter)

    assert result["schema_version"] == "3.0"
    assert result["status"] == "success"
    assert [item["identity_id"] for item in result["candidates"]] == [
        "person_a",
        "person_b",
    ]
    first = result["candidates"][0]
    assert first["score"] == pytest.approx((1.0 + 0.8) / 2)
    assert first["mean_similarity"] == first["score"]
    assert first["reference_count"] == 2
    assert first["best_reference_score"] == pytest.approx(1.0)
    assert Path(first["best_reference_image"]).name == "ref_1.jpg"
    assert len(first["reference_scores"]) == 2

    metrics = result["metrics"]
    assert metrics["query_face_count"] == 1
    assert metrics["selected_face_bbox"] == [1.0, 2.0, 30.0, 40.0]
    assert metrics["face_detection_score"] == pytest.approx(0.987)
    assert metrics["embedding_dimension"] == 512
    assert metrics["library_identity_count"] == 2
    assert metrics["library_image_count"] == 3
    assert metrics["valid_embedding_count"] == 3
    assert metrics["top1_top2_margin"] == pytest.approx(first["score"])
    assert result["method"]["aggregation_method"] == "mean"
    assert result["method"]["open_set_rejection_enabled"] is False
    assert {event["stage"] for event in _events(events_path)} >= {
        "compare_identities",
        "identity_ranking_complete",
    }


@pytest.mark.parametrize("faces, exception_type, count", [
    ([], NoFaceError, 0),
    ([_Face(np.ones(512)), _Face(np.ones(512))], MultipleFacesError, 2),
])
def test_identity_face_failures_keep_true_diagnostics(
    faces, exception_type, count
) -> None:
    class _App:
        def get(self, _image):
            return faces

    extractor = FeatureExtractor(
        app=_App(), image_loader=lambda _path: np.zeros((10, 10, 3), np.uint8)
    )
    with pytest.raises(exception_type) as captured:
        extractor.extract("unused.jpg")
    assert captured.value.diagnostics["face_count"] == count
    assert len(captured.value.diagnostics["detected_faces"]) == count


def test_local_score_formulas_cover_ransac_and_insufficient_matches() -> None:
    local, components = calculate_local_score(
        20, 10, ransac_attempted=True
    )
    assert local == pytest.approx(0.6 * (10 / 20) + 0.4 * (10 / 40))
    assert components["mode"] == "normal"
    assert components["ransac_inlier_ratio"] == pytest.approx(0.5)

    limited, components = calculate_local_score(
        3, 0, ransac_attempted=False
    )
    assert limited == pytest.approx(3 / 40)
    assert components["mode"] == "insufficient_matches"
    assert components["formula"] == "min(good_match_count / 40, 0.15)"


def test_source_v3_diagnostics_and_real_count_progress(tmp_path: Path) -> None:
    query = tmp_path / "query.jpg"
    library = tmp_path / "source_library"
    library.mkdir()
    rng = np.random.default_rng(20260825)
    base = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    write_image(query, base)
    write_image(library / "same.jpg", base)
    write_image(library / "shifted.jpg", np.roll(base, 3, axis=1))
    write_image(library / "different.jpg", 255 - base)
    (library / "broken.jpg").write_bytes(b"not an image")

    events_path = tmp_path / "source_events.jsonl"
    latest_path = tmp_path / "source_latest.json"
    reporter = ProgressEventWriter(
        "source-task",
        "source",
        events_path,
        latest_path,
        min_interval_seconds=0,
        min_percent_delta=0,
    )
    candidates, diagnostics = search_source_images(
        query,
        library,
        top_k=2,
        output_dir=tmp_path / "output",
        visualization_stem="test",
        progress_reporter=reporter,
    )

    assert len(candidates) == 2
    first = candidates[0]
    assert {
        "query_keypoint_count",
        "candidate_keypoint_count",
        "good_match_count",
        "ransac_inlier_count",
        "ransac_inlier_ratio",
        "local_score_components",
        "global_weight",
        "local_weight",
    } <= set(first)
    assert first["final_score"] == pytest.approx(
        0.45 * first["global_score"] + 0.55 * first["local_score"],
        abs=2e-6,
    )
    assert diagnostics["library_file_count"] == 4
    assert diagnostics["library_decoded"] == 3
    assert diagnostics["library_skipped_count"] == 1
    assert diagnostics["shortlist_size"] == 3
    assert diagnostics["global_weight"] == 0.45
    assert diagnostics["local_weight"] == 0.55
    assert Path(diagnostics["masked_query_path"]).is_file()
    assert diagnostics["top1_top2_margin"] == round(
        candidates[0]["final_score"] - candidates[1]["final_score"], 6
    )

    records = _events(events_path)
    coarse = [r for r in records if r["stage"] == "scan_source_library"]
    review = [r for r in records if r["stage"] == "source_geometric_review"]
    assert [r["progress_current"] for r in coarse] == [1, 2, 3]
    assert all(r["progress_total"] == 3 for r in coarse)
    assert [r["progress_current"] for r in review] == [1, 2, 3]
    assert all(r["progress_total"] == 3 for r in review)
    numeric_percent = [
        r["progress_percent"]
        for r in records
        if r["progress_percent"] is not None
    ]
    assert numeric_percent == sorted(numeric_percent)
    assert records[-1]["stage"] == "source_search_complete"
    assert records[-1]["status"] == "success"
