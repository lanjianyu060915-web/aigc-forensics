"""Explicit-selection engineering checks using only generated pixels/fake models.

Equality here verifies plumbing and normalization, never real recognition quality.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "modules/identity_search", ROOT / "modules/identity_search_v2"):
    sys.path.insert(0, str(location))

from identity_search.errors import InputValidationError, InvalidEmbeddingError, MultipleFacesError, NoFaceError
from identity_search.extractor import FeatureExtractor
from identity_search.searcher import IdentitySearcher
from identity_search.selected_face import INTERFACE_VERSION, validate_selected_face
from scale_fallback import attempt_scale_chain, selection_trace, write_main_face_query


BOX = [10, 10, 90, 90]
POINTS = [[30, 35], [70, 35], [50, 50], [35, 70], [65, 70]]
CONTEXT = {"source_id": "independent-input", "frame_id": "image-0"}


def make_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((100, 100, 3), 173, dtype=np.uint8)
    image[10:20, 30:40] = (2, 4, 6)
    path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
    return image


def selection(path, **updates):
    return {"interface_version": INTERFACE_VERSION, **CONTEXT,
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "image_size": [100, 100],
            "coordinate_space": "original_image", "bbox": list(BOX), "keypoints_5": deepcopy(POINTS),
            "detection_scope_id": "synthetic-detection-0", "detection_id": "face-observation-0",
            "det_score": 0.9, **updates}


class FakeRecognition:
    def __init__(self):
        self.calls = []
        self.on_call = None
        self.vector = np.arange(1, 513, dtype=np.float32)

    def get(self, image, face):
        self.calls.append((image.shape, face.kps.tolist()))
        face.embedding = self.vector.copy()
        if self.on_call:
            self.on_call()


class FakeApp:
    def __init__(self):
        self.recognition = FakeRecognition()
        self.models = {"recognition": self.recognition}
        self.detection_calls = 0

    def get(self, image):
        self.detection_calls += 1
        face = SimpleNamespace(bbox=np.asarray(BOX, np.float32), kps=np.asarray(POINTS, np.float32), det_score=0.9)
        self.recognition.get(image, face)
        return [face]


@pytest.mark.parametrize("enabled,experimental", [(False, False), (True, False), (False, True), (1, True), (True, 1)])
def test_two_real_boolean_switches_required_before_reading_input(enabled, experimental):
    app = FakeApp()
    extractor = FeatureExtractor(app=app)
    with pytest.raises(InputValidationError, match="同时启用"):
        extractor.extract_selected_face("file-does-not-exist.png", {}, selection_context=CONTEXT,
                                        selected_face_enabled=enabled, selected_face_experimental=experimental)
    assert app.detection_calls == 0 and app.recognition.calls == []


@pytest.mark.parametrize("updates", [
    {"input_sha256": "0" * 64}, {"source_id": "other-input"}, {"frame_id": "other-frame"},
    {"image_size": [101, 100]}, {"image_size": [True, 100]}, {"coordinate_space": "crop_pixels"},
    {"bbox": [-1, 10, 90, 90]}, {"bbox": [10, 10, 10, 90]}, {"bbox": [10, 10, float("nan"), 90]},
    {"keypoints_5": [[1, 1]] * 5}, {"keypoints_5": POINTS[:4]}, {"detection_id": ""},
    {"detection_scope_id": None}, {"interface_version": "unknown"}, {"det_score": 2.0},
])
def test_wrong_input_source_coordinates_and_provenance_never_reach_recognition(tmp_path, updates):
    path = tmp_path / "synthetic.png"
    make_image(path)
    app = FakeApp()
    extractor = FeatureExtractor(app=app)
    with pytest.raises(InputValidationError):
        extractor.extract_selected_face(path, selection(path, **updates), selection_context=CONTEXT,
                                        selected_face_enabled=True, selected_face_experimental=True)
    assert app.detection_calls == 0 and app.recognition.calls == []


def test_selected_route_reuses_recognition_and_exact_normalization_without_detector(tmp_path):
    path = tmp_path / "synthetic.png"
    make_image(path)
    app = FakeApp()
    extractor = FeatureExtractor(app=app)
    legacy, _ = extractor.extract(path)
    calls = app.detection_calls
    explicit, face, metadata = extractor.extract_selected_face(path, selection(path), selection_context=CONTEXT,
                                                               selected_face_enabled=True, selected_face_experimental=True)
    assert np.array_equal(legacy, explicit)
    assert app.detection_calls == calls
    assert app.recognition.calls[-1] == ((100, 100, 3), POINTS)
    assert metadata["detection_performed"] is False
    assert metadata["equivalence_status"] == "not_validated_on_independent_data"
    assert metadata["query_cache_policy"] == "not_cached"
    assert face.kps.dtype == np.float32


def test_missing_independent_context_or_recognition_cannot_fall_back_to_detection(tmp_path):
    path = tmp_path / "synthetic.png"
    make_image(path)
    app = FakeApp()
    extractor = FeatureExtractor(app=app)
    with pytest.raises(InputValidationError):
        extractor.extract_selected_face(path, selection(path), selection_context=None,
                                        selected_face_enabled=True, selected_face_experimental=True)
    app.models = {}
    with pytest.raises(InputValidationError, match="recognition.get"):
        extractor.extract_selected_face(path, selection(path), selection_context=CONTEXT,
                                        selected_face_enabled=True, selected_face_experimental=True)
    assert app.detection_calls == 0 and app.recognition.calls == []


def test_searcher_rejects_orphan_flags_before_query_or_library_access():
    searcher = IdentitySearcher.__new__(IdentitySearcher)
    with pytest.raises(InputValidationError, match="连同选脸对象"):
        searcher.search("not-read.png", selected_face_enabled=True)
    with pytest.raises(InputValidationError, match="同时启用"):
        searcher.search("not-read.png", selected_face={}, selected_face_enabled=True)


def test_zero_or_invalid_embeddings_still_rejected_by_shared_normalization(tmp_path):
    path = tmp_path / "synthetic.png"
    make_image(path)
    app = FakeApp()
    app.recognition.vector = np.zeros(512, np.float32)
    extractor = FeatureExtractor(app=app)
    with pytest.raises(InvalidEmbeddingError):
        extractor.extract(path)
    with pytest.raises(InvalidEmbeddingError):
        extractor.extract_selected_face(path, selection(path), selection_context=CONTEXT,
                                        selected_face_enabled=True, selected_face_experimental=True)


def test_selection_cache_identity_binds_interface_input_source_frame_and_geometry(tmp_path):
    path = tmp_path / "synthetic.png"
    make_image(path)
    value = selection(path)
    def key(payload, context=CONTEXT):
        return validate_selected_face(payload, context=context, input_sha256=payload["input_sha256"],
                                      image_size=(100, 100))["selection_cache_key"]
    first = key(value)
    assert key(deepcopy(value)) == first
    assert key({**value, "bbox": [9, 9, 91, 91]}) != first
    assert key({**value, "input_sha256": "a" * 64}) != first
    assert key({**value, "frame_id": "frame-1"}, {**CONTEXT, "frame_id": "frame-1"}) != first
    assert key({**value, "source_id": "input-2"}, {**CONTEXT, "source_id": "input-2"}) != first
    assert key({**value, "detection_id": "other-observation"}) != first


def test_searcher_default_and_opt_in_keep_scores_with_fake_features_and_do_not_reuse_query_cache(tmp_path):
    path = tmp_path / "query.png"
    make_image(path)
    library = tmp_path / "library"
    make_image(library / "synthetic-entry" / "reference.png")
    app = FakeApp()
    searcher = IdentitySearcher(library, cache_dir=tmp_path / "cache", extractor=FeatureExtractor(app=app),
                                model_fingerprint="independent-fake-model")
    legacy = searcher.search(path)
    before = app.detection_calls
    explicit = searcher.search(path, selected_face=selection(path), selected_face_enabled=True,
                                selected_face_experimental=True, selection_context=CONTEXT)
    assert app.detection_calls == before
    assert explicit["candidates"] == legacy["candidates"]
    assert "query_extraction" not in legacy["method"]
    assert explicit["method"]["query_extraction"]["query_cache_policy"] == "not_cached"
    assert explicit["diagnostics"]["detected_faces"] == []
    assert explicit["diagnostics"]["provided_faces"]
    before = len(app.recognition.calls)
    searcher.search(path, selected_face=selection(path), selected_face_enabled=True,
                    selected_face_experimental=True, selection_context=CONTEXT)
    assert len(app.recognition.calls) == before + 1


def test_searcher_does_not_commit_result_if_file_changes_during_selected_inference(tmp_path):
    path = tmp_path / "query.png"
    make_image(path)
    library = tmp_path / "library"
    make_image(library / "synthetic-entry" / "reference.png")
    app = FakeApp()
    searcher = IdentitySearcher(library, cache_dir=tmp_path / "cache", extractor=FeatureExtractor(app=app),
                                model_fingerprint="independent-fake-model")
    payload = selection(path)
    app.recognition.on_call = lambda: path.write_bytes(path.read_bytes() + b"synthetic-change")
    with pytest.raises(InputValidationError, match="发生变化"):
        searcher.search(path, selected_face=payload, selected_face_enabled=True,
                        selected_face_experimental=True, selection_context=CONTEXT)


def test_mask_trace_binds_bytes_and_actual_detection_without_changing_pixels(tmp_path):
    path = tmp_path / "query.png"
    image = make_image(path)
    faces = [{"bbox": [10, 10, 45, 45], "keypoints_5": None}, {"bbox": [60, 60, 90, 90]}]
    trace = {}
    masked = write_main_face_query(path, faces, 0, tmp_path / "out", trace=trace)
    assert trace["original_input_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert trace["masked_input_sha256"] == hashlib.sha256(masked.read_bytes()).hexdigest()
    assert trace["image_size"] == [100, 100] and len(trace["mask_regions"]) == 1
    assert np.array_equal(cv2.imread(str(masked))[10:45, 10:45], image[10:45, 10:45])
    result = {"diagnostics": {"detected_faces": [{"bbox": [11, 10, 46, 45]}]}}
    bound = selection_trace(faces[0], result, trace)
    assert bound["correspondence_status"] == "geometry_changed_unverified"
    assert bound["second_detection"]["bbox"] == [11, 10, 46, 45]
    assert selection_trace(faces[0], {}, trace)["correspondence_status"] == "unrecorded"


def test_failed_redetection_keeps_trace_on_original_exception(tmp_path):
    path = tmp_path / "query.png"
    make_image(path)
    faces = [{"bbox": [10, 10, 45, 45]}, {"bbox": [60, 60, 90, 90]}]
    class FailingSearcher:
        def __init__(self):
            self.extractor = None
            self.count = 0
        def search(self, *args, **kwargs):
            self.count += 1
            if self.count == 1:
                raise MultipleFacesError("synthetic-multiple", diagnostics={"detected_faces": faces})
            raise NoFaceError("synthetic-redetection", diagnostics={"face_count": 0})
    with pytest.raises(NoFaceError) as captured:
        attempt_scale_chain(FailingSearcher(), str(path), (1280,), lambda scale: None, top_k=3,
                            multi_face_policy="largest_face", main_face_dir=tmp_path / "out")
    trace = captured.value.diagnostics["selection_trace"]
    assert trace["status"] == "failed" and trace["error_type"] == "NoFaceError"
    assert trace["original_input_sha256"] and trace["masked_input_sha256"]
