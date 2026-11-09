"""纯合成参考入库回归：不读取测试资料，不加载模型或发送网络请求。"""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "modules/identity_search", ROOT / "modules/identity_search_v2"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from identity_search.errors import (NoFaceError, MultipleFacesError, ImageDecodeError,
                                    InferenceError, InvalidEmbeddingError, InputValidationError)
from identity_search.reference_detection import (extract_reference, inspect_reference,
                                                 reference_policy)
from identity_search.searcher import IdentitySearcher


class FakeExtractor:
    """按文件名与固定尺度返回预设结果，完全不做图像推理。"""
    def __init__(self, outcomes=None):
        self.det_size = (1280, 1280)
        self.app = SimpleNamespace(det_size=self.det_size, det_model=SimpleNamespace(
            input_size=self.det_size, input_sizes=[self.det_size], static_input_size=None))
        self.outcomes = outcomes or {}
        self.calls = []

    def extract(self, path):
        scale = self.app.det_model.input_size[0]
        key = (Path(path).name, scale)
        self.calls.append(key)
        value = self.outcomes.get(key, self.outcomes.get(scale, 1))
        if isinstance(value, Exception):
            raise value
        if value == 0:
            raise NoFaceError("合成无脸", diagnostics={"face_count": 0})
        if value > 1:
            raise MultipleFacesError("合成多脸", diagnostics={"face_count": value})
        return np.array([1., 0.], dtype=np.float32), SimpleNamespace(
            bbox=np.array([0., 0., 20., 20.]), kps=np.zeros((5, 2)), det_score=.9)


def state(extractor):
    return copy.deepcopy((vars(extractor.app.det_model), extractor.det_size,
                          extractor.app.det_size))


@pytest.mark.parametrize("outcomes,expected", [({1280: 1}, [1280]),
    ({1280: 0, 640: 1}, [1280, 640]), ({1280: 0, 640: 0, 320: 1}, [1280, 640, 320])])
def test_first_single_face_stops_and_restores(outcomes, expected):
    extractor = FakeExtractor(outcomes)
    before = state(extractor)
    _, _, diagnostic = extract_reference(extractor, "reference.jpg", (1280, 640, 320))
    assert [s for _, s in extractor.calls] == expected
    assert diagnostic["detection_scale_used"] == [expected[-1]] * 2
    assert state(extractor) == before
    extract_reference(extractor, "next.jpg", (1280, 640, 320))
    assert extractor.calls[len(expected)][1] == 1280


@pytest.mark.parametrize("outcomes,expected", [({1280: 2}, [1280]),
    ({1280: 0, 640: 2, 320: 1}, [1280, 640])])
def test_multiple_faces_never_become_single_by_further_fallback(outcomes, expected):
    extractor = FakeExtractor(outcomes)
    before = state(extractor)
    with pytest.raises(MultipleFacesError) as caught:
        extract_reference(extractor, "reference.jpg", (1280, 640, 320))
    assert [s for _, s in extractor.calls] == expected
    assert caught.value.diagnostics["reference_detection"]["attempts"][-1]["face_count"] == 2
    assert state(extractor) == before


@pytest.mark.parametrize("error", [ImageDecodeError("解码失败"), InferenceError("推理失败"),
    InvalidEmbeddingError("无效特征", diagnostics={"face_count": 1}), RuntimeError("意外失败")])
def test_errors_are_not_retried_and_restore_state(error):
    extractor = FakeExtractor({1280: 0, 640: error})
    before = state(extractor)
    with pytest.raises(type(error)):
        extract_reference(extractor, "reference.jpg", (1280, 640, 320))
    assert [s for _, s in extractor.calls] == [1280, 640]
    assert state(extractor) == before


def test_all_no_face_has_complete_attempts():
    extractor = FakeExtractor({1280: 0, 640: 0, 320: 0})
    with pytest.raises(NoFaceError) as caught:
        extract_reference(extractor, "reference.jpg", (1280, 640, 320))
    diagnostic = caught.value.diagnostics["reference_detection"]
    assert len(diagnostic["attempts"]) == 3
    assert diagnostic["detection_scale_used"] is None


def test_static_model_is_not_retried_or_relabelled_as_new_scale():
    extractor = FakeExtractor({1280: 0})
    extractor.app.det_model.static_input_size = (1280, 1280)
    before = state(extractor)
    with pytest.raises(NoFaceError):
        extract_reference(extractor, "reference.jpg", (1280, 640, 320))
    assert len(extractor.calls) == 1
    assert state(extractor) == before


@pytest.mark.parametrize("chain", [(), (0,), (True,), (1280, 1280), (640.0,), ("640",)])
def test_invalid_chains_rejected(chain):
    with pytest.raises(InputValidationError):
        reference_policy(chain)


@pytest.mark.parametrize("outcomes,count,usable", [({1280: 0, 640: 1}, 1, True),
    ({1280: 0, 640: 2}, 2, False), ({1280: 0, 640: 0, 320: 0}, 0, False),
    ({1280: InvalidEmbeddingError("合成无效", diagnostics={"face_count": 1})}, 1, False)])
def test_quality_check_shares_extraction_contract(outcomes, count, usable):
    extractor = FakeExtractor(outcomes)
    result = inspect_reference(extractor, "reference.jpg", (1280, 640, 320))
    assert result["face_count"] == count
    assert (not result["issues"]) is usable
    assert result["reference_detection"]["scale_chain"] == [1280, 640, 320]


def make_searcher(tmp_path, extractor, scales=None):
    library = tmp_path / "library"
    for name in ("needs_fallback.jpg", "single.jpg"):
        path = library / "person" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(name.encode())
    return IdentitySearcher(library, cache_dir=tmp_path / "cache", extractor=extractor,
                            model_fingerprint="synthetic-model", reference_scales=scales)


def test_policy_cache_is_separate_preserves_old_skips_and_reuses_new_records(tmp_path):
    behavior = {("needs_fallback.jpg", 1280): 0}
    legacy = make_searcher(tmp_path, FakeExtractor(behavior))
    old_paths = legacy.get_cache_paths()
    old_bytes = [p.read_bytes() for p in old_paths]
    assert len(legacy.identities["person"]["features"]) == 1
    repaired = make_searcher(tmp_path, FakeExtractor(behavior), (1280, 640, 320))
    assert len(repaired.identities["person"]["features"]) == 2
    assert repaired._fingerprint != legacy._fingerprint
    assert repaired.get_cache_paths() != old_paths
    assert [p.read_bytes() for p in old_paths] == old_bytes
    no_calls = FakeExtractor({1280: RuntimeError("缓存命中不得推理")})
    restored = make_searcher(tmp_path, no_calls, (1280, 640, 320))
    assert not no_calls.calls
    assert restored.diagnostics["reference_detection"] == repaired.diagnostics["reference_detection"]
    metadata = json.loads(restored.get_cache_paths()[0].read_text(encoding="utf-8"))
    assert metadata["reference_policy"] == reference_policy((1280, 640, 320))
    changed = make_searcher(tmp_path, FakeExtractor(behavior), (1280, 320))
    assert changed._fingerprint != repaired._fingerprint
    assert changed.get_cache_paths() != repaired.get_cache_paths()


def test_reference_helper_does_not_change_query_extract_path(tmp_path):
    extractor = FakeExtractor({("needs_fallback.jpg", 1280): 0, ("query.jpg", 1280): 0})
    searcher = make_searcher(tmp_path, extractor, (1280, 640, 320))
    query = tmp_path / "query.jpg"
    query.write_bytes(b"synthetic-query")
    extractor.calls.clear()
    with pytest.raises(NoFaceError):
        searcher.search(query)
    assert extractor.calls == [("query.jpg", 1280)]


def test_quality_entry_uses_full_config_chain(monkeypatch, tmp_path):
    import cv2
    import case_libraries as module
    for target in (tmp_path / "source/image.png", tmp_path / "identity/person/image.png"):
        target.parent.mkdir(parents=True)
        ok, encoded = cv2.imencode(".png", np.zeros((10, 10, 3), dtype=np.uint8))
        assert ok
        encoded.tofile(target)
    extractor = FakeExtractor({960: 0, 640: 1})
    loaded = []
    def prepare(model, scale):
        loaded.append(scale)
        return None, extractor
    monkeypatch.setattr(module, "_prepare_identity", prepare)
    result = module.check_case_libraries(source_root=tmp_path / "source",
        identity_root=tmp_path / "identity", product_root=ROOT, model_root=tmp_path,
        identity_scales=(960, 640))
    assert loaded == [960]
    assert result["identity"]["usable_count"] == 1
    diagnostic = result["identity"]["files"][0]["reference_detection"]
    assert diagnostic["scale_chain"] == [960, 640]
    assert [a["scale"] for a in diagnostic["attempts"]] == [960, 640]


def test_persistent_first_and_switched_library_share_reference_policy(monkeypatch, tmp_path):
    import persistent_worker
    import persistent_worker_v2
    calls = []
    extractor = FakeExtractor()
    def new_searcher(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(extractor=extractor)
    monkeypatch.setattr(persistent_worker, "check_required_dependencies", lambda: None)
    monkeypatch.setattr(persistent_worker, "preflight_model", lambda *a, **k:
                        SimpleNamespace(root=tmp_path, fingerprint="synthetic-model"))
    monkeypatch.setattr(persistent_worker, "resolve_device", lambda *a: ("cpu", -1))
    monkeypatch.setattr(persistent_worker, "_new_searcher", new_searcher)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    args = SimpleNamespace(cache_dir=str(tmp_path / "cache"), library=str(first),
                           device="cpu", ctx=-1, model_root=str(tmp_path))
    runtime = persistent_worker_v2.ScaleFallbackIdentityRuntime(args, (960, 640))
    runtime._searcher_for(second)
    assert len(calls) == 2
    assert all(c["reference_scales"] == (960, 640) and c["det_size"] == (960, 960) for c in calls)
    assert persistent_worker.PersistentIdentityRuntime._reference_searcher_options(runtime) == {}


@pytest.mark.parametrize("prewarm,mutate_source", [(None, False), (True, False),
                                                   (False, False), (False, True)])
def test_build_entry_quality_and_index_use_identical_policy(monkeypatch, tmp_path,
                                                           prewarm, mutate_source):
    import cv2
    import case_libraries as module
    from source_search import face_mask, feature_cache
    source, identity = tmp_path / "source", tmp_path / "identity"
    for target in (source / "image.png", identity / "person/needs_fallback.png"):
        target.parent.mkdir(parents=True)
        ok, encoded = cv2.imencode(".png", np.zeros((10, 10, 3), dtype=np.uint8))
        assert ok
        encoded.tofile(target)
    extractor = FakeExtractor({1280: 0, 640: 1})
    monkeypatch.setattr(module, "_prepare_identity", lambda *a:
                        (SimpleNamespace(root=tmp_path, fingerprint="synthetic-model"), extractor))
    calls = {"global": 0, "sift_create": 0, "sift_features": 0}
    class FakeSourceCache:
        def __init__(self, *a):
            pass
        def load_or_build(self, *a, **kw):
            calls["global"] += 1
            if mutate_source:
                (source / "image.png").write_bytes(b"changed during indexing")
            return [SimpleNamespace(path=source / "image.png")], []
        def sift_features(self, *a):
            calls["sift_features"] += 1
    def create_sift(**kw):
        assert kw == {"nfeatures": 2500}
        calls["sift_create"] += 1
        return object()
    monkeypatch.setattr(feature_cache, "SourceFeatureCache", FakeSourceCache)
    monkeypatch.setattr(face_mask, "FaceMasker", lambda: None)
    monkeypatch.setattr(cv2, "SIFT_create", create_sift)
    kwargs = dict(name="合成库", source_root=source,
        identity_root=identity, product_root=ROOT, model_root=tmp_path,
        cache_root=tmp_path / "records", source_cache=tmp_path / "source_cache",
        identity_cache=tmp_path / "identity_cache")
    if prewarm is not None:
        kwargs["prewarm_sift"] = prewarm
    if mutate_source:
        with pytest.raises(module.LibraryValidationError, match="文件发生变化"):
            module.build_case_libraries(**kwargs)
        assert calls == {"global": 1, "sift_create": 0, "sift_features": 0}
        return
    result = module.build_case_libraries(**kwargs)
    expected_prewarm = prewarm is not False
    assert calls == {"global": 1, "sift_create": int(expected_prewarm),
                     "sift_features": int(expected_prewarm)}
    assert result["source_sift_prewarmed"] is expected_prewarm
    assert result["source_indexed_count"] == result["source"]["usable_count"] == 1
    assert result["source_skipped_count"] == 0
    assert result["identity"]["usable_count"] == result["identity_indexed_count"] == 1
    quality = result["identity"]["files"][0]["reference_detection"]
    metadata = json.loads(next((tmp_path / "identity_cache").glob("*.json")).read_text(encoding="utf-8"))
    assert quality == metadata["reference_detection"]["person/needs_fallback.png"]
    assert extractor.calls == [("needs_fallback.png", 1280), ("needs_fallback.png", 640)] * 2


def test_cli_build_receives_full_policy_before_query(monkeypatch, tmp_path):
    import run_identity_search_v2 as cli
    from identity_search.errors import EmptyLibraryError
    calls = []
    query, library = tmp_path / "query.jpg", tmp_path / "library"
    query.write_bytes(b"synthetic-query")
    library.mkdir()
    monkeypatch.setattr(cli, "create_progress_writer", lambda **kw: None)
    monkeypatch.setattr(cli, "check_required_dependencies", lambda: None)
    monkeypatch.setattr(cli, "resolve_device", lambda *a: ("cpu", -1))
    monkeypatch.setattr(cli, "preflight_model", lambda *a, **kw:
                        SimpleNamespace(root=tmp_path, fingerprint="synthetic-model"))
    monkeypatch.setattr(cli, "FeatureExtractor", lambda **kw: FakeExtractor())
    def searcher(**kw):
        calls.append(kw)
        raise EmptyLibraryError("合成终止，未发起查询")
    monkeypatch.setattr(cli, "IdentitySearcher", searcher)
    result = cli.main(["--query", str(query), "--library", str(library),
        "--output", str(tmp_path / "result.json"), "--cache-dir", str(tmp_path / "cache"),
        "--model-root", str(tmp_path), "--det-size-chain", "960,640"])
    assert result == 1
    assert calls[0]["reference_scales"] == (960, 640)
    assert calls[0]["det_size"] == (960, 960)


def test_reference_evidence_reextraction_restores_current_query_scale(tmp_path):
    from identity_search.reference_detection import apply_detection_scale
    extractor = FakeExtractor({("needs_fallback.jpg", 1280): 0})
    searcher = make_searcher(tmp_path, extractor, (1280, 640, 320))
    apply_detection_scale(extractor, 320, -1)
    before = state(extractor)
    extractor.calls.clear()
    searcher._extract_reference(searcher.library_path / "person/needs_fallback.jpg")
    assert extractor.calls == [("needs_fallback.jpg", 1280), ("needs_fallback.jpg", 640)]
    assert state(extractor) == before
