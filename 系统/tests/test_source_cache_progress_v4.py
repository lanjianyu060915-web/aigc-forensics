"""Source-cache stages with fake cache entries and no reference media."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def load_tool():
    path = Path(__file__).resolve().parents[1] / "tools/build_source_cache.py"
    spec = importlib.util.spec_from_file_location("synthetic_source_cache_tool_v4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", [False, True])
def test_cache_progress_reports_actual_phase_and_failure(tmp_path, monkeypatch, failure):
    module = load_tool()
    phases = []
    original_write = module.atomic_write_json
    def record(path, value, **kwargs):
        phases.append(dict(value))
        return original_write(path, value, **kwargs)
    class Cache:
        stats = SimpleNamespace(as_dict=lambda: {"synthetic": True})
        def __init__(self, *_args):
            pass
        def load_or_build(self, masker, *, progress_callback):
            progress_callback(1, 2, Path("new-synthetic-a"), True)
            progress_callback(2, 2, Path("new-synthetic-b"), True)
            return [object(), object()], []
        def sift_features(self, entry, sift):
            if failure:
                raise RuntimeError("new synthetic feature failure")
    monkeypatch.setattr(module, "SourceFeatureCache", Cache)
    monkeypatch.setattr(module, "FaceMasker", lambda: object())
    monkeypatch.setattr(module.cv2, "SIFT_create", lambda **kwargs: object())
    monkeypatch.setattr(module, "atomic_write_json", record)
    progress = tmp_path / "progress.json"
    monkeypatch.setattr(sys, "argv", ["tool", "--library", str(tmp_path / "library"), "--cache-dir", str(tmp_path / "cache"),
                                      "--progress-path", str(progress)])
    if failure:
        with pytest.raises(RuntimeError, match="synthetic"):
            module.main()
    else:
        assert module.main() == 0
    saved = json.loads(progress.read_text("utf-8"))
    assert [row["stage"] for row in phases[:3]] == ["checking_library", "loading_face_mask", "global_features"]
    assert saved["stage"] == ("failed" if failure else "ready")
    assert saved["status"] == ("error" if failure else "success")
    assert saved["elapsed_seconds"] >= 0
    assert all(row["kind"] == "source_cache_build" for row in phases)
