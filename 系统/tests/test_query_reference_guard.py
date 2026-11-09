"""Synthetic files only: no real test media, model inference, or network calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import orchestrator
import query_reference_guard


class ReachedInferenceBoundary(RuntimeError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source, identity = tmp_path / "source", tmp_path / "identity"
    source.mkdir()
    (identity / "person").mkdir(parents=True)
    (source / "reference.jpg").write_bytes(b"source reference")
    (identity / "person" / "reference.jpg").write_bytes(b"identity reference")
    query = tmp_path / "upload.jpg"
    query.write_bytes(b"query original")
    config = {
        "runtime": {"tasks_root": str(tmp_path / "tasks"), "identity_cache": str(tmp_path / "cache")},
        "deepfake": {}, "source": {"library": str(source)}, "identity": {"library": str(identity)},
        "frontend": {}, "test_only_policy": {"dataset_id": "synthetic_test_only"},
    }
    path = tmp_path / "config.json"

    def stop_before_inference(*args, **kwargs):
        raise ReachedInferenceBoundary

    monkeypatch.setattr(orchestrator, "_ProgressReporter", stop_before_inference)

    def run(*, selected=("source", "identity"), **kwargs):
        path.write_text(json.dumps(config), encoding="utf-8")
        return orchestrator.run_task(query, config_path=path, selected_modules=selected, **kwargs)

    return tmp_path, config, query, source, identity, run


def manifest(root: Path, config: dict, rows: list[dict]) -> Path:
    path = root / "queries_B.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    config["test_only_policy"]["query_manifest"] = path.name
    return path


def audit(root: Path) -> dict:
    tasks = list((root / "tasks").iterdir())
    assert len(tasks) == 1
    return json.loads((tasks[0] / "query_reference_guard.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind", ["source", "identity"])
def test_renamed_same_bytes_block_before_task_or_inference(setup, kind):
    root, config, query, source, identity, run = setup
    target = source / "renamed.png" if kind == "source" else identity / "person" / "renamed.png"
    target.write_bytes(query.read_bytes())
    with pytest.raises(orchestrator.OrchestrationError, match="查询自身的同字节副本"):
        run()
    assert not (root / "tasks").exists()


def test_override_is_checked_instead_of_configured_library(setup):
    root, config, query, source, identity, run = setup
    override = root / "selected_source"
    override.mkdir()
    (override / "copy.jpg").write_bytes(query.read_bytes())
    with pytest.raises(orchestrator.OrchestrationError, match="素材库包含"):
        run(selected=("source",), source_library=override)
    assert not (root / "tasks").exists()


def test_only_selected_retrieval_libraries_are_checked(setup):
    root, config, query, source, identity, run = setup
    (identity / "person" / "copy.jpg").write_bytes(query.read_bytes())
    with pytest.raises(ReachedInferenceBoundary):
        run(selected=("source",))
    assert list(audit(root)["libraries"]) == ["source"]


def test_registered_pair_version_blocks_without_same_bytes(setup):
    root, config, query, source, identity, run = setup
    manifest(root, config, [
        {"sha256": digest(query.read_bytes()), "pair_id": "same-query", "role": "query"},
        {"sha256": digest(b"compressed version"), "pair_id": "same-query", "role": "query"},
    ])
    (source / "compressed.jpg").write_bytes(b"compressed version")
    with pytest.raises(orchestrator.OrchestrationError, match="同一查询的已登记版本"):
        run()
    assert not (root / "tasks").exists()


def test_other_pairs_and_legitimate_source_ancestors_are_not_excluded(setup):
    root, config, query, source, identity, run = setup
    reference_sha = digest(b"source reference")
    manifest(root, config, [
        {"sha256": digest(query.read_bytes()), "pair_id": "query-a", "ancestor_sha256": [reference_sha]},
        {"sha256": reference_sha, "pair_id": "query-b"},
    ])
    with pytest.raises(ReachedInferenceBoundary):
        run()
    saved = audit(root)
    assert saved["matched_pair_ids"] == ["query-a"]
    assert saved["excluded_sha256"] == [digest(query.read_bytes())]
    assert saved["copy_verified"] is True
    assert saved["usage"] == "test_only" and saved["training_allowed"] is False
    assert saved["query_manifest"]["sha256"] == digest((root / "queries_B.json").read_bytes())


@pytest.mark.parametrize("contents", [b"not json", b"{}", b'{"rows":[]}',
    b'{"rows":[{"sha256":"bad","pair_id":"p"}]}',
    b'{"rows":[null]}',
    json.dumps({"rows": [{"sha256": "a" * 64, "pair_id": ""}]}).encode(),
    json.dumps({"rows": [{"sha256": "a" * 64, "pair_id": "p", "role": "reference"}]}).encode(),
])
def test_malformed_configured_manifest_fails_closed(setup, contents):
    root, config, query, source, identity, run = setup
    path = root / "queries_B.json"
    path.write_bytes(contents)
    config["test_only_policy"]["query_manifest"] = str(path)
    with pytest.raises(orchestrator.OrchestrationError, match="防自检索检查未通过"):
        run()
    assert not (root / "tasks").exists()


@pytest.mark.parametrize("value", ["missing.json", "", None, 42])
def test_unreadable_or_invalid_manifest_path_fails_closed(setup, value):
    root, config, query, source, identity, run = setup
    config["test_only_policy"]["query_manifest"] = value
    with pytest.raises(orchestrator.OrchestrationError, match="防自检索检查未通过"):
        run()
    assert not (root / "tasks").exists()


def test_unknown_query_still_checks_exact_bytes_with_manifest(setup):
    root, config, query, source, identity, run = setup
    manifest(root, config, [{"sha256": "a" * 64, "pair_id": "other-query"}])
    (source / "copy.jpg").write_bytes(query.read_bytes())
    with pytest.raises(orchestrator.OrchestrationError, match="查询自身的同字节副本"):
        run()


@pytest.mark.parametrize("mutation", ["copied_query", "reference", "manifest"])
def test_copy_window_mutations_stop_before_inference(setup, monkeypatch, mutation):
    root, config, query, source, identity, run = setup
    path = manifest(root, config, [{"sha256": digest(query.read_bytes()), "pair_id": "p"}])
    real_copy = orchestrator.shutil.copy2

    def changed_copy(src, dst, *args, **kwargs):
        result = real_copy(src, dst, *args, **kwargs)
        if mutation == "copied_query":
            Path(dst).write_bytes(b"different but nonoverlapping query")
        elif mutation == "reference":
            (source / "reference.jpg").write_bytes(b"different but nonoverlapping reference")
        else:
            path.write_text(json.dumps({"rows": [{"sha256": digest(query.read_bytes()), "pair_id": "changed"}]}))
        return result

    monkeypatch.setattr(orchestrator.shutil, "copy2", changed_copy)
    with pytest.raises(orchestrator.OrchestrationError, match="复制输入期间"):
        run()
    assert not list((root / "tasks").glob("*/query_reference_guard.json"))


def test_video_original_hash_checked_before_frame_selection(setup):
    root, config, query, source, identity, run = setup
    video = query.with_suffix(".mp4")
    query.rename(video)
    (source / "disguised.jpg").write_bytes(video.read_bytes())
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    with pytest.raises(orchestrator.OrchestrationError, match="查询自身的同字节副本"):
        orchestrator.run_task(video, config_path=config_path, selected_modules=("source",))
    assert not (root / "tasks").exists()


@pytest.mark.parametrize("policy", ["absent", "empty"])
def test_normal_product_does_not_hash_or_change_behavior(setup, monkeypatch, policy):
    root, config, query, source, identity, run = setup
    if policy == "absent":
        config.pop("test_only_policy")
    else:
        config["test_only_policy"] = {}
    (source / "copy.jpg").write_bytes(query.read_bytes())

    def forbidden(*args, **kwargs):
        pytest.fail("Ordinary tasks must not run the test-only hashing guard")

    monkeypatch.setattr(query_reference_guard, "_read_binding", forbidden)
    with pytest.raises(ReachedInferenceBoundary):
        run()
    assert not list((root / "tasks").glob("*/query_reference_guard.json"))
