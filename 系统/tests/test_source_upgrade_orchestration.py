"""用临时素材和假模型验证溯源升级的编排接线及失败隔离。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

import orchestrator
from case_libraries import case_source_cache
import source_upgrade


def _picture(path: Path, value: int = 80) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", np.full((24, 32, 3), value, dtype=np.uint8))
    assert ok
    encoded.tofile(path)
    return path


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch):
    source = tmp_path / "素材库"
    identity = tmp_path / "身份库"
    _picture(source / "底图.png")
    _picture(identity / "人员甲" / "参考.png")
    query = _picture(tmp_path / "查询.png")
    rule = tmp_path / "配对规则.json"
    rule.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config = {
        "runtime": {"tasks_root": str(tmp_path / "任务"), "identity_cache": str(tmp_path / "身份缓存")},
        "deepfake": {"enabled": True, "保留口径": {"边界": 0.6}},
        "identity": {"enabled": True, "library": str(identity), "model_root": str(tmp_path / "模型")},
        "source": {"enabled": True, "library": str(source),
                   "feature_cache": "{DATA_ROOT}/runtime/cache/source", "thresholds": {"high": 0.9}},
        "frontend": {},
    }
    seen: dict[str, list] = {"searches": [], "detections": [], "reports": []}
    real_search = orchestrator._run_search_module
    monkeypatch.setenv("FSTS_DATA_ROOT", str(tmp_path / "用户数据"))
    monkeypatch.setattr(orchestrator, "load_config", lambda path=None: (config, config_path))
    monkeypatch.setattr(orchestrator, "_run_file_provenance", lambda **kwargs: None)

    def search(**kwargs):
        seen["searches"].append({"name": kwargs["name"], "section": deepcopy(kwargs["section"]),
                                 "query": str(kwargs["query_path"]), "override": kwargs["library_override"]})
        candidates = ([{"rank": 1, "file_path": str(source / "底图.png"), "similarity": 0.8}]
                      if kwargs["name"] == "source" else
                      [{"rank": 1, "identity_id": "人员甲", "mean_similarity": 0.8}])
        payload = {"status": "success", "query_image": str(kwargs["query_path"]),
                   "candidates": candidates, "runtime_seconds": 0.01, "error_message": None}
        orchestrator._write_json(kwargs["output_path"], payload)
        return payload, {"status": "success", "runtime_seconds": 0.01, "error": None}

    def detection(**kwargs):
        seen["detections"].append(deepcopy(kwargs["section"]))
        payload = {"status": "success", "query_image": str(kwargs["query_path"]),
                   "is_deepfake": False, "score": 0.1, "label": "real",
                   "details": {"band": "low"}, "runtime_seconds": 0.01, "error_message": None}
        orchestrator._write_json(kwargs["output_path"], payload)
        return payload, {"status": "success", "runtime_seconds": 0.01, "error": None}

    def reports(task_dir, **kwargs):
        seen["reports"].append(deepcopy(kwargs))
        orchestrator._write_json(task_dir / "final_report.json", kwargs["unified_result"])
        return {"file_names": {"json": "final_report.json"}}

    monkeypatch.setattr(orchestrator, "_run_search_module", search)
    monkeypatch.setattr(orchestrator, "_run_deepfake_module", detection)
    monkeypatch.setattr(orchestrator, "generate_task_reports", reports)
    return {"root": tmp_path, "source": source, "identity": identity, "query": query,
            "rule": rule, "config": config, "config_path": config_path, "seen": seen,
            "real_search": real_search}


def _enable_pair(workspace):
    workspace["config"]["pair_compare"] = {
        "enabled": True, "rule_path": str(workspace["rule"]),
        "expected_rule_sha256": hashlib.sha256(workspace["rule"].read_bytes()).hexdigest(),
    }


def _read(task: dict, filename: str) -> dict:
    return json.loads((Path(task["task_dir"]) / filename).read_text(encoding="utf-8"))


def _pair_with_artifacts(**kwargs):
    directory = Path(kwargs["task_dir"]) / "pair_compare" / "candidate_1"
    artifact = _picture(directory / "query.png")
    return {
        "status": "success", "mode": "manual_review", "conclusion_enabled": False,
        "scope": kwargs["scope"], "timestamp_seconds": kwargs["timestamp_seconds"],
        "candidates": [{"rank": 1, "status": "success", "comparison": {"alignment_status": "completed"}}],
        "artifact_manifest": [
            {"path": artifact.relative_to(kwargs["task_dir"]).as_posix(), "kind": "pair_compare",
             "sha256": "0" * 64},
            {"path": "../外部.png", "kind": "不安全路径"},
        ],
    }


def test_old_configuration_keeps_modules_and_disables_optional_pair(workspace, monkeypatch):
    pair = Mock(side_effect=AssertionError("旧配置不能执行配对"))
    monkeypatch.setattr(orchestrator, "run_source_pair", pair)
    original = deepcopy(workspace["config"])
    task = orchestrator.run_task(workspace["query"])
    pair.assert_not_called()
    assert task["status"] == "success"
    assert task["pair_compare"] is None and task["case_libraries"] is None
    assert task["forgery_type"] is None
    assert not (Path(task["task_dir"]) / "pair_compare_result.json").exists()
    assert workspace["config"] == original
    assert workspace["seen"]["detections"] == [original["deepfake"]]
    assert _read(task, "detection_result.json")["score"] == 0.1


def test_case_context_only_changes_source_copy_and_reaches_reports(workspace):
    workspace["config"]["case_libraries"] = {"enabled": True}
    original = deepcopy(workspace["config"])
    task = orchestrator.run_task(workspace["query"])
    metadata = task["case_libraries"]
    assert metadata["status"] == "success"
    source_call = next(row for row in workspace["seen"]["searches"] if row["name"] == "source")
    identity_call = next(row for row in workspace["seen"]["searches"] if row["name"] == "identity")
    expected = case_source_cache(
        orchestrator.resolve_path(original["source"]["feature_cache"], workspace["config_path"]),
        workspace["source"], metadata["libraries"]["source"]["manifest_sha256"])
    assert Path(source_call["section"]["feature_cache"]) == expected
    assert source_call["section"]["thresholds"] == original["source"]["thresholds"]
    assert identity_call["section"] == original["identity"]
    assert workspace["seen"]["detections"] == [original["deepfake"]]
    assert workspace["config"] == original
    assert _read(task, "unified_result.json")["case_libraries"] == metadata
    assert workspace["seen"]["reports"][0]["unified_result"]["case_libraries"] == metadata


def test_pair_result_and_evidence_join_manifest_without_changed_source(workspace, monkeypatch):
    _enable_pair(workspace)
    compare = Mock(side_effect=_pair_with_artifacts)
    monkeypatch.setattr(source_upgrade, "run_pair_comparison", compare)
    task = orchestrator.run_task(workspace["query"], selected_modules={"source"})
    pair = task["pair_compare"]
    assert task["status"] == "success"
    assert task["source"]["status"] == "success"
    assert pair["conclusion_enabled"] is False and pair["forgery_type"] is None
    assert _read(task, "unified_result.json")["pair_compare"] == pair
    assert _read(task, "pair_compare_result.json") == pair
    manifest = _read(task, "artifact_manifest.json")
    paths = {row["path"] for row in manifest["artifacts"]}
    assert {"pair_compare_result.json", "pair_compare/candidate_1/query.png"} <= paths
    assert "../外部.png" not in paths
    evidence = next(row for row in manifest["artifacts"] if row["path"].endswith("candidate_1/query.png"))
    assert evidence["sha256"] == hashlib.sha256((Path(task["task_dir"]) / evidence["path"]).read_bytes()).hexdigest()
    source = _read(task, "source_result.json")
    assert source["status"] == "success" and "pair_compare" not in source
    assert compare.call_args.kwargs["scope"] == "image"


@pytest.mark.parametrize("outer_failure", [False, True])
def test_pair_failures_do_not_change_detection_source_or_task_status(workspace, monkeypatch, outer_failure):
    _enable_pair(workspace)
    target = orchestrator if outer_failure else source_upgrade
    name = "run_source_pair" if outer_failure else "run_pair_comparison"
    monkeypatch.setattr(target, name, Mock(side_effect=RuntimeError("模拟配对故障")))
    task = orchestrator.run_task(workspace["query"], selected_modules={"deepfake", "source"})
    assert task["status"] == "success"
    assert task["errors"] == []
    assert task["source"]["status"] == task["deepfake"]["status"] == "success"
    assert task["pair_compare"]["status"] == "error"
    assert "模拟配对故障" in task["pair_compare"]["error_message"]
    assert _read(task, "detection_result.json")["details"]["band"] == "low"
    assert _read(task, "pair_compare_result.json")["status"] == "error"


def test_snapshot_exception_degrades_metadata_only(workspace, monkeypatch):
    workspace["config"]["case_libraries"] = {"enabled": True}
    monkeypatch.setattr(orchestrator, "prepare_case_context", Mock(side_effect=OSError("模拟清单故障")))
    task = orchestrator.run_task(workspace["query"], selected_modules={"deepfake", "source"})
    assert task["status"] == "success" and task["errors"] == []
    assert task["case_libraries"]["status"] == "error"
    source_call = next(row for row in workspace["seen"]["searches"] if row["name"] == "source")
    assert source_call["section"]["feature_cache"] is None
    assert workspace["config"]["source"]["feature_cache"] == "{DATA_ROOT}/runtime/cache/source"


def test_video_pair_is_primary_frame_without_changing_existing_frame_count(workspace, monkeypatch):
    _enable_pair(workspace)
    workspace["config"]["video"] = {"source_frames": 2}
    video = workspace["root"] / "查询.mp4"
    video.write_bytes(b"temporary video fixture")

    def selection(**kwargs):
        frames = []
        for rank, timestamp in ((1, 2.5), (2, 7.0)):
            name = f"frame_{rank:03d}.jpg"
            path = _picture(kwargs["task_dir"] / "key_frames" / name, rank * 30)
            frames.append({"frame_rank": rank, "file_name": name, "path": name,
                           "timestamp_seconds": timestamp, "quality_score": 0.9,
                           "face_count": 1, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        result = {"status": "success", "key_frames": frames, "video_sha256": "a" * 64,
                  "video": {"width": 32, "height": 24, "fps": 24, "duration_seconds": 10}}
        orchestrator._write_json(kwargs["task_dir"] / "frame_selection.json", result)
        return result, {"status": "success", "runtime_seconds": 0.01}

    monkeypatch.setattr(orchestrator, "_run_frame_selection", selection)
    compare = Mock(side_effect=_pair_with_artifacts)
    monkeypatch.setattr(source_upgrade, "run_pair_comparison", compare)
    task = orchestrator.run_task(video, selected_modules={"source"})
    assert task["status"] == "success"
    assert len(workspace["seen"]["searches"]) == 2
    assert workspace["config"]["video"]["source_frames"] == 2
    assert compare.call_count == 1
    assert compare.call_args.kwargs["scope"] == "frame"
    assert compare.call_args.kwargs["timestamp_seconds"] == 2.5
    assert Path(compare.call_args.kwargs["query_path"]).name == "query.jpg"
    assert task["pair_compare"]["scope"] == "frame"
    assert task["pair_compare"]["timestamp_seconds"] == 2.5


def test_pair_is_not_run_when_source_is_unselected(workspace, monkeypatch):
    _enable_pair(workspace)
    pair = Mock(side_effect=AssertionError("未选择素材模块时不能执行配对"))
    monkeypatch.setattr(orchestrator, "run_source_pair", pair)
    task = orchestrator.run_task(workspace["query"], selected_modules={"identity"})
    pair.assert_not_called()
    assert task["pair_compare"] is None


def test_boundary_preflight_cannot_be_bypassed_by_metadata_failure(workspace, monkeypatch):
    workspace["config"]["case_libraries"] = {"enabled": True}
    outside = workspace["root"] / "库外"
    _picture(outside / "外部.png")
    link = workspace["source"] / "越界链接"
    if os.name == "nt":
        quoted_link = str(link).replace("'", "''")
        quoted_outside = str(outside).replace("'", "''")
        command = f"New-Item -ItemType Junction -Path '{quoted_link}' -Target '{quoted_outside}' | Out-Null"
        completed = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True)
        if completed.returncode:
            pytest.skip("当前环境不能创建目录联接")
    else:
        link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(orchestrator, "_run_search_module", workspace["real_search"])
    worker = Mock(side_effect=AssertionError("越界库不得进入模型进程"))
    monkeypatch.setattr(orchestrator, "_run_subprocess", worker)
    try:
        task = orchestrator.run_task(workspace["query"], selected_modules={"deepfake", "source"})
        assert task["case_libraries"]["status"] == "error"
        assert task["status"] == "partial_success"
        assert task["deepfake"]["status"] == "success"
        assert task["source"]["status"] == "error"
        assert "越出" in _read(task, "source_result.json")["error_message"]
        worker.assert_not_called()
    finally:
        if os.name == "nt":
            os.rmdir(link)
        else:
            link.unlink()


def test_identity_preflight_accepts_tiff_before_fake_service(workspace, monkeypatch):
    identity = workspace["root"] / "含扫描图的身份库"
    _picture(identity / "人员乙" / "照片.tiff")
    monkeypatch.setattr(orchestrator, "_run_search_module", workspace["real_search"])
    calls = []

    def identity_service(**kwargs):
        calls.append(kwargs)
        return {"status": "success", "query_image": str(kwargs["query_path"]),
                "candidates": [{"rank": 1, "identity_id": "人员乙", "mean_similarity": 0.8}],
                "runtime_seconds": 0.01, "error_message": None}

    task = orchestrator.run_task(workspace["query"], selected_modules={"identity"},
                                 identity_library=identity, services={"run_identity": identity_service})
    assert task["status"] == "success"
    assert task["identity"]["execution_mode"] == "service"
    assert len(calls) == 1 and Path(calls[0]["library_path"]) == identity


def test_empty_library_override_does_not_fall_back_to_a_different_library(workspace, monkeypatch):
    monkeypatch.setattr(orchestrator, "_run_search_module", workspace["real_search"])
    worker = Mock(side_effect=AssertionError("空库路径不得进入模型进程"))
    monkeypatch.setattr(orchestrator, "_run_subprocess", worker)
    task = orchestrator.run_task(workspace["query"], selected_modules={"source"}, source_library="  ")
    assert task["source"]["status"] == "error"
    assert "未配置或为空" in _read(task, "source_result.json")["error_message"]
    worker.assert_not_called()
