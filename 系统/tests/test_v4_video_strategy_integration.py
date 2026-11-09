"""Fresh metadata fixtures and a fake subprocess only; no media or models.

The two selection helpers are compiled from their actual source AST to avoid
importing unrelated product services. The subprocess writes marker JSON, never
executes a decoder. Aggregate fixtures are hand-constructed numeric contracts.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import pytest

from video_pipeline.aggregate import aggregate_deepfake_frames, build_video_summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def selection_harness(tmp_path):
    source = (ROOT / "orchestrator.py").read_text("utf-8")
    tree = ast.parse(source)
    names = {"_run_frame_selection", "_video_positive_int"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == 2
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    calls = []
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), "utf-8")
    def run(command, **kwargs):
        calls.append((command, kwargs))
        destination = Path(command[command.index("--selection") + 1])
        save(destination, {"status": "success", "key_frames": [{"frame_index": 1}],
                           "fixture_origin": "independently_constructed_metadata"})
        return SimpleNamespace(returncode=0, stdout="synthetic", stderr="")
    def text(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(value), "utf-8")
    namespace = {"Mapping": Mapping, "Path": Path, "time": time, "math": math,
                 "json": json, "subprocess": subprocess, "_now_iso": lambda: "synthetic-time",
                 "resolve_path": lambda path, config: Path(path), "_run_subprocess": run,
                 "_write_json": save, "_write_text": text}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / "orchestrator.py"), "exec"), namespace)
    for name in ("fake-python", "fake-runner", "fake-video"):
        (tmp_path / name).write_text("synthetic nonmedia marker", "utf-8")
    section = {"python": str(tmp_path / "fake-python"), "runner": str(tmp_path / "fake-runner")}
    progress = SimpleNamespace(task_id="synthetic-task", events_path=tmp_path / "events", latest_path=tmp_path / "latest")
    def invoke(extra=None, profile=None):
        return namespace["_run_frame_selection"](
            section={**section, **(extra or {})}, execution_profile=profile,
            config_path=tmp_path / "config.json", video_path=tmp_path / "fake-video",
            task_dir=tmp_path / "task", logs_dir=tmp_path / "logs", progress=progress)
    return invoke, calls, tmp_path


def profile(purpose="model_selection", **updates):
    return {"schema_version": "execution_profile.independent.v1", "mode": "independent_inference",
            "query_admission": {"purpose": purpose}, **updates}


def candidate():
    return {"selection_strategy": "segmented_time_v1", "allow_experimental_selection": True}


def test_selection_default_passes_legacy_and_keeps_original_budget(selection_harness):
    invoke, calls, _ = selection_harness
    payload, _ = invoke()
    assert payload["status"] == "success" and len(calls) == 1
    argv = calls[0][0]
    for flag, value in (("--selection-strategy", "legacy_quality_v1"), ("--candidate-frames", "24"),
                        ("--max-key-frames", "5"), ("--min-sharpness-ratio", "0.35")):
        assert argv[argv.index(flag) + 1] == value
    assert "--allow-experimental-selection" not in argv


def test_selection_candidate_is_wired_and_logged_only_for_independent_model_selection(selection_harness):
    invoke, calls, root = selection_harness
    payload, _ = invoke(candidate(), profile())
    assert payload["status"] == "success" and len(calls) == 1
    argv = calls[0][0]
    assert argv[argv.index("--selection-strategy") + 1] == "segmented_time_v1"
    assert "--allow-experimental-selection" in argv
    assert json.loads((root / "logs/video_command.json").read_text("utf-8"))["argv"] == argv


@pytest.mark.parametrize("extra,context", [
    ({"selection_strategy": "segmented_time_v1"}, profile()),
    (candidate(), None), (candidate(), profile("acceptance")),
    (candidate(), profile("calibration")), (candidate(), profile("prompt_tuning")),
    (candidate(), profile(mode="engineering_only")),
    (candidate(), profile(schema_version="unsupported")),
    ({**candidate(), "allow_experimental_selection": "true"}, profile()),
    ({**candidate(), "allow_experimental_selection": 1}, profile()),
    ({"selection_strategy": "legacy_quality_v1", "allow_experimental_selection": True}, profile()),
    ({"selection_strategy": "unknown"}, profile()),
])
def test_selection_rejects_unauthorized_or_ambiguous_candidate_before_subprocess(selection_harness, extra, context):
    invoke, calls, _ = selection_harness
    payload, metadata = invoke(extra, context)
    assert payload["status"] == metadata["status"] == "error"
    assert not calls


def test_run_task_passes_its_same_execution_profile_to_selection():
    tree = ast.parse((ROOT / "orchestrator.py").read_text("utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_run_frame_selection"]
    assert len(calls) == 1
    value = next(kw.value for kw in calls[0].keywords if kw.arg == "execution_profile")
    assert ast.unparse(value) == "config.get('execution_profile')"


def frame(index, *, model="a" * 64, version="synthetic-v1", frame_threshold=.3, video_threshold=.3):
    return {"frame": {"frame_id": f"frame-{index}", "frame_index": index},
            "payload": {"status": "success", "score": .2 if index == 0 else .8,
                        "details": {"band": "low" if index == 0 else "high", "branch": "synthetic-branch",
                                    "band_thresholds": {"low_max": frame_threshold, "medium_max": .6},
                                    "video_band_thresholds": {"low_max": video_threshold, "medium_max": .6}},
                        "provenance": {"model_sha256": model, "model_version": version}}}


def aggregate(rows):
    return aggregate_deepfake_frames(rows, query_image="independent-marker.jpg",
                                    video_band_thresholds={"low_max": .3, "medium_max": .6})


def test_aggregate_conflicts_record_every_reported_value_without_changing_numeric_policy():
    baseline = aggregate([frame(0), frame(1)])
    altered = aggregate([frame(0), frame(1, model="b" * 64, version="synthetic-v2", frame_threshold=.4, video_threshold=.4)])
    for key in ("score", "label", "status", "is_deepfake"):
        assert baseline[key] == altered[key]
    for key in ("band", "band_thresholds", "video_band_thresholds", "video_band_rule", "score_statistics"):
        assert baseline["details"][key] == altered["details"][key]
    audit = altered["details"]["aggregation_consistency"]
    assert audit["status"] == "conflict" and audit["numerical_policy_unchanged"] is True
    assert set(audit["conflict_fields"]) == {"frame_band_thresholds", "video_band_thresholds", "model_sha256", "model_version"}
    assert audit["selected_video_thresholds_disagree_with_reported"] is True
    for value in audit["fields"].values():
        assert [row["frame_ids"] for row in value["variants"]] == [["frame-0"], ["frame-1"]]
    assert baseline["details"]["aggregation_consistency"]["status"] == "consistent_reported"
    summary = build_video_summary(selection={"key_frames": []}, module_payloads={"deepfake": altered}, input_file="new-synthetic.mp4")
    assert summary["modules"]["deepfake"]["aggregation_consistency"] == audit


def test_aggregate_missing_identity_is_unknown_and_failed_frame_is_not_discarded():
    item = frame(0)
    item["payload"].pop("provenance")
    failure = {"frame": {"frame_id": "failed-frame"}, "payload": None, "error": "synthetic timeout"}
    value = aggregate([item, failure])
    audit = value["details"]["aggregation_consistency"]
    assert value["score"] == .2 and value["status"] == "success"
    assert audit["status"] == "incomplete" and audit["conflict_fields"] == []
    assert audit["fields"]["model_sha256"]["missing_frame_ids"] == ["frame-0", "failed-frame"]
    assert [row["included_in_score"] for row in audit["frame_records"]] == [True, False]
    assert aggregate([failure])["details"]["aggregation_consistency"]["status"] == "incomplete"


def test_aggregate_detects_identity_alias_conflict_and_records_invalid_metadata_as_missing():
    item = frame(0)
    item["payload"]["model"] = {"sha256": "b" * 64, "version": ["invalid-version"]}
    value = aggregate([item])["details"]["aggregation_consistency"]
    assert value["fields"]["model_sha256"]["status"] == "conflict"
    assert value["fields"]["model_version"]["invalid_frame_ids"] == ["frame-0"]
    json.dumps(value, allow_nan=False)


def test_aggregate_does_not_infer_model_version_from_branch_name():
    items = [frame(0), frame(1)]
    for item in items:
        del item["payload"]["provenance"]["model_version"]
    audit = aggregate(items)["details"]["aggregation_consistency"]
    assert audit["fields"]["model_version"]["variants"] == []
    assert audit["fields"]["model_version"]["missing_frame_ids"] == ["frame-0", "frame-1"]
    assert audit["status"] == "consistent_reported"  # model SHA present; version still explicitly unknown
