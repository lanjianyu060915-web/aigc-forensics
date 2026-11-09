"""溯源接线的缓存命名、快照、冻结规则与失败隔离测试。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from case_libraries import case_source_cache, describe_case_libraries
from source_upgrade import case_report_lines, prepare_case_context, run_source_pair


def picture(path: Path, value: int = 80) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", np.full((24, 32, 3), value, dtype=np.uint8))
    assert ok
    encoded.tofile(path)
    return path


@pytest.fixture
def setup(tmp_path):
    product = tmp_path / "产品"
    product.mkdir()
    data = tmp_path / "用户数据"
    source = tmp_path / "案件素材"
    identity = tmp_path / "案件身份"
    picture(source / "底图.png")
    picture(identity / "人员甲" / "参考.png")
    task = tmp_path / "任务"
    task.mkdir()
    query = picture(task / "query.png")
    rule = product / "规则.json"
    rule.write_text("{}", encoding="utf-8")

    def resolver(value, config_path):
        raw = str(value).replace("{DATA_ROOT}", str(data)).replace("{APP_ROOT}", str(config_path.parent))
        path = Path(raw)
        return (path if path.is_absolute() else config_path.parent / path).resolve()

    config = {
        "source": {"library": str(source), "feature_cache": "{DATA_ROOT}/runtime/cache/source",
                   "thresholds": {"high": 0.9}},
        "identity": {"library": str(identity), "model_root": "模型"},
        "case_libraries": {"enabled": True},
        "pair_compare": {"enabled": True, "rule_path": str(rule),
                         "expected_rule_sha256": hashlib.sha256(rule.read_bytes()).hexdigest()},
    }
    return {"config": config, "config_path": product / "config.json", "resolve_path": resolver,
            "source": source, "identity": identity, "task": task, "query": query, "rule": rule}


def context_args(setup):
    return {key: setup[key] for key in ("config", "config_path", "resolve_path")}


def pair_args(setup):
    return {**context_args(setup), "task_dir": setup["task"], "query_path": setup["query"],
            "source_result": {"status": "success", "candidates": []}}


def test_case_namespace_matches_builder_and_config_is_unchanged(setup):
    original = deepcopy(setup["config"])
    context = prepare_case_context(**context_args(setup), selected_modules={"source", "identity"})
    snapshot = describe_case_libraries(
        cache_root=setup["resolve_path"]("{DATA_ROOT}/runtime/cache", setup["config_path"]),
        source_root=setup["source"], identity_root=setup["identity"])
    expected = case_source_cache(
        setup["resolve_path"](original["source"]["feature_cache"], setup["config_path"]),
        setup["source"], snapshot["libraries"]["source"]["manifest_sha256"])
    assert context["case_libraries"]["status"] == "success"
    assert Path(context["source_section"]["feature_cache"]) == expected
    assert setup["config"] == original
    context["source_section"]["thresholds"]["high"] = 0.1
    assert setup["config"]["source"]["thresholds"]["high"] == 0.9


def test_library_overrides_are_the_actual_snapshot(setup, tmp_path):
    source = tmp_path / "其他素材"
    identity = tmp_path / "其他身份"
    picture(source / "新图.png", 120)
    picture(identity / "人员乙" / "参考.png", 160)
    context = prepare_case_context(**context_args(setup), source_library=source, identity_library=identity)
    assert context["case_libraries"]["libraries"]["source"]["name"] == "其他素材"
    assert context["case_libraries"]["libraries"]["identity"]["name"] == "其他身份"


def test_legacy_configuration_and_detection_only_are_noops(setup):
    config = deepcopy(setup["config"])
    config.pop("case_libraries")
    config.pop("pair_compare")
    args = {**context_args(setup), "config": config}
    context = prepare_case_context(**args)
    assert context == {"enabled": False, "source_section": config["source"], "case_libraries": None}
    assert run_source_pair(**{**pair_args(setup), "config": config}) is None
    assert not (setup["task"] / "pair_compare_result.json").exists()
    with patch("source_upgrade.describe_case_libraries") as describe:
        context = prepare_case_context(**context_args(setup), selected_modules={"deepfake"})
    describe.assert_not_called()
    assert context["case_libraries"] is None


def test_identity_only_does_not_replace_source_cache(setup):
    context = prepare_case_context(**context_args(setup), selected_modules={"identity"})
    assert context["source_section"] == setup["config"]["source"]
    assert context["case_libraries"]["status"] == "success"


@pytest.mark.parametrize("failure", [FileNotFoundError("目录不存在"), ValueError("资源引用越出库根目录")])
def test_metadata_failure_is_recorded_without_changing_config(setup, failure):
    original = deepcopy(setup["config"])
    with patch("source_upgrade.describe_case_libraries", side_effect=failure):
        context = prepare_case_context(**context_args(setup))
    assert context["case_libraries"]["status"] == "error"
    assert str(failure) in context["case_libraries"]["error_message"]
    assert context["source_section"]["feature_cache"] is None
    assert setup["config"] == original


def test_report_has_current_digest_without_invented_build_time_or_root(setup):
    context = prepare_case_context(**context_args(setup))
    metadata = context["case_libraries"]
    assert metadata["built_at"] is None
    text = "\n".join(case_report_lines(metadata))
    assert metadata["libraries"]["source"]["manifest_sha256"] in text
    assert "未记录与当前内容匹配的建库时间" in text
    assert str(setup["source"]) not in text
    assert str(setup["identity"]) not in text
    assert case_report_lines(None) == []


@pytest.mark.parametrize("expected", [None, "", "错误摘要"])
def test_rule_hash_is_required_before_comparator_is_called(setup, expected):
    setup["config"]["pair_compare"]["expected_rule_sha256"] = expected
    with patch("source_upgrade.run_pair_comparison") as compare:
        result = run_source_pair(**pair_args(setup))
    compare.assert_not_called()
    assert result["status"] == "error"
    assert "冻结规则" in result["error_message"]
    assert result["forgery_type"] is None
    stored = json.loads((setup["task"] / "pair_compare_result.json").read_text(encoding="utf-8"))
    assert stored == result


def test_mismatched_hash_fails_before_model_loading(setup):
    setup["config"]["pair_compare"]["expected_rule_sha256"] = "0" * 64
    args = pair_args(setup)
    args["source_result"]["candidates"] = [{"file_path": str(setup["source"] / "底图.png")}]
    with patch("pair_integration._load_comparator") as load:
        result = run_source_pair(**args)
    load.assert_not_called()
    assert result["status"] == "error"
    assert "冻结记录不一致" in result["error_message"]
    assert result["conclusion_enabled"] is False


def test_mismatched_hash_is_recorded_even_without_candidates(setup):
    setup["config"]["pair_compare"]["expected_rule_sha256"] = "0" * 64
    with patch("source_upgrade.run_pair_comparison") as compare:
        result = run_source_pair(**pair_args(setup))
    compare.assert_not_called()
    assert result["status"] == "error"
    assert result["rule_sha256"] == hashlib.sha256(setup["rule"].read_bytes()).hexdigest()


def test_source_cache_cannot_be_placed_inside_a_case_library(setup):
    setup["config"]["source"]["feature_cache"] = str(setup["source"] / "缓存")
    context = prepare_case_context(**context_args(setup))
    assert context["case_libraries"]["status"] == "error"
    assert "不能放在案件库里面" in context["case_libraries"]["error_message"]
    assert context["source_section"]["feature_cache"] is None
    assert not (setup["source"] / "缓存").exists()


def test_video_uses_first_evidence_timestamp_without_mutating_source(setup):
    args = pair_args(setup)
    args["source_result"].update(frame_evidence=[{"timestamp_seconds": 4.25}, {"timestamp_seconds": 8.0}],
                                 source_frames=["首要帧", "其他帧"])
    original = deepcopy(args["source_result"])
    with patch("source_upgrade.run_pair_comparison", return_value={"status": "success", "candidates": []}) as compare:
        result = run_source_pair(**args, input_kind="video")
    assert compare.call_args.kwargs["scope"] == "frame"
    assert compare.call_args.kwargs["timestamp_seconds"] == 4.25
    assert compare.call_args.kwargs["expected_rule_sha256"] == setup["config"]["pair_compare"]["expected_rule_sha256"]
    assert args["source_result"] == original
    assert result["mode"] == "manual_review"
    assert result["forgery_type"] is None


def test_pair_crash_and_result_save_failure_do_not_escape(setup):
    args = pair_args(setup)
    original_source = deepcopy(args["source_result"])
    original_config = deepcopy(setup["config"])
    with patch("source_upgrade.run_pair_comparison", side_effect=RuntimeError("模拟配对故障")):
        result = run_source_pair(**args)
    assert result["status"] == "error"
    assert "模拟配对故障" in result["error_message"]
    assert args["source_result"] == original_source
    assert setup["config"] == original_config
    with patch("source_upgrade.run_pair_comparison", return_value={"status": "success"}), \
            patch("source_upgrade._write_pair_result", side_effect=OSError("模拟写入失败")):
        result = run_source_pair(**args)
    assert result["status"] == "error"
    assert "模拟写入失败" in result["persistence_error"]


def test_image_scope_does_not_reuse_a_video_timestamp(setup):
    args = pair_args(setup)
    args["source_result"]["frame_evidence"] = [{"timestamp_seconds": 4.25}]
    with patch("source_upgrade.run_pair_comparison", return_value={"status": "success"}) as compare:
        run_source_pair(**args)
    assert compare.call_args.kwargs["scope"] == "image"
    assert compare.call_args.kwargs["timestamp_seconds"] is None
