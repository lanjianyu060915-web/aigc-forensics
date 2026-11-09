"""配对接入失败隔离、候选覆盖与人工核验模式测试。"""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pair_integration import MANUAL_NOTICE, pair_report_lines, run_pair_comparison


class FakeComparator:
    def __init__(self, failed: str = "", code: str = "same_source_unchanged"):
        self.failed = failed
        self.code = code
        self.prepared = []
        self.compared = 0
        self.model_load_seconds = 0.0

    def prepare(self, path):
        self.prepared.append(Path(path).name)
        if Path(path).name == self.failed:
            raise RuntimeError("模拟候选解码失败")
        return SimpleNamespace(image=np.zeros((32, 32, 3), np.uint8), faces=[])

    def compare_prepared(self, query, reference, return_artifacts):
        self.compared += 1
        return {"code": self.code, "alignment": {"inliers": 12}, "faces": []}, {}


def fake_writer(directory, visual, compared):
    paths = {}
    for name, image in visual.items():
        path = directory / f"{name}.png"
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        encoded.tofile(path)
        paths[name] = str(path)
    (directory / "result.json").write_text(json.dumps(compared), encoding="utf-8")
    return paths


@pytest.fixture
def setup(tmp_path):
    library = tmp_path / "素材库"
    library.mkdir()
    candidates = []
    for index in range(1, 5):
        path = library / f"{index}.png"
        path.write_bytes(b"fixture")
        candidates.append({"rank": index, "file_path": str(path)})
    rule = tmp_path / "rule.json"
    rule.write_text("{}", encoding="utf-8")
    task = tmp_path / "任务"
    task.mkdir()
    query = task / "query.png"
    query.write_bytes(b"fixture")
    return dict(query_path=query, source_result={"status": "success", "candidates": candidates},
                source_library=library, task_dir=task, rule_path=rule,
                model_dir=tmp_path / "模型", cache_dir=tmp_path / "缓存", artifact_writer=fake_writer)


def test_processes_top_three_and_reuses_query(setup):
    engine = FakeComparator()
    value = run_pair_comparison(**setup, comparator=engine)
    assert engine.compared == 3
    assert engine.prepared == ["query.png", "1.png", "2.png", "3.png"]
    assert len(value["candidates"]) == 3
    assert value["status"] == "success"
    assert value["conclusion_enabled"] is False
    assert value["mode"] == "manual_review"
    assert "forgery_type" not in value
    assert len(value["artifact_manifest"]) == 9
    assert all(not Path(row["path"]).is_absolute() for row in value["artifact_manifest"])


def test_one_bad_candidate_does_not_skip_other_candidates(setup):
    value = run_pair_comparison(**setup, comparator=FakeComparator(failed="2.png"))
    assert [row["status"] for row in value["candidates"]] == ["success", "error", "success"]
    assert value["status"] == "partial_success"
    assert setup["source_result"]["status"] == "success"


def test_model_or_query_failure_is_an_optional_result(setup):
    value = run_pair_comparison(**setup, comparator=FakeComparator(failed="query.png"))
    assert value["status"] == "error"
    assert "模拟" in value["error_message"]
    assert setup["source_result"]["status"] == "success"


def test_invalid_candidate_path_is_not_opened(setup, tmp_path):
    outside = tmp_path / "越界.png"
    outside.write_bytes(b"fixture")
    setup["source_result"]["candidates"][0]["file_path"] = str(outside)
    engine = FakeComparator()
    value = run_pair_comparison(**setup, comparator=engine)
    assert value["candidates"][0]["status"] == "error"
    assert "越界.png" not in engine.prepared
    assert engine.compared == 2


def test_alignment_failure_keeps_parallel_images(setup):
    value = run_pair_comparison(**setup, comparator=FakeComparator(code="alignment_failed"))
    for row in value["candidates"]:
        assert set(row["artifacts"]) == {"query", "reference"}
        assert row["comparison"]["alignment_status"] == "failed"


def test_manual_report_does_not_publish_experimental_class(setup):
    value = run_pair_comparison(**setup, comparator=FakeComparator(code="face_replaced"))
    text = "\n".join(pair_report_lines(value))
    assert MANUAL_NOTICE in text
    assert "face_replaced" not in text
    assert "支持换脸" not in text
    assert "face_replaced" not in json.dumps(value)
    for path in Path(setup["task_dir"]).rglob("result.json"):
        assert "face_replaced" not in path.read_text(encoding="utf-8")


def test_video_is_frame_scoped(setup):
    value = run_pair_comparison(**setup, comparator=FakeComparator(), scope="frame", timestamp_seconds=1.5)
    assert value["scope"] == "frame"
    text = "\n".join(pair_report_lines(value))
    assert "1.5" in text and "不代表整段视频" in text


def test_opt_in_clue_record_preserves_frozen_rule_output_without_enabling_type(setup):
    value = run_pair_comparison(**setup, comparator=FakeComparator(code="face_replaced"), record_clue_assessment=True)
    assert value["conclusion_enabled"] is False
    assert "forgery_type" not in value
    assert all(row["clue_assessment"]["code"] == "face_replaced" for row in value["candidates"])
    assert all("code" not in row["comparison"] for row in value["candidates"])
    assert "face_replaced" not in "\n".join(pair_report_lines(value))


def test_missing_candidates_and_legacy_report_are_supported(setup):
    setup["source_result"] = {"status": "success", "candidates": []}
    engine = FakeComparator()
    value = run_pair_comparison(**setup, comparator=engine)
    assert value["status"] == "skipped"
    assert engine.prepared == []
    assert pair_report_lines(None) == []


def test_changed_rule_is_rejected_before_model_call(setup):
    engine = FakeComparator()
    value = run_pair_comparison(**setup, comparator=engine, expected_rule_sha256="0" * 64)
    assert value["status"] == "error"
    assert "冻结" in value["error_message"]
    assert engine.prepared == []
