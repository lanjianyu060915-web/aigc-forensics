"""人工比对和案件库的报告贯通，不产生自动类型结论。"""

import json
import pytest

from pair_integration import MANUAL_NOTICE
from reporting.report_builder import build_final_report, render_html_report, render_text_report
from source_upgrade import run_source_pair


@pytest.mark.parametrize("code", ["alignment_failed", "source_not_supported", "same_source_unchanged",
    "face_replaced", "face_edited_identity_kept", "edited_elsewhere", "same_background_other_person", "inconclusive"])
def test_every_experimental_class_is_removed_in_manual_mode(code):
    from pair_integration import _manual_observations
    public = _manual_observations({"code": code, "source_supported": True,
        "reason": "实验说明", "other_edits": True, "alignment": {"inliers": 5},
        "faces": [{"query_face_id": 0, "reference_face_id": 1,
                   "status": code, "reason": "实验说明", "cosine_similarity": 0.4}]})
    assert all(key not in public for key in ("code", "source_supported", "reason", "other_edits"))
    assert all(key not in public["faces"][0] for key in ("status", "reason"))
    assert public["faces"][0]["match_status"] == "matched"
    assert public["faces"][0]["cosine_similarity"] == 0.4


def test_manual_pair_and_case_snapshot_reach_all_text_formats():
    pair = {"mode": "manual_review", "conclusion_enabled": False, "scope": "frame",
            "timestamp_seconds": 2.25, "candidates": [], "forgery_type": None}
    library = {"name": "演示案件", "libraries": {"source": {
        "name": "素材样例", "file_count": 3, "manifest_sha256": "a" * 64,
        "built_at": "2026-01-01T10:00:00+08:00", "build_record_matches": True}}}
    report = build_final_report("task", unified_result={"pair_compare": pair, "case_libraries": library})
    assert report["forgery_type"] is None
    assert report["pair_compare"] == pair
    pair["scope"] = "changed"
    assert report["pair_compare"]["scope"] == "frame"
    for rendered in (render_text_report(report), render_html_report(report)):
        assert MANUAL_NOTICE in rendered
        assert "2.25" in rendered and "首要帧" in rendered
        assert "演示案件" in rendered and "文件数：3" in rendered
        assert "a" * 64 in rendered and "2026-01-01T10:00:00+08:00" in rendered


def test_old_report_has_no_new_automatic_conclusion():
    report = build_final_report("old")
    assert "pair_compare" not in report and "forgery_type" not in report
    assert "来源结论与本次比对库" not in render_text_report(report)


def test_alignment_failure_is_not_reported_as_non_source():
    from pair_integration import pair_report_lines
    text = "\n".join(pair_report_lines({"candidates": [{"rank": 1, "status": "success",
        "comparison": {"alignment_status": "failed", "alignment": {"inliers": 0}}}]}))
    assert "失败，暂不能判断是否同源" in text
    assert "不是同一张" not in text


def test_default_face_model_resolves_to_buffalo_directory(tmp_path, monkeypatch):
    import hashlib
    import source_upgrade

    rule = tmp_path / "rule.json"
    rule.write_text("{}", encoding="utf-8")
    received = {}
    def fake(**kwargs):
        received.update(kwargs)
        return {"status": "success", "candidates": []}
    monkeypatch.setattr(source_upgrade, "run_pair_comparison", fake)
    result = run_source_pair(
        config={"pair_compare": {"enabled": True, "rule_path": str(rule),
                                  "expected_rule_sha256": hashlib.sha256(b"{}").hexdigest()},
                "identity": {"model_root": str(tmp_path / "resources")},
                "source": {"library": str(tmp_path)}},
        config_path=tmp_path / "config.json", resolve_path=lambda value, _: tmp_path / value,
        task_dir=tmp_path, query_path=tmp_path / "query.png", source_result={})
    assert result["status"] == "success"
    assert received["model_dir"] == tmp_path / "resources" / "models" / "buffalo_l"
