"""总体结论和文件标识进入研判要点、解释、报告和历史列表的行为。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

FRONTEND_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_ROOT = FRONTEND_ROOT.parent
for path in (PRODUCT_ROOT, FRONTEND_ROOT):
    sys.path.insert(0, str(path))

from components.aigc_views import overall_finding, provenance_finding  # noqa: E402
from components.evidence_dashboard import PAGES, build_key_findings  # noqa: E402
from components.task_history import summarize_task  # noqa: E402
from decision.aigc_overall import attach_general_aigc  # noqa: E402
from decision.evidence_matrix import build_evidence_matrix  # noqa: E402
from reporting.explainability import build_deepfake_explanation  # noqa: E402
from reporting.report_builder import build_final_report, render_text_report  # noqa: E402


LEGACY = {
    "status": "success",
    "query_image": "query.jpg",
    "is_deepfake": False,
    "score": 0.00001,
    "label": "real",
    "runtime_seconds": 0.1,
    "details": {
        "branch": "vit_full_image",
        "band": "low",
        "band_label": "证据不足",
        "band_thresholds": {"low_max": 0.0001, "medium_max": 0.0002},
    },
    "method": {"name": "二代整图深伪风险甄别（ViT 分支）", "branch": "vit_full_image"},
    "error_message": None,
    "warnings": [],
    "limitations": [],
}
GENERAL = {
    "status": "success",
    "score": 0.95,
    "logit": 2.9,
    "band": "high",
    "band_label": "高风险提示",
    "band_reason": "分数高于上界",
    "branches": {
        "formula": "0.8 × [0.5 × CLIP 原头 + 0.5 × 重训头] + 0.2 × CommunityForensics",
        "clip_original_logit": 3.0,
        "clip_refit_logit": 4.0,
        "clip_logit": 3.5,
        "community_logit": 0.5,
    },
    "warnings": ["通用分支告警"],
    "limitations": ["通用分支局限"],
    "error_message": None,
}
LABELLED = {
    "status": "success",
    "verdict": "labelled_ai",
    "verdict_label": "元数据声明属于生成合成内容（未核验声明真实性）",
    "aigc_labels": [
        {
            "standard": "GB 45438-2025",
            "label": "1",
            "label_meaning": "属于人工智能生成合成内容",
            "content_producer": "00119144030008867405X210001",
            "content_propagator": "",
        }
    ],
    "c2pa": {"present": True, "note": "只做结构检测，未验证签名"},
    "generator_hints": [{"source": "png_text:parameters", "value": "Steps: 20"}],
    "limitations": ["没有标识不代表不是生成内容"],
}


def _merged() -> dict:
    return attach_general_aigc(LEGACY, GENERAL)


def test_pages_include_file_labels_page() -> None:
    assert "文件标识" in PAGES


def test_overall_card_replaces_branch_band() -> None:
    card = overall_finding(_merged())
    assert card is not None
    assert card["value"] == "高风险提示"
    assert card["tone"] == "bad"
    assert overall_finding(LEGACY) is None


def test_key_findings_use_overall_and_add_label_card() -> None:
    empty = {"status": "skipped", "candidates": []}
    cards, steps = build_key_findings(_merged(), empty, empty, LABELLED)
    assert [card["title"] for card in cards] == ["是否生成或伪造", "文件自带标识", "可能是谁", "原图在哪"]
    assert cards[0]["value"] == "高风险提示"
    assert any("文件标识" in step for step in steps)
    cards, _ = build_key_findings(LEGACY, empty, empty)
    assert len(cards) == 3


def test_provenance_card_states() -> None:
    assert provenance_finding(LABELLED)["tone"] == "bad"
    none = provenance_finding({"status": "success", "verdict": "no_label", "verdict_label": "未发现"})
    assert none["tone"] == "muted"
    assert "不代表" in none["note"]
    failed = provenance_finding({"status": "error", "error_message": "读不出"})
    assert failed["value"] == "未完成"


def test_explanation_leads_with_overall_and_keeps_branch_basis() -> None:
    output = build_deepfake_explanation(_merged())
    assert output["headline"] == "总体结论：高风险提示"
    assert "通用 AIGC 分支为高风险提示" in output["summary"]
    assert any("各路档位推导" in item for item in output["basis"])
    assert any("0.8 × [0.5 × CLIP 原头" in item for item in output["basis"])
    assert "通用分支局限" in output["limitations"]
    assert output["technical"]["general_aigc"]["logit"] == 2.9


def test_explanation_without_general_branch_is_unchanged() -> None:
    output = build_deepfake_explanation(LEGACY)
    assert "总体结论" not in output["headline"]


def test_evidence_matrix_grades_by_overall() -> None:
    matrix = build_evidence_matrix(_merged(), {}, {})
    block = matrix["modules"]["deepfake"]
    assert block["grade"] == "high"
    assert block["primary_value"] == 2.9


def test_text_report_lists_file_labels() -> None:
    report = build_final_report(
        "task1",
        deepfake_result=_merged(),
        identity_result={"status": "skipped", "candidates": []},
        source_result={"status": "skipped", "candidates": []},
        file_provenance=LABELLED,
    )
    text = render_text_report(report)
    assert "文件标识" in text
    assert "属于人工智能生成合成内容" in text
    assert "00119144030008867405X210001" in text
    assert "C2PA" in text
    assert "总体结论：高风险提示" in text


def test_text_report_without_file_labels_has_no_section() -> None:
    report = build_final_report(
        "task1",
        deepfake_result=LEGACY,
        identity_result={"status": "skipped", "candidates": []},
        source_result={"status": "skipped", "candidates": []},
    )
    assert "文件标识（" not in render_text_report(report)


def test_task_list_shows_overall_label(tmp_path: Path) -> None:
    (tmp_path / "task_status.json").write_text(json.dumps({"task_id": "t", "status": "success"}), encoding="utf-8")
    (tmp_path / "detection_result.json").write_text(json.dumps(_merged(), ensure_ascii=False), encoding="utf-8")
    summary = summarize_task(tmp_path)
    assert summary is not None
    assert summary["deepfake_label"] == "高风险提示"


def test_history_row_pairs_overall_band_with_general_score() -> None:
    from components.task_history import _deepfake_summary

    label, score = _deepfake_summary(_merged())
    assert label == "高风险提示"
    assert score == 2.9

