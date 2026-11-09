"""New synthetic dictionaries only: no image, model, library or historical score reads."""
from copy import deepcopy
from pathlib import Path

import pytest

from reporting.coverage_summary_v4 import (
    append_coverage_notes, face_binding_coverage_rows, module_face_coverage_rows,
)
from reporting.face_evidence_v4 import build_face_coverage_evidence


SCOPE = {"source_id": "synthetic-query", "frame_id": "synthetic-frame", "coordinate_space": "original_image"}
BOX = [2, 3, 18, 22]


def audit():
    return {"status": "recorded", "coverage": {"count_status": "recorded", "detected_total": 1,
            "cropped_count": 1, "scored_count": 1, "used_for_band_count": 1,
            "truncated_count": 0, "failed_count": 0, "model_failed_count": 0},
            "face_checks": [{**SCOPE, "face_uid": "synthetic/0", "bbox": BOX[:], "score": 0.321,
                             "crop_status": "success", "score_status": "scored", "used_for_band": True}]}


def observation(**overrides):
    return {"query_face_id": 0, "reference_face_id": 9, "query_bbox_original": BOX[:],
            "reference_bbox_original": [50, 50, 66, 69], "match_status": "matched",
            "cosine_similarity": 0.432, **overrides}


def candidate(*, rows=None, status="success", alignment="completed", rank=1):
    return {"rank": rank, "status": status, "comparison": {"alignment_status": alignment,
            "faces": [observation()] if rows is None else rows}}


def pair(*candidates, scope=None):
    return {"status": "success", "scope": "image", "query_face_scope": dict(SCOPE) if scope is None else scope,
            "query_face_count": 1, "candidates": list(candidates)}


def test_pair_processed_requires_unique_binding_and_actual_finite_measurement():
    original = audit()
    before = deepcopy(original)
    result = build_face_coverage_evidence(original, {}, pair(candidate()))
    assert result["face_checks"][0]["pair_processed"] is True
    assert result["pair_coverage"]["processed_bound_count"] == 1
    assert result["pair_coverage"]["bound_count"] == 1
    assert original == before
    assert result["face_checks"][0]["score"] == 0.321
    assert result["face_checks"][0]["used_for_band"] is True


@pytest.mark.parametrize("cosine", [None, float("nan"), float("inf"), True, "0.4"])
def test_pair_missing_or_nonfinite_measurement_remains_unknown(cosine):
    result = build_face_coverage_evidence(audit(), {}, pair(candidate(rows=[observation(cosine_similarity=cosine)])))
    assert result["face_checks"][0]["pair_processed"] is None
    assert result["face_checks"][0]["pair_processing_status"] == "unknown"
    assert result["pair_coverage"]["bound_count"] == 1
    assert result["pair_coverage"]["processed_bound_count"] == 0


@pytest.mark.parametrize("entry,expected", [
    (candidate(status="error"), "failed"),
    (candidate(alignment="failed"), "failed"),
    (candidate(alignment="unavailable"), "unknown"),
    (candidate(rows=[observation(match_status="unmatched", reference_face_id=None)]), "unmatched"),
    (candidate(rows=[observation(processing_status="skipped")]), "skipped"),
    (candidate(rows=[observation(status=["malformed"])]), "processed"),
])
def test_pair_processing_states_are_not_candidate_success(entry, expected):
    result = build_face_coverage_evidence(audit(), {}, pair(entry))
    assert result["face_checks"][0]["pair_processing_status"] == expected
    assert result["pair_coverage"]["candidate_attempt_status_counts"][expected] == 1
    assert result["pair_coverage"]["processed_bound_count"] == (1 if expected == "processed" else 0)


@pytest.mark.parametrize("scope", [{}, {**SCOPE, "frame_id": "other-frame"}, {**SCOPE, "source_id": "other-query"}])
def test_pair_same_numeric_face_id_and_box_are_insufficient_without_query_scope(scope):
    result = build_face_coverage_evidence(audit(), {}, pair(candidate(), scope=scope))
    assert result["pair_coverage"]["bound_count"] == 0
    assert result["pair_coverage"]["processed_bound_count"] == 0
    assert result["face_checks"][0]["pair_processed"] is not True


def test_pair_duplicate_geometry_is_ambiguous_not_double_processed():
    result = build_face_coverage_evidence(audit(), {}, pair(candidate(rows=[observation(), observation(query_face_id=2)])))
    assert result["face_checks"][0]["pair_processing_status"] == "ambiguous"
    assert result["face_checks"][0]["pair_processed"] is None
    assert result["pair_coverage"]["ambiguous_count"] == 1
    assert result["pair_coverage"]["processed_bound_count"] == 0


def test_pair_reference_only_coordinates_remain_unknown():
    result = build_face_coverage_evidence(audit(), {}, pair(candidate(rows=[observation(query_bbox_original=None)])))
    assert result["pair_coverage"]["bound_count"] == 0
    assert result["pair_coverage"]["unknown_count"] == 1
    assert result["face_checks"][0]["pair_processed"] is None


def test_pair_multiple_candidates_count_query_face_once_and_keep_failed_attempt():
    result = build_face_coverage_evidence(audit(), {}, pair(candidate(), candidate(rank=2), candidate(status="error", rank=3)))
    coverage = result["pair_coverage"]
    assert coverage["processed_bound_count"] == 1
    assert coverage["partial_count"] == 1
    assert coverage["candidate_failed_count"] == 1
    assert coverage["candidate_attempt_status_counts"]["processed"] == 2
    assert coverage["candidate_attempt_status_counts"]["failed"] == 1
    assert result["face_checks"][0]["pair_processing_status"] == "partial"
    assert result["face_checks"][0]["pair_processed"] is True


def test_candidate_failure_without_query_observation_is_not_attributed_to_a_face():
    result = build_face_coverage_evidence(audit(), {}, pair(candidate(status="error", rows=[])))
    assert result["pair_coverage"]["candidate_failed_count"] == 1
    assert result["pair_coverage"]["failed_bound_count"] == 0
    assert result["pair_coverage"]["unknown_count"] == 1


def test_module_own_totals_do_not_replace_query_audit_denominator():
    identity = {"status": "success", "query_face_scope": dict(SCOPE),
                "faces": [{"embedding_face_bbox_original": BOX[:], "status": "success"}],
                "face_search": {"coverage": {"detected_total": 9, "processed_count": 3, "failed_count": 1, "truncated_count": 6}}}
    result = build_face_coverage_evidence(audit(), identity, pair(candidate()))
    rows = module_face_coverage_rows(result)
    assert rows[0]["检出记录"] == "1"
    assert rows[1]["检出记录"] == "9" and rows[1]["处理记录"] == "3"
    assert rows[1]["处理口径"] == "模块尝试记录，包含失败"
    assert rows[2]["处理记录"] == "1"
    bindings = face_binding_coverage_rows(result)
    assert bindings[0]["已处理对应脸"] == "1"
    assert result["identity_coverage"]["query_observation_count"] == 1
    assert result["coverage"]["detected_total"] == 1


def test_identity_timeout_is_failed_only_with_actual_embedding_correspondence():
    result = build_face_coverage_evidence(audit(), {"query_face_scope": dict(SCOPE),
        "faces": [{"embedding_face_bbox_original": BOX[:], "status": "timeout"}]}, {})
    assert result["identity_coverage"]["failed_bound_count"] == 1
    assert result["face_checks"][0]["identity_processed"] is False


def test_old_missing_audit_preserves_unknown_instead_of_zero_filled_module_counts():
    result = build_face_coverage_evidence({}, {}, {})
    assert all(row["检出记录"] == "未记录" for row in module_face_coverage_rows(result))
    assert module_face_coverage_rows({})[2]["处理记录"] == "未记录"
    assert face_binding_coverage_rows({})[0]["已处理对应脸"] == "未记录"


def test_report_appends_module_notes_after_existing_writer_text():
    enriched = build_face_coverage_evidence(audit(), {}, pair(candidate(), candidate(status="error", rank=2)))
    package = {"items": [{"item_id": "E04", "data": {"face_audit": enriched}}]}
    result = append_coverage_notes({"process": {"detection": "D", "identity": "I", "source": "S"}}, package)
    assert result["process"]["detection"].startswith("D\n")
    assert result["process"]["identity"].startswith("I\n身份候选覆盖")
    assert result["process"]["source"].startswith("S\n素材配对覆盖")
    assert "已处理对应脸 1" in result["process"]["source"]
    assert "部分完成 1 已包含" in result["process"]["source"]
    assert "不证明同一身份或伪造类型" in result["process"]["source"]


def test_ui_prefers_merged_audit_and_shows_separate_module_and_face_states(monkeypatch):
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    monkeypatch.syspath_prepend(str(frontend))
    from components import aigc_views
    tables, captions = [], []
    monkeypatch.setattr(aigc_views.st, "dataframe", lambda rows, **_kwargs: tables.append(rows))
    monkeypatch.setattr(aigc_views.st, "caption", captions.append)
    monkeypatch.setattr(aigc_views.st, "markdown", lambda _text: None)
    enriched = build_face_coverage_evidence(audit(), {}, pair(candidate(status="error")))
    raw = {"coverage": {"count_status": "recorded", "detected_total": 99}}
    payload = {"face_coverage_evidence": enriched, "general_aigc": {"face_preprocessing": raw}, "overall": {"band": "low"}}
    before = deepcopy(payload)
    aigc_views.render_face_branch_section(payload)
    assert len(tables) == 3
    assert [row["模块"] for row in tables[0]] == ["人脸鉴伪", "身份候选", "素材配对"]
    assert tables[0][0]["检出记录"] == "1"
    assert tables[1][1]["失败对应脸"] == "1"
    assert tables[2][0]["配对检查"] == "失败"
    assert tables[2][0]["身份检查"] == "未知"
    assert any("未知和歧义不计为已处理" in text for text in captions)
    assert payload == before


def test_ui_old_raw_audit_fallback_does_not_invent_identity_or_pair(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "frontend"))
    from components import aigc_views
    tables = []
    monkeypatch.setattr(aigc_views.st, "dataframe", lambda rows, **_kwargs: tables.append(rows))
    monkeypatch.setattr(aigc_views.st, "caption", lambda _text: None)
    monkeypatch.setattr(aigc_views.st, "markdown", lambda _text: None)
    aigc_views.render_face_branch_section({"general_aigc": {"face_preprocessing": audit()}})
    assert tables[0][1]["检出记录"] == "未记录"
    assert tables[0][2]["处理记录"] == "未记录"
    assert tables[2][0]["配对检查"] == "未知"


def test_ui_no_eligible_main_face_does_not_claim_nothing_was_detected(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "frontend"))
    from components import aigc_views
    from contextlib import nullcontext
    cards = []
    monkeypatch.setattr(aigc_views, "verdict_card", lambda *args: cards.append(args))
    monkeypatch.setattr(aigc_views.st, "columns", lambda count: [nullcontext() for _ in range(count)])
    monkeypatch.setattr(aigc_views.st, "caption", lambda _text: None)
    monkeypatch.setattr(aigc_views.st, "markdown", lambda _text: None)
    aigc_views.render_overall_section({"face_coverage_evidence": audit(), "overall": {"band": "low",
        "branches": [{"status": "no_face", "label": "face"}]}})
    assert "未覆盖情况见逐脸记录" in cards[1][2]
    assert "没有可用的人脸" not in cards[1][2]


@pytest.mark.parametrize("face,coverage,expected", [
    ({"crops": [{"file": "unused-crop.jpg"}]}, {}, "未记录"),
    ({"per_face": [{"score": 0.2}, {"score": float("nan")}, {"score": None}]}, {}, "1 张"),
    ({"per_face": [{"score": 0.2}]}, {"scored_count": 0}, "0 张"),
])
def test_ui_scored_count_never_substitutes_crop_count(monkeypatch, face, coverage, expected):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "frontend"))
    from components.aigc_views import _scored_face_count
    assert _scored_face_count(face, {"coverage": coverage}) == expected


def test_streamlit_renders_synthetic_merged_coverage_without_loading_media():
    from streamlit.testing.v1 import AppTest
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    enriched = build_face_coverage_evidence(audit(), {}, pair(candidate(), candidate(status="error", rank=2)))
    payload = {"face_coverage_evidence": enriched, "general_aigc": {}}
    script = (f"import sys\nsys.path.insert(0, {str(frontend)!r})\n"
              "from components.aigc_views import render_face_branch_section\n"
              f"render_face_branch_section({payload!r})\n")
    app = AppTest.from_string(script).run(timeout=15)
    assert not app.exception
    assert len(app.dataframe) == 3
    assert app.dataframe[0].value.iloc[2]["模块"] == "素材配对"
    assert app.dataframe[1].value.iloc[1]["部分完成"] == "1"
    assert app.dataframe[2].value.iloc[0]["配对检查"] == "部分完成"
