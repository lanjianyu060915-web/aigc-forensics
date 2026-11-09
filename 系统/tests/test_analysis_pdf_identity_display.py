"""身份表只展示每张查询脸首候选的开放集结果；不改变检测证据。"""
from copy import deepcopy

import pytest

from reporting import analysis_pdf as pdf


def candidates():
    return [{"rank": rank, "identity_id": f"person_{rank}", "score": score,
             "open_set_state": "high_confidence"}
            for rank, score in ((1, .8), (2, .6), (3, .4))]


def displays(data):
    return [row["open_set_display"] for row in pdf._identity_candidate_rows(data)]


def test_top_three_only_first_has_own_face_state():
    data = {"faces": [{"face_index": 0, "open_set": {"state": "high_confidence"}, "candidates": candidates()}]}
    assert displays(data) == ["高可信候选", "未单独判定", "未单独判定"]


def test_multiple_faces_use_own_state_and_keep_original_order():
    data = {"open_set_state": "high_confidence", "faces": [
        {"face_index": 8, "open_set": {"state": "no_reliable_candidate"}, "candidates": candidates()[::-1]},
        {"face_index": 2, "open_set": {"state": "insufficient_evidence"}, "candidates": candidates()}]}
    rows = pdf._identity_candidate_rows(data)
    assert [r["rank"] for r in rows] == [3, 2, 1, 1, 2, 3]
    assert [r["face_index"] for r in rows] == [8, 8, 8, 2, 2, 2]
    assert displays(data) == ["未单独判定", "未单独判定", "未发现可靠候选", "证据不足，建议复核", "未单独判定", "未单独判定"]


@pytest.mark.parametrize("state", [None, "", "unknown", "high", "completed", {"state": "high_confidence"}])
def test_missing_or_unknown_face_state_never_borrows_global_or_candidate(state):
    data = {"open_set_state": "high_confidence", "open_set": {"state": "high_confidence"},
            "faces": [{"face_index": 0, "candidates": candidates(), "open_set": {"state": state}}]}
    assert displays(data) == ["未记录", "未单独判定", "未单独判定"]


def test_second_face_missing_state_never_borrows_first_or_global():
    data = {"open_set_state": "high_confidence", "faces": [
        {"face_index": 0, "open_set": {"state": "high_confidence"}, "candidates": candidates()},
        {"face_index": 1, "candidates": candidates()}]}
    assert displays(data) == ["高可信候选", "未单独判定", "未单独判定", "未记录", "未单独判定", "未单独判定"]


@pytest.mark.parametrize("state,expected", [("high_confidence", "高可信候选"),
    ("insufficient_evidence", "证据不足，建议复核"),
    ("no_reliable_candidate", "未发现可靠候选"), ("unavailable", "无法完成比对")])
def test_four_existing_open_set_states_remain_displayable(state, expected):
    assert displays({"faces": [{"open_set": {"state": state}, "candidates": candidates()}]})[0] == expected


@pytest.mark.parametrize("top_state", [{"open_set_state": "high_confidence"},
    {"open_set": {"state": "high_confidence"}},
    {"open_set_state": None, "open_set": {"state": "high_confidence"}}])
def test_top_level_fallback_only_when_no_per_face_candidates(top_state):
    assert displays({**top_state, "faces": [{"face_index": 1, "candidates": []}], "candidates": candidates()}) == ["高可信候选", "未单独判定", "未单独判定"]


def test_no_per_face_candidates_no_top_state_does_not_borrow_candidate_state():
    assert displays({"candidates": candidates()}) == ["未记录", "未单独判定", "未单独判定"]


def test_per_face_precedence_keeps_previous_duplicate_avoidance():
    data = {"candidates": candidates(), "faces": [{"face_index": 4, "candidates": candidates(), "open_set": {"state": "unavailable"}}]}
    assert len(pdf._identity_candidate_rows(data)) == 3
    assert displays(data)[0] == "无法完成比对"


@pytest.mark.parametrize("rank", [None, "1", "2", True, False, 0, -1, 1.0, 2.5])
def test_unknown_or_invalid_rank_not_inferred_from_list_position(rank):
    row = {"rank": rank, "identity_id": "person_1"}
    assert displays({"faces": [{"open_set": {"state": "high_confidence"}, "candidates": [row]}]}) == ["未记录"]


def test_missing_rank_not_inferred_from_list_position():
    assert displays({"open_set_state": "high_confidence", "candidates": [{"identity_id": "person_1"}]}) == ["未记录"]


def test_source_data_unchanged_and_display_field_not_written_into_evidence():
    data = {"open_set_state": "high_confidence", "candidates": candidates(), "faces": [
        {"face_index": 2, "open_set": {"state": "no_reliable_candidate"}, "candidates": candidates()}]}
    before = deepcopy(data)
    result = pdf._identity_candidate_rows(data)
    assert data == before
    assert all("open_set_display" not in row for row in data["faces"][0]["candidates"])
    assert [r["score"] for r in result] == [.8, .6, .4]
    assert all(r["open_set_state"] == "high_confidence" for r in result)


def test_table_header_and_footnote_explain_candidate_limit(monkeypatch):
    tables, paragraphs = [], []
    monkeypatch.setattr(pdf, "_table", lambda story, title, headers, data, *args: tables.append((title, headers, data)))
    monkeypatch.setattr(pdf, "_p", lambda value, *args: paragraphs.append(value) or value)
    pdf._module_tables([], "identity", {"E07": {"open_set_state": "high_confidence", "candidates": candidates()}}, {})
    assert tables[0][1][-1] == "首候选开放集结果"
    assert [r[-1] for r in tables[0][2]] == ["高可信候选", "未单独判定", "未单独判定"]
    assert paragraphs == ["开放集结果仅针对每张查询脸的首候选；其余候选仅作排名展示，候选不等于身份确认。"]
