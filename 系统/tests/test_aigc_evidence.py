"""核对说明字段和全部总体档位组合，确认它们不改变检测结果。"""

from copy import deepcopy
import json

import pytest

from decision.aigc_evidence import (
    DISCLAIMER, LOW_NOTE, attach_evidence, build_hints, evidence_lines,
    get_band_evidence, load_band_evidence,
)
from decision.aigc_overall import attach_general_aigc, decide_overall
from tools.research.build_band_evidence import band_statistics, build


@pytest.mark.parametrize(("real_pct", "fake_pct", "ratio", "small"), [(10, 30, 3.0, False), (0, 30, None, True), (9, 30, 30 / 9, True)])
def test_counts_ratios_and_zero_denominator(real_pct, fake_pct, ratio, small):
    stats = band_statistics({"n": 100, "high": real_pct}, {"n": 100, "high": fake_pct}, "high")
    assert stats["real_count"] == real_pct
    assert stats["fake_count"] == fake_pct
    assert stats["ratio"] is None if ratio is None else stats["ratio"] == pytest.approx(ratio)
    assert stats["small_sample"] is small


def test_generated_statistics_match_frozen_summaries():
    data = build()
    assert data == load_band_evidence()
    expected = [21.262, 17.1, 10.977, 11.0, 2.778]
    for scenario, value in zip(data["scenarios"], expected, strict=True):
        assert scenario["bands"]["high"]["ratio"] == pytest.approx(value, abs=0.001)
        assert len(scenario["source_sha256"]) == 64


def test_missing_or_invalid_statistics_are_optional(tmp_path):
    path = tmp_path / "统计.json"
    assert load_band_evidence(path) == {}
    for text in ("坏文件", "[]", '{"scenarios": "错误类型"}'):
        path.write_text(text, encoding="utf-8")
        assert load_band_evidence(path) == {}
    assert get_band_evidence("low", {}) == {}
    assert get_band_evidence(None) == {}
    assert get_band_evidence("high", {"scenarios": [{"label": "缺少统计"}]}) == {}


@pytest.mark.parametrize(("edge", "expected"), [(255, True), (256, True), (257, False)])
def test_low_resolution_boundary(edge, expected):
    for width, height in ((edge, 1000), (1000, edge)):
        hints = build_hints("low", {"width": width, "height": height}, {})
        assert bool(hints) is expected
        if expected:
            assert hints[0]["code"] == "low_resolution"
            assert str(edge) in hints[0]["message"]


@pytest.mark.parametrize("media", [None, {}, {"width": 200}, {"width": 0, "height": 200}, {"width": "200", "height": 100}, {"width": float("nan"), "height": 100}])
def test_incomplete_media_has_no_hint(media):
    assert build_hints("low", media, {}) == []


@pytest.mark.parametrize("verdict", ["labelled_ai", "labelled_possible", "generator_hint", "no_label"])
@pytest.mark.parametrize("band", ["low", "medium", "high", None])
@pytest.mark.parametrize("status", ["success", "error"])
def test_every_provenance_verdict_and_band(verdict, band, status):
    hints = build_hints(band, {}, {"status": status, "verdict": verdict, "verdict_label": "标识说明"})
    expected = status == "success" and verdict != "no_label" and band == "low"
    assert bool(hints) is expected
    if expected:
        assert hints[0]["code"] == "provenance_conflict"
        assert "标识说明" in hints[0]["message"]


BANDS = ("low", "medium", "high", None)
EXPECTED = (
    ("low", "low", "low", "low"), ("medium",) * 4, ("medium",) * 4, ("low",) * 4,
    ("medium",) * 4, ("medium",) * 4, ("high",) * 4, ("medium",) * 4,
    ("high",) * 4, ("high",) * 4, ("high",) * 4, ("high",) * 4,
    ("low",) * 4, ("medium",) * 4, ("medium",) * 4, ("low", "medium", "medium", None),
)


@pytest.mark.parametrize(("general", "face", "whole", "expected"), [
    (general, face, whole, EXPECTED[g * 4 + f][w])
    for g, general in enumerate(BANDS) for f, face in enumerate(BANDS) for w, whole in enumerate(BANDS)
])
def test_all_overall_combinations_remain_fixed(general, face, whole, expected):
    assert decide_overall(general, face, whole)[0] == expected


def test_enrichment_preserves_every_existing_value():
    legacy = {"status": "success", "score": 0.12, "runtime_seconds": 1, "details": {"band": "high"}, "warnings": ["原警告"]}
    general = {"status": "success", "band": "low", "logit": -3.4, "media": {"width": 200, "height": 300}}
    before = attach_general_aigc(legacy, general)
    snapshot = deepcopy(before)
    after = attach_evidence(before, {"status": "success", "verdict": "labelled_ai"})
    assert before == snapshot
    assert len(after["overall"]["hints"]) == 2
    assert len(after["overall"]["band_evidence"]["rows"]) == 5
    assert DISCLAIMER in evidence_lines(after["overall"])
    assert LOW_NOTE in evidence_lines(after["overall"])
    after["overall"].pop("band_evidence")
    after["overall"].pop("hints")
    assert after == before
    assert attach_evidence(legacy, {}) == legacy


def test_legacy_report_has_no_new_text():
    assert evidence_lines({"band": "low"}) == []


def test_enrichment_is_json_serializable():
    assert json.loads(json.dumps(get_band_evidence("high"), ensure_ascii=False))["band"] == "high"
