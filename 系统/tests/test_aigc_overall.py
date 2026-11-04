"""通用 AIGC 分支并入鉴伪结果的规则。"""

from __future__ import annotations

import pytest

from decision.aigc_overall import attach_general_aigc, decide_overall


def _legacy(band: str | None = "low", status: str = "success") -> dict:
    return {
        "status": status,
        "query_image": "query.jpg",
        "is_deepfake": False,
        "score": 0.1,
        "label": "real",
        "runtime_seconds": 1.0,
        "details": {"band": band, "band_label": "证据不足"} if band else {},
        "error_message": None if status == "success" else "模型缺失",
        "warnings": ["原有告警"],
    }


def _general(band: str | None = "high", status: str = "success") -> dict:
    return {
        "status": status,
        "score": 0.9,
        "logit": 2.2,
        "band": band,
        "band_label": {"high": "高风险提示", "medium": "需人工复核", "low": "证据不足", None: "未标定"}[band],
        "band_reason": "理由",
        "band_thresholds": {"low_max": 0.7, "medium_max": 1.5},
        "error_message": None if status == "success" else "解码失败",
    }


@pytest.mark.parametrize(
    ("general", "legacy", "expected"),
    [
        ("high", "low", "high"),
        ("high", "high", "high"),
        ("medium", "low", "medium"),
        ("low", "low", "low"),
        ("low", "medium", "medium"),
        ("low", "high", "medium"),
        ("medium", "high", "medium"),
        (None, "high", "medium"),
        (None, "medium", "medium"),
        (None, "low", "low"),
        ("high", None, "high"),
        (None, None, None),
    ],
)
def test_overall_band_rule(general, legacy, expected) -> None:
    assert decide_overall(general, legacy)[0] == expected


def test_full_image_branch_alone_never_reaches_high() -> None:
    merged = attach_general_aigc(_legacy("high"), _general(status="error", band=None))
    assert merged["overall"]["band"] == "medium"
    assert merged["status"] == "success"
    assert any("通用 AIGC 分支未完成" in text for text in merged["warnings"])


def test_original_fields_are_kept_and_input_is_not_mutated() -> None:
    legacy = _legacy("low")
    merged = attach_general_aigc(legacy, _general("high"))
    assert merged["score"] == 0.1
    assert merged["details"]["band"] == "low"
    assert merged["general_aigc"]["logit"] == 2.2
    assert merged["overall"]["band"] == "high"
    assert [item["key"] for item in merged["overall"]["branches"]] == [
        "general_aigc",
        "full_image_face",
    ]
    assert "overall" not in legacy
    assert legacy["warnings"] == ["原有告警"]


def test_legacy_failure_is_reported_but_task_keeps_general_result() -> None:
    merged = attach_general_aigc(_legacy(None, status="error"), _general("medium"))
    assert merged["status"] == "success"
    assert merged["error_message"] is None
    assert merged["legacy_error_message"] == "模型缺失"
    assert merged["score"] == 0.9
    assert merged["details"]["band"] == "medium"
    assert merged["overall"]["band"] == "medium"
    assert any("整图/人脸分支未完成" in text for text in merged["warnings"])


def test_both_failed_stays_an_error() -> None:
    merged = attach_general_aigc(_legacy(None, status="error"), _general(None, status="error"))
    assert merged["status"] == "error"
    assert merged["overall"]["band"] is None
    assert merged["overall"]["band_label"] == "未标定"


def test_uncalibrated_general_does_not_invent_a_band() -> None:
    merged = attach_general_aigc(_legacy("low"), _general(None))
    assert merged["overall"]["band"] == "low"
    assert merged["overall"]["branches"][0]["band"] is None
    assert merged["overall"]["branches"][0]["band_label"] == "未标定"
