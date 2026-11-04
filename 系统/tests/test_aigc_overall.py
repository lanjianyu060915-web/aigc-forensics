"""通用 AIGC 分支和人脸分支并入鉴伪结果的规则。"""

from __future__ import annotations

import pytest

from decision.aigc_overall import attach_general_aigc, decide_overall


def _whole_image(band: str | None = "low", status: str = "success") -> dict:
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


def _general(
    band: str | None = "high",
    status: str = "success",
    face: dict | None = None,
) -> dict:
    payload = {
        "status": status,
        "score": 0.9,
        "logit": 2.2,
        "band": band,
        "band_label": {"high": "高风险提示", "medium": "需人工复核", "low": "证据不足", None: "未标定"}[band],
        "band_reason": "理由",
        "band_thresholds": {"low_max": 0.7, "medium_max": 1.5},
        "error_message": None if status == "success" else "解码失败",
    }
    if face is not None:
        payload["face_branch"] = face
    return payload


def _face(band: str | None) -> dict:
    return {"status": "success", "logit": -4.0, "band": band, "band_label": "人脸档位"}


@pytest.mark.parametrize(
    ("general", "face", "whole_image", "expected"),
    [
        # 通用分支定主结论
        ("high", None, "low", "high"),
        ("medium", None, "low", "medium"),
        ("low", None, "low", "low"),
        ("low", "low", "low", "low"),
        ("medium", "low", "low", "medium"),
        ("high", "low", "low", "high"),
        # 人脸分支最多把档位抬一级
        ("low", "medium", "low", "medium"),
        ("low", "high", "low", "medium"),
        ("medium", "medium", "low", "medium"),
        ("medium", "high", "low", "high"),
        ("high", "high", "low", "high"),
        # 整图分支不影响档位
        ("low", None, "high", "low"),
        ("low", "low", "high", "low"),
        ("medium", None, "high", "medium"),
        # 通用分支不可用时，单路高风险封顶
        (None, "high", "low", "medium"),
        (None, "medium", None, "medium"),
        (None, "low", "high", "low"),
        (None, None, "high", "medium"),
        (None, None, "medium", "medium"),
        (None, None, "low", "low"),
        ("high", None, None, "high"),
        (None, None, None, None),
    ],
)
def test_overall_band_rule(general, face, whole_image, expected) -> None:
    assert decide_overall(general, face, whole_image)[0] == expected


def test_whole_image_warning_is_mentioned_without_changing_the_band() -> None:
    band, reason = decide_overall("low", None, "high")
    assert band == "low"
    assert "只作提示" in reason


def test_full_image_branch_alone_never_reaches_high() -> None:
    merged = attach_general_aigc(_whole_image("high"), _general(status="error", band=None))
    assert merged["overall"]["band"] == "medium"
    assert merged["status"] == "success"
    assert any("通用 AIGC 分支未完成" in text for text in merged["warnings"])


def test_original_fields_are_kept_and_input_is_not_mutated() -> None:
    whole_image = _whole_image("low")
    merged = attach_general_aigc(whole_image, _general("high"))
    assert merged["score"] == 0.1
    assert merged["details"]["band"] == "low"
    assert merged["general_aigc"]["logit"] == 2.2
    assert merged["overall"]["band"] == "high"
    assert [item["key"] for item in merged["overall"]["branches"]] == [
        "general_aigc",
        "face",
        "full_image",
    ]
    assert "overall" not in whole_image
    assert whole_image["warnings"] == ["原有告警"]


def test_whole_image_failure_is_reported_but_task_keeps_general_result() -> None:
    merged = attach_general_aigc(_whole_image(None, status="error"), _general("medium"))
    assert merged["status"] == "success"
    assert merged["error_message"] is None
    assert merged["legacy_error_message"] == "模型缺失"
    assert merged["score"] == 0.9
    assert merged["details"]["band"] == "medium"
    assert merged["overall"]["band"] == "medium"
    assert any("整图分支未完成" in text for text in merged["warnings"])


def test_both_failed_stays_an_error() -> None:
    merged = attach_general_aigc(_whole_image(None, status="error"), _general(None, status="error"))
    assert merged["status"] == "error"
    assert merged["overall"]["band"] is None
    assert merged["overall"]["band_label"] == "未标定"


def test_uncalibrated_general_does_not_invent_a_band() -> None:
    merged = attach_general_aigc(_whole_image("low"), _general(None))
    assert merged["overall"]["band"] == "low"
    assert merged["overall"]["branches"][0]["band"] is None
    assert merged["overall"]["branches"][0]["band_label"] == "未标定"


def test_face_branch_raises_the_band_and_is_listed() -> None:
    merged = attach_general_aigc(_whole_image("low"), _general("low", face=_face("high")))
    assert merged["overall"]["band"] == "medium"
    face = merged["overall"]["branches"][1]
    assert face["status"] == "success"
    assert face["band"] == "high"
    assert face["score"] == -4.0


@pytest.mark.parametrize(
    ("face", "status"),
    [
        ({"status": "no_face", "logit": None, "band": None, "band_label": "未检出人脸"}, "no_face"),
        ({"status": "error", "logit": None, "band": None, "error_message": "坏图"}, "error"),
        (None, "skipped"),
    ],
)
def test_face_branch_without_a_band_does_not_change_the_result(face, status) -> None:
    merged = attach_general_aigc(_whole_image("low"), _general("low", face=face))
    assert merged["overall"]["band"] == "low"
    branch = merged["overall"]["branches"][1]
    assert branch["status"] == status
    assert branch["band"] is None
