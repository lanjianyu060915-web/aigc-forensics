# -*- coding: utf-8 -*-
"""M4 视频入口的纯逻辑回归测试（关键帧选择 + 多帧聚合）。

这些用例不加载模型、不解码真实视频：selection 与 aggregate 两层都是确定性纯函数，
测试只验证「规则可复现、证据字段可追溯、失败路径可解释」。真实视频端到端结果与
耗时放在 experiments/m4_video/results/，不在本文件覆盖。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video_pipeline.aggregate import (  # noqa: E402
    FROZEN_BRANCH_WARNING,
    aggregate_deepfake_frames,
    aggregate_identity_frames,
    annotate_single_frame,
    build_video_summary,
    rebase_frame_artifacts,
)
from video_pipeline.selection import (  # noqa: E402
    candidate_indices,
    percentile,
    select_key_frames,
)


BAND_LABELS = {"high": "高风险提示", "medium": "需人工复核", "low": "证据不足"}


def _candidate_frame(index, timestamp, *, sharpness, brightness=120.0, face_count=1):
    return {
        "frame_index": index,
        "timestamp_seconds": timestamp,
        "sharpness": sharpness,
        "brightness": brightness,
        "face_count": face_count,
    }


def _deepfake_payload(score, *, band=None, is_deepfake=None, frozen=False, limitations=None):
    details = {"band_thresholds": {"low_max": 0.5, "medium_max": 0.7}}
    if band is not None:
        details["band"] = band
        details["band_label"] = BAND_LABELS[band]
    if frozen:
        details["native_status"] = "SUCCESS"
        details["media_type"] = "image"
        details["fusion_score"] = score
    method = {"name": "多域融合单图深伪风险甄别" if frozen else "二代整图深伪风险甄别（ViT 分支）"}
    if not frozen:
        method["branch"] = "vit_full_image"
    return {
        "schema_version": "3.0",
        "module": "deepfake",
        "status": "success",
        "score": score,
        "is_deepfake": is_deepfake if is_deepfake is not None else band == "high",
        "details": details,
        "method": method,
        "limitations": list(limitations or []),
        "runtime_seconds": 0.5,
        "provenance": {"input_sha256": "a" * 64},
        "artifact_manifest": [],
    }


def _identity_candidate(identity_id, score):
    return {
        "rank": 1,
        "identity_id": identity_id,
        "identity_name": identity_id,
        "score": score,
        "mean_similarity": score,
        "reference_count": 5,
    }


def _identity_payload(candidates):
    return {
        "schema_version": "3.0",
        "module": "identity",
        "status": "success",
        "candidates": candidates,
        "metrics": {"query_face_count": 1, "top1_top2_margin": 0.05},
        "method": {"name": "insightface_closed_set_person_mean_cosine"},
        "limitations": ["身份检索为封闭集排序，不是身份确认。"],
        "warnings": [],
        "runtime_seconds": 1.5,
        "provenance": {},
    }


def _frame_entry(rank, payload, *, timestamp=None, error=None):
    return {
        "frame": {
            "frame_rank": rank,
            "file_name": f"frame_{rank:03d}.jpg",
            "timestamp_seconds": timestamp if timestamp is not None else 0.4 * rank,
            "frame_index": rank * 10,
        },
        "payload": payload,
        "error": error,
    }


def test_candidate_indices_are_deterministic_and_trimmed():
    first = candidate_indices(150, 24)
    second = candidate_indices(150, 24)

    assert first == second
    assert len(first) == 24
    assert first == sorted(set(first))
    assert first[0] >= int(150 * 0.05)
    assert first[-1] <= 150 - int(150 * 0.05)
    with pytest.raises(ValueError):
        candidate_indices(0, 24)


def test_percentile_matches_linear_interpolation():
    assert percentile([0.0, 10.0], 50) == pytest.approx(5.0)
    assert percentile([4.0], 95) == pytest.approx(4.0)
    assert percentile([], 90) == 0.0


def test_select_key_frames_prefers_single_face_and_keeps_time_gap():
    candidates = [
        _candidate_frame(0, 0.0, sharpness=1000.0, face_count=2),
        _candidate_frame(1, 0.1, sharpness=900.0),
        _candidate_frame(2, 0.2, sharpness=800.0),
        _candidate_frame(3, 3.0, sharpness=700.0),
        _candidate_frame(4, 6.0, sharpness=600.0),
    ]

    result = select_key_frames(candidates, max_frames=2, duration_seconds=6.0)
    selected = result["key_frames"]

    assert len(selected) == 2
    assert all(item["face_count"] == 1 for item in selected)
    assert selected[0]["quality_score"] >= selected[1]["quality_score"]
    gap = abs(selected[0]["timestamp_seconds"] - selected[1]["timestamp_seconds"])
    assert gap >= result["policy"]["min_temporal_gap_seconds"]
    assert "单人脸帧优先（身份检索要求恰好一张人脸）" in selected[0]["selection_reasons"]
    assert result["policy"]["selected_frames"] == 2
    assert len(result["candidates"]) == 5
    assert any(item["selected"] is False for item in result["candidates"])


def test_select_key_frames_validates_max_frames_and_empty_input():
    with pytest.raises(ValueError):
        select_key_frames([], max_frames=0)

    empty = select_key_frames([], max_frames=3)
    assert empty["key_frames"] == []
    assert empty["policy"]["empty_reason"] == "没有候选帧"


def test_video_deepfake_uses_calibrated_bands_when_available():
    frames = [
        _frame_entry(1, _deepfake_payload(0.9, band="high")),
        _frame_entry(2, _deepfake_payload(0.8, band="high")),
        _frame_entry(3, _deepfake_payload(0.2, band="low")),
        _frame_entry(4, _deepfake_payload(0.85, band="high")),
    ]

    payload = aggregate_deepfake_frames(frames, query_image="query.jpg")
    details = payload["details"]

    assert payload["status"] == "success"
    assert payload["score"] == pytest.approx(0.825)
    assert payload["is_deepfake"] is True
    assert details["band"] == "high"
    assert details["band_label"] == "高风险提示"
    assert details["band_rule"] == "calibrated_band"
    assert details["frame_band_sources"] == {
        "calibrated_band": 4,
        "frozen_threshold": 0,
        "unavailable": 0,
    }
    assert details["frame_branches"] == {"vit_full_image": 4}
    assert details["per_frame"][0]["band_source"] == "calibrated_band"
    assert details["per_frame"][0]["branch"] == "vit_full_image"
    assert payload["warnings"] == []


def test_video_deepfake_falls_back_to_frozen_threshold_band():
    frame_limitation = "模型输出为深伪风险筛查分数，不是系统准确率或司法鉴定结论。"
    frames = [
        _frame_entry(
            rank,
            _deepfake_payload(
                0.24,
                frozen=True,
                is_deepfake=False,
                limitations=[frame_limitation],
            ),
        )
        for rank in (1, 2, 3, 4)
    ]

    payload = aggregate_deepfake_frames(frames, query_image="query.jpg")
    details = payload["details"]

    assert payload["is_deepfake"] is False
    assert details["band"] == "low"
    assert details["band_label"] == "证据不足"
    assert details["band_rule"] == "frozen_threshold"
    assert details["frame_band_counts"]["low"] == 4
    assert details["frame_band_sources"] == {
        "calibrated_band": 0,
        "frozen_threshold": 4,
        "unavailable": 0,
    }
    assert details["frame_branches"] == {"frozen_v040_spatial_frequency_fusion": 4}
    assert "冻结" in details["band_reason"]
    assert details["per_frame"][0]["band_source"] == "frozen_threshold"
    assert details["per_frame"][0]["band_label"] == "证据不足"
    assert "占位结果" in payload["warnings"][0]
    assert FROZEN_BRANCH_WARNING.split("{")[0] in payload["warnings"][0]
    # 逐帧 limitations 必须整句保留，不能被按字符拆开
    assert frame_limitation in payload["limitations"]
    assert all(len(item) > 1 for item in payload["limitations"])


def test_video_deepfake_marks_mixed_band_sources():
    frames = [
        _frame_entry(1, _deepfake_payload(0.9, band="high")),
        _frame_entry(2, _deepfake_payload(0.1, frozen=True, is_deepfake=False)),
    ]

    payload = aggregate_deepfake_frames(frames, query_image="query.jpg")
    details = payload["details"]

    assert details["band_rule"] == "mixed"
    assert details["band"] == "medium"
    assert details["frame_band_sources"]["calibrated_band"] == 1
    assert details["frame_band_sources"]["frozen_threshold"] == 1
    assert "冻结" not in details["band_reason"]


def test_video_deepfake_reports_when_every_frame_fails():
    frames = [
        _frame_entry(1, None, error="RuntimeError: boom"),
        _frame_entry(2, {"status": "error", "error_message": "未检测到人脸", "runtime_seconds": 0.1}),
    ]

    payload = aggregate_deepfake_frames(frames, query_image="query.jpg")

    assert payload["status"] == "error"
    assert payload["score"] is None
    assert payload["metrics"]["frames_total"] == 2
    assert payload["metrics"]["frames_scored"] == 0
    assert payload["details"]["frame_failures"][0]["error_message"] == "RuntimeError: boom"
    assert payload["details"]["frame_failures"][1]["error_message"] == "未检测到人脸"


def test_video_deepfake_uses_video_calibration_thresholds_when_supplied():
    frames = [
        _frame_entry(1, _deepfake_payload(0.9, band="high")),
        _frame_entry(2, _deepfake_payload(0.8, band="high")),
        _frame_entry(3, _deepfake_payload(0.2, band="low")),
        _frame_entry(4, _deepfake_payload(0.85, band="high")),
    ]

    payload = aggregate_deepfake_frames(
        frames,
        query_image="query.jpg",
        video_band_thresholds={"low_max": 0.6, "medium_max": 0.7},
    )
    details = payload["details"]

    # 中位分 0.825 高于视频校准上界 0.7，因此视频级结论由中位分给出。
    assert details["band"] == "high"
    assert details["video_band_rule"] == "median_video_thresholds"
    assert details["video_band_thresholds"] == {
        "statistic": "median",
        "low_max": 0.6,
        "medium_max": 0.7,
    }
    assert "中位分" in details["band_reason"]
    assert "视频校准集" in details["band_reason"]
    assert details["frame_band_counts"]["high"] == 3
    assert payload["provenance"]["aggregation_rule"] == "median_video_thresholds"
    assert payload["method"]["fusion_method"].startswith("帧级分数取中位数")
    assert "对照独立视频校准集" in payload["method"]["fusion_method"]


def test_video_deepfake_video_thresholds_override_frame_majority():
    frames = [
        _frame_entry(1, _deepfake_payload(0.9, band="high")),
        _frame_entry(2, _deepfake_payload(0.1, band="low")),
        _frame_entry(3, _deepfake_payload(0.1, band="low")),
        _frame_entry(4, _deepfake_payload(0.1, band="low")),
    ]

    majority = aggregate_deepfake_frames(frames, query_image="query.jpg")
    calibrated = aggregate_deepfake_frames(
        frames,
        query_image="query.jpg",
        video_band_thresholds={"low_max": 0.5, "medium_max": 0.7},
    )

    # 一颗高风险帧会把帧级一致性规则抬到需复核；校准口径只看中位分 0.1。
    assert majority["details"]["band"] == "medium"
    assert calibrated["details"]["band"] == "low"
    assert calibrated["details"]["video_band_rule"] == "median_video_thresholds"


def test_video_deepfake_falls_back_when_video_thresholds_are_unusable():
    frames = [
        _frame_entry(1, _deepfake_payload(0.9, band="high")),
        _frame_entry(2, _deepfake_payload(0.8, band="high")),
    ]

    for thresholds in (
        None,
        {},
        {"low_max": 0.7, "medium_max": 0.6},
        {"low_max": 0.6},
        {"low_max": "x", "medium_max": 0.6},
    ):
        payload = aggregate_deepfake_frames(
            frames, query_image="query.jpg", video_band_thresholds=thresholds
        )
        assert payload["details"]["video_band_rule"] == "band_majority"
        assert payload["details"]["video_band_thresholds"] == {}
        assert payload["details"]["band"] == "high"


def test_video_deepfake_ignores_video_thresholds_for_frozen_frames():
    frames = [
        _frame_entry(1, _deepfake_payload(0.9, frozen=True, is_deepfake=True)),
        _frame_entry(2, _deepfake_payload(0.1, frozen=True, is_deepfake=False)),
    ]

    payload = aggregate_deepfake_frames(
        frames,
        query_image="query.jpg",
        video_band_thresholds={"low_max": 0.6, "medium_max": 0.7},
    )

    # 冻结内核没有校准分带，不许拿视频校准阈值去解释它的二值判定。
    assert payload["details"]["video_band_rule"] == "band_majority"
    assert payload["details"]["video_band_thresholds"] == {}
    assert payload["details"]["band"] == "medium"


def test_video_deepfake_limitations_state_measured_cross_corpus_risk():
    frames = [_frame_entry(1, _deepfake_payload(0.9, band="high"))]

    payload = aggregate_deepfake_frames(frames, query_image="query.jpg")
    text = "".join(payload["limitations"])

    assert "53.3%" in text
    assert "70.9%" in text


def test_rebase_rewrites_only_task_local_relative_paths(tmp_path: Path):
    frame_dir = tmp_path / "frame_results" / "source"
    (frame_dir / "visualizations").mkdir(parents=True)
    (frame_dir / "visualizations" / "match.jpg").write_bytes(b"x")
    payload = {
        "file_path": "D:/library/164.jpg",
        "relative": "visualizations/match.jpg",
        "missing": "visualizations/absent.jpg",
        "already": "frame_results/source/visualizations/match.jpg",
        "mixed": ["D:\\library\\a.jpg", "visualizations/match.jpg"],
        "count": 3,
    }

    output = rebase_frame_artifacts(payload, frame_dir=frame_dir, prefix="frame_results/source")

    assert output["file_path"] == "D:/library/164.jpg"
    assert output["relative"] == "frame_results/source/visualizations/match.jpg"
    assert output["missing"] == "visualizations/absent.jpg"
    assert output["already"] == "frame_results/source/visualizations/match.jpg"
    assert output["mixed"][0] == "D:\\library\\a.jpg"
    assert output["mixed"][1] == "frame_results/source/visualizations/match.jpg"
    assert output["count"] == 3


def test_video_identity_ranks_by_mean_similarity_across_frames():
    frames = [
        _frame_entry(
            1,
            _identity_payload(
                [_identity_candidate("person_a", 0.9), _identity_candidate("person_b", 0.4)]
            ),
        ),
        _frame_entry(
            2,
            _identity_payload(
                [_identity_candidate("person_a", 0.5), _identity_candidate("person_b", 0.95)]
            ),
        ),
        _frame_entry(3, None, error="NoFaceError: 未检出人脸"),
    ]

    payload = aggregate_identity_frames(
        frames,
        query_image="query.jpg",
        extra_limitations=["本模块视频模式只使用 3 个关键帧。"],
    )
    top = payload["candidates"][0]

    assert payload["status"] == "success"
    assert payload["metrics"]["frames_analyzed"] == 3
    assert payload["metrics"]["frames_with_faces"] == 2
    assert top["identity_id"] == "person_a"
    assert top["mean_across_frames"] == pytest.approx(0.7)
    assert top["frame_support"] == 2
    assert [item["frame_rank"] for item in top["frame_scores"]] == [1, 2]
    assert payload["candidates"][1]["identity_id"] == "person_b"
    assert payload["metrics"]["top1_top2_margin"] == pytest.approx(0.025)
    assert payload["warnings"][0].startswith("以下关键帧没有参与身份聚合")
    assert "本模块视频模式只使用 3 个关键帧。" in payload["limitations"]


def test_video_identity_without_any_success_reports_error():
    frames = [
        _frame_entry(1, None, error="RuntimeError: boom"),
        _frame_entry(2, {"status": "error", "error_message": "未检出人脸", "runtime_seconds": 0.2}),
    ]

    payload = aggregate_identity_frames(frames, query_image="query.jpg")

    assert payload["status"] == "error"
    assert payload["candidates"] == []
    assert payload["metrics"]["frames_with_faces"] == 0
    assert payload["details"]["video_aggregation"]["rule"] == "mean_across_frames"


def test_annotate_single_frame_marks_video_role_and_frame():
    payload = {
        "status": "success",
        "candidates": [],
        "limitations": ["底图检索为开放集候选排序。"],
        "runtime": {},
    }

    output = annotate_single_frame(
        payload,
        frame={"frame_rank": 1, "file_name": "frame_001.jpg", "timestamp_seconds": 1.2},
        role="底图检索（视频首要帧）",
    )

    assertion = output["details"]["video_aggregation"]
    assert assertion["rule"] == "primary_frame"
    assert assertion["role"] == "底图检索（视频首要帧）"
    assert assertion["frame"]["file_name"] == "frame_001.jpg"
    assert output["frame_evidence"][0]["status"] == "success"
    assert output["runtime"]["frames_analyzed"] == 1
    assert "底图检索为开放集候选排序。" in output["limitations"]


def test_build_video_summary_reads_runner_policy_and_module_fields():
    selection = {
        "video_sha256": "b" * 64,
        "video": {"file_name": "query_video.mp4", "fps": 30.0},
        "selection": {"rule": "按质量分降序贪心选帧", "selected_frames": 1},
        "key_frames": [
            {
                "frame_rank": 1,
                "file_name": "frame_001.jpg",
                "timestamp_seconds": 1.0,
                "quality_score": 0.9,
                "face_count": 1,
                "selection_reasons": ["单人脸帧优先（身份检索要求恰好一张人脸）"],
            }
        ],
    }
    deepfake = aggregate_deepfake_frames(
        [_frame_entry(1, _deepfake_payload(0.9, band="high"))], query_image="query.jpg"
    )
    identity = aggregate_identity_frames(
        [_frame_entry(1, _identity_payload([_identity_candidate("person_a", 0.6)]))],
        query_image="query.jpg",
    )
    source = {
        "status": "success",
        "candidates": [
            {
                "rank": 1,
                "file_path": "D:/library/source_library/164.jpg",
                "global_score": 0.51,
                "local_score": 0.64,
                "final_score": 0.58,
            }
        ],
    }

    summary = build_video_summary(
        selection=selection,
        module_payloads={"deepfake": deepfake, "identity": identity, "source": source},
        input_file="query_video.mp4",
        sha256=selection["video_sha256"],
    )

    assert summary["input_type"] == "video"
    assert summary["video_sha256"] == "b" * 64
    assert summary["frame_selection"]["selected_frames"] == 1
    assert summary["key_frames"][0]["file_name"] == "frame_001.jpg"
    assert summary["modules"]["deepfake"]["band"] == "high"
    assert summary["modules"]["deepfake"]["band_rule"] == "calibrated_band"
    assert summary["modules"]["identity"]["top1_frame_support"] == 1
    assert summary["modules"]["source"]["top1_candidate"] == "164.jpg"
    assert summary["modules"]["source"]["top1_score"] == pytest.approx(0.58)
    assert summary["modules"]["source"]["top1_global_score"] == pytest.approx(0.51)


def test_build_video_summary_tolerates_legacy_policy_key():
    summary = build_video_summary(
        selection={"policy": {"selected_frames": 0}},
        module_payloads={},
        input_file="query_video.mp4",
    )

    assert summary["frame_selection"] == {"selected_frames": 0}
    assert summary["modules"] == {}
    assert summary["key_frames"] == []
