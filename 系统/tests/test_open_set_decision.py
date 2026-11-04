from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision import open_set  # noqa: E402
from decision.evidence_matrix import SCHEMA_VERSION, build_evidence_matrix  # noqa: E402
from decision.open_set import (  # noqa: E402
    STATE_HIGH,
    STATE_INSUFFICIENT,
    STATE_NONE,
    STATE_UNAVAILABLE,
    decide_identity_open_set,
    decide_source_open_set,
    load_thresholds,
)
from reporting.explainability import build_explanations  # noqa: E402
from reporting.report_builder import (  # noqa: E402
    build_final_report,
    render_text_report,
)


CALIBRATED_FILE = (
    ROOT / "packaging_v0.4" / "resources" / "decision" / "open_set_thresholds.json"
)
CALIBRATED = load_thresholds(CALIBRATED_FILE)

FORBIDDEN_FUSION_KEYS = {
    "overall_score",
    "overall_confidence",
    "composite_score",
    "combined_score",
    "fusion_probability",
    "total_confidence",
    "final_probability",
}


def source_payload(score: float | None, *, status: str = "success", **metrics) -> dict:
    candidates = []
    if score is not None:
        candidates.append(
            {
                "rank": 1,
                "file_path": r"D:\source_library\006.jpg",
                "global_score": score,
                "local_score": 0.920812,
                "final_score": 0.744832,
                "ransac_inlier_count": 171,
                "ransac_inlier_ratio": 0.868,
                "ransac_applied": True,
                "good_match_count": 197,
            }
        )
    payload = {
        "schema_version": "3.0",
        "module": "source",
        "status": status,
        "query_image": r"D:\cases\query.jpg",
        "candidates": candidates,
        "runtime_seconds": 1.0,
        "error_message": None if status == "success" else "检索失败",
        "method": {"name": "sift_ransac"},
        "metrics": {
            "library_decoded": 203,
            "returned_candidate_count": len(candidates),
            "top1_top2_margin": 0.011282,
            **metrics,
        },
    }
    return payload


def identity_payload(
    score: float | None,
    *,
    status: str = "success",
    margin: float | None = 0.5,
    **metrics,
) -> dict:
    candidates = []
    if score is not None:
        candidates.append(
            {
                "rank": 1,
                "identity_id": "person_003",
                "identity_name": "person_003",
                "score": score,
                "mean_similarity": score,
                "best_reference_score": score + 0.02,
                "reference_count": 4,
                "valid_reference_count": 4,
            }
        )
    payload = {
        "schema_version": "3.0",
        "status": status,
        "query_image": r"D:\cases\query.jpg",
        "candidates": candidates,
        "runtime_seconds": 1.0,
        "error_message": None if status == "success" else "未检出人脸",
        "method": {"model_name": "InsightFace buffalo_l"},
        "metrics": {
            "query_face_count": 1,
            "face_detection_score": 0.795726,
            "library_identity_count": 9,
            "library_image_count": 45,
            "valid_embedding_count": 43,
            "top1_top2_margin": margin,
            **metrics,
        },
    }
    return payload


def deepfake_vit_payload(*, band: str | None, label: str | None, branch: str = "vit_full_image") -> dict:
    details: dict = {"branch": branch, "input_short_edge_px": 720}
    if band is not None:
        details["band"] = band
    if label is not None:
        details["band_label"] = label
    return {
        "schema_version": "3.0",
        "module": "deepfake",
        "status": "success",
        "runtime_seconds": 2.0,
        "score": 0.005633,
        "details": details,
    }


class TestSourceOpenSet:
    @pytest.mark.parametrize(
        "score, expected",
        [
            (0.996107, STATE_HIGH),
            (0.70, STATE_INSUFFICIENT),
            (0.529744, STATE_NONE),
        ],
    )
    def test_score_bands_follow_calibrated_thresholds(self, score: float, expected: str) -> None:
        verdict = decide_source_open_set(source_payload(score), CALIBRATED)
        assert verdict["state"] == expected
        assert verdict["state_label"]
        assert verdict["thresholds"]["config_origin"] == "explicit"

    def test_failed_search_is_unavailable_not_no_candidate(self) -> None:
        verdict = decide_source_open_set(source_payload(None, status="error"), CALIBRATED)
        assert verdict["state"] == STATE_UNAVAILABLE
        assert verdict["decidable"] is False
        texts = verdict["reasons"] + verdict["caveats"]
        assert any("不能" in text or "未成功" in text for text in texts)

    def test_geometry_signals_do_not_change_state(self) -> None:
        low_inliers = decide_source_open_set(
            source_payload(0.996107, ransac_inlier_count=4), CALIBRATED
        )
        high_inliers = decide_source_open_set(
            source_payload(0.996107, ransac_inlier_count=1459), CALIBRATED
        )
        assert low_inliers["state"] == high_inliers["state"] == STATE_HIGH


class TestIdentityOpenSet:
    @pytest.mark.parametrize(
        "score, margin, expected",
        [
            (0.593203, 0.5, STATE_HIGH),
            (0.30, 0.5, STATE_INSUFFICIENT),
            (0.593203, 0.02, STATE_INSUFFICIENT),
            (0.088423, 0.0193, STATE_NONE),
        ],
    )
    def test_score_and_margin_bands(self, score: float, margin: float, expected: str) -> None:
        verdict = decide_identity_open_set(
            identity_payload(score, margin=margin), CALIBRATED
        )
        assert verdict["state"] == expected

    def test_high_score_with_small_margin_is_downgraded_with_reason(self) -> None:
        verdict = decide_identity_open_set(identity_payload(0.60, margin=0.02), CALIBRATED)
        assert verdict["state"] == STATE_INSUFFICIENT
        assert any("间隔" in reason for reason in verdict["reasons"])

    def test_missing_face_is_unavailable(self) -> None:
        verdict = decide_identity_open_set(
            identity_payload(None, status="error", query_face_count=0, face_detection_score=None),
            CALIBRATED,
        )
        assert verdict["state"] == STATE_UNAVAILABLE
        assert "0" in verdict["reasons"][0]

    def test_empty_library_is_unavailable(self) -> None:
        verdict = decide_identity_open_set(
            identity_payload(None, library_identity_count=0, valid_embedding_count=0),
            CALIBRATED,
        )
        assert verdict["state"] == STATE_UNAVAILABLE
        assert any("库覆盖不足" in caveat for caveat in verdict["caveats"])


class TestThresholdResolution:
    def test_packaged_calibration_is_resolved_by_default(self) -> None:
        resolved = open_set.resolve_thresholds_path()
        assert resolved is not None
        assert resolved.name == CALIBRATED_FILE.name
        assert open_set.active_thresholds()["source_version"] == CALIBRATED["source_version"]

    def test_missing_file_falls_back_to_placeholder(self, tmp_path: Path) -> None:
        fallback = load_thresholds(tmp_path / "absent.json")
        assert fallback["source_version"] == "uncalibrated-default"
        assert fallback["modules"] == open_set.DEFAULT_THRESHOLDS["modules"]

    def test_placeholder_run_is_flagged_uncalibrated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(open_set, "resolve_thresholds_path", lambda: None)
        verdict = decide_source_open_set(source_payload(0.996107))
        assert verdict["thresholds"]["config_origin"] == "builtin-default"
        assert open_set.UNCALIBRATED_CAVEAT in verdict["caveats"]

    def test_env_override_takes_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        override = tmp_path / "override.json"
        override.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "source_version": "unit-test-override",
                    "modules": {"source": {"high_global_score": 0.9, "low_global_score": 0.8}},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv(open_set.ENV_THRESHOLDS_PATH, str(override))
        assert open_set.resolve_thresholds_path() == override
        verdict = decide_source_open_set(source_payload(0.85))
        assert verdict["state"] == STATE_INSUFFICIENT
        assert verdict["thresholds"]["config_version"] == "unit-test-override"


class TestEvidenceMatrix:
    def _matrix(self) -> dict:
        return build_evidence_matrix(
            deepfake_vit_payload(band="high", label="高风险提示"),
            identity_payload(0.088423, margin=0.0193),
            source_payload(0.529744),
            thresholds=CALIBRATED,
        )

    def test_matrix_structure(self) -> None:
        matrix = self._matrix()
        assert matrix["schema_version"] == SCHEMA_VERSION
        assert set(matrix) >= {
            "modules",
            "input_quality",
            "library_coverage",
            "geometry",
            "open_set_risk",
            "notes",
        }
        assert set(matrix["modules"]) == {"deepfake", "identity", "source"}
        for block in matrix["modules"].values():
            assert "grade" in block and "grade_label" in block and "primary_label" in block

    def test_matrix_never_fuses_module_scores(self) -> None:
        payload = json.dumps(self._matrix(), ensure_ascii=False)
        for forbidden in FORBIDDEN_FUSION_KEYS:
            assert forbidden not in payload
        assert "综合概率" in json.dumps(self._matrix()["notes"], ensure_ascii=False)

    def test_out_of_library_rows_are_graded_low(self) -> None:
        matrix = self._matrix()
        assert matrix["modules"]["identity"]["grade"] == "low"
        assert matrix["modules"]["source"]["grade"] == "low"
        assert matrix["modules"]["identity"]["grade_label"] == "当前库中未发现可靠候选"
        assert matrix["open_set_risk"]["level"] == "存在"

    def test_v040_fallback_branch_is_not_graded_as_band_evidence(self) -> None:
        matrix = build_evidence_matrix(
            deepfake_vit_payload(band="high", label="高风险提示", branch="v040_face_crop"),
            identity_payload(0.088423),
            source_payload(0.529744),
            thresholds=CALIBRATED,
        )
        assert matrix["modules"]["deepfake"]["grade"] == "unavailable"
        assert "冻结人脸裁剪分支" in matrix["modules"]["deepfake"]["text"]

    def test_legacy_fusion_result_has_no_band(self) -> None:
        legacy = {
            "schema_version": "3.0",
            "module": "deepfake",
            "status": "success",
            "score": 0.293447,
            "details": {},
        }
        matrix = build_evidence_matrix(
            legacy, identity_payload(0.088423), source_payload(0.529744), thresholds=CALIBRATED
        )
        assert matrix["modules"]["deepfake"]["grade_label"] == "旧版判定口径"

    def test_geometry_grade_is_descriptive_only(self) -> None:
        matrix = self._matrix()
        geometry = matrix["geometry"]
        assert geometry["grade"] in {"强", "中", "弱", "不适用"}
        assert "不参与来源判定" in geometry["note"]


class TestReportSection:
    def test_report_numbers_matrix_before_evidence_summary(self) -> None:
        report = build_final_report(
            "unit_matrix",
            deepfake_result=deepfake_vit_payload(band="high", label="高风险提示"),
            identity_result=identity_payload(0.593203, margin=0.5),
            source_result=source_payload(0.996107),
        )
        text = render_text_report(report)
        matrix_at = text.index("五、证据质量矩阵（研判卡）")
        summary_at = text.index("六、并列证据总结")
        assert matrix_at < summary_at
        section = text[matrix_at:summary_at]
        for label in ("AIGC鉴伪", "人脸身份检索", "原始底图溯源", "开放集风险"):
            assert label in section
        assert "综合可信度" not in text

    def test_explanations_carry_open_set_verdict_without_claiming_identity(self) -> None:
        identity = identity_payload(0.088423, margin=0.0193)
        explanations = build_explanations({}, identity, source_payload(0.529744))
        decision = explanations["identity"]["technical"]["open_set"]
        assert decision["state"] == STATE_NONE
        assert any("开放集判定依据" in line for line in explanations["identity"]["basis"])
        limitations = " ".join(explanations["identity"]["limitations"])
        assert "不能表述为已确认身份" in limitations
        source_limitations = " ".join(explanations["source"]["limitations"])
        assert "不能表述为已确认原始底图" in source_limitations
