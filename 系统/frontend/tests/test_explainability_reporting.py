from __future__ import annotations

import html
import json
from pathlib import Path
import sys
import tempfile
import unittest


PRODUCT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = PRODUCT_ROOT / "frontend"
for import_root in (PRODUCT_ROOT, FRONTEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from components.data_loader import load_task_results
from components.report_views import ReportViewError, load_report_bundle
from components.result_views import _resolve_deepfake_images
from reporting.explainability import (
    build_deepfake_explanation,
    build_identity_explanation,
    build_source_explanation,
)
from reporting.report_builder import (
    DEFAULT_DISCLAIMER,
    build_final_report,
    generate_task_reports,
    render_html_report,
    render_text_report,
    report_file_names,
)


def deepfake_result(
    *,
    spatial: float = 0.8,
    frequency: float = 0.4,
    threshold: float = 0.5,
    is_deepfake: bool | None = True,
    label: str = "deepfake",
) -> dict:
    fusion = 0.62 * spatial + 0.38 * frequency
    return {
        "schema_version": "3.0",
        "status": "success",
        "query_image": r"D:\案件库\案件甲\query.jpg",
        "is_deepfake": is_deepfake,
        "score": fusion,
        "label": label,
        "runtime_seconds": 1.25,
        "details": {},
        "error_message": None,
        "method": {
            "name": "多域融合单图深伪风险甄别",
            "spatial_model": "MobileNetV3-Small",
            "frequency_model": "FrequencyStatDetector 87维统计特征 + LogisticRegression",
        },
        "metrics": {
            "spatial_score": spatial,
            "frequency_score": frequency,
            "spatial_weight": 0.62,
            "frequency_weight": 0.38,
            "fusion_score": fusion,
            "decision_threshold": threshold,
            "valid_frame_count": 1,
            "media_metadata": {"width": 112, "height": 96, "format": "JPG"},
        },
        "explanation": {},
        "warnings": [],
        "limitations": [],
        "runtime": {"actual_device": "cpu", "elapsed_seconds": 1.25},
        "provenance": {
            "input_sha256": "a" * 64,
            "model_sha256": {"spatial": "b" * 64, "frequency": "c" * 64},
            "fusion_config_sha256": "d" * 64,
            "model_paths": {"spatial": r"C:\Users\example\private\model.pt"},
        },
    }


def kernel_deepfake_result(*, band: str = "high", score: float = 0.031) -> dict:
    band_labels = {"low": "证据不足", "medium": "需人工复核", "high": "高风险提示"}
    return {
        "schema_version": "3.0",
        "status": "success",
        "query_image": r"D:\案件库\案件甲\query.jpg",
        "is_deepfake": band == "high",
        "score": score,
        "label": "fake" if band == "high" else "real",
        "runtime_seconds": 0.18,
        "details": {
            "branch": "vit_full_image",
            "band": band,
            "band_label": band_labels[band],
            "band_reason": "分数超过校准集真人图分布上界，提示高风险，仍需人工复核",
            "band_thresholds": {"low_max": 0.000292, "medium_max": 0.000445},
            "short_edge_px": 512,
            "routing_threshold_px": 256,
            "routing_matched": True,
            "calibration_set": {"name": "SWAN-DF 案例01-10", "n_real": 40, "n_fake": 10, "auc": 0.8425},
            "validation_set": {"name": "SWAN-DF 案例11-20", "auc": 0.9925, "auc_ci95": [0.9697, 1.0]},
        },
        "error_message": None,
        "method": {"name": "二代整图深伪风险甄别（ViT 分支）", "branch": "vit_full_image"},
        "metrics": {},
        "explanation": {},
        "warnings": [],
        "limitations": [],
        "runtime": {"actual_device": "cpu", "elapsed_seconds": 0.18},
        "provenance": {"input_sha256": "a" * 64, "model_sha256": "e" * 64},
    }


def identity_result(*, candidates: bool = True) -> dict:
    rows = []
    if candidates:
        rows = [
            {
                "rank": 1,
                "identity_id": "person_001",
                "identity_name": "张某<script>",
                "score": 0.81,
                "reference_image": r"D:\身份库\person_001\best.jpg",
                "reference_count": 3,
                "mean_similarity": 0.81,
                "best_reference_score": 0.92,
                "best_reference_image": r"D:\身份库\person_001\best.jpg",
                "reference_scores": [
                    {"reference_image": r"D:\身份库\person_001\best.jpg", "similarity": 0.92},
                    {"reference_image": r"D:\身份库\person_001\second.jpg", "similarity": 0.78},
                    {"reference_image": r"D:\身份库\person_001\third.jpg", "similarity": 0.73},
                ],
            },
            {
                "rank": 2,
                "identity_id": "person_002",
                "identity_name": "李某",
                "score": 0.72,
                "reference_image": r"D:\身份库\person_002\best.jpg",
                "reference_count": 2,
                "mean_similarity": 0.72,
                "best_reference_score": 0.77,
                "best_reference_image": r"D:\身份库\person_002\best.jpg",
                "reference_scores": [],
            },
        ]
    return {
        "schema_version": "3.0",
        "status": "success",
        "query_image": r"D:\案件库\案件甲\query.jpg",
        "candidates": rows,
        "runtime_seconds": 2.5,
        "error_message": None,
        "method": {
            "model_name": "InsightFace buffalo_l",
            "embedding_dimension": 512,
            "aggregation_method": "mean",
            "similarity": "cosine_similarity",
            "open_set_rejection_enabled": False,
        },
        "metrics": {
            "query_face_count": 1,
            "selected_face_bbox": [1, 2, 30, 40],
            "face_detection_score": 0.97,
            "embedding_dimension": 512,
            "library_identity_count": 10,
            "library_image_count": 26,
            "valid_embedding_count": 25,
            "top1_top2_margin": 0.09 if candidates else None,
        },
        "explanation": {},
        "warnings": [],
        "limitations": [],
        "runtime": {"seconds": 2.5, "device": "cpu"},
        "provenance": {"library_path": r"C:\Users\example\identity_library"},
    }


def source_result(*, candidates: bool = True, insufficient_matches: bool = False) -> dict:
    rows = []
    if candidates:
        if insufficient_matches:
            good, inliers, ratio, local = 2, 0, 0.0, 0.05
            components = {
                "mode": "insufficient_matches",
                "good_match_count": good,
                "normalized_good_match_count": good / 40,
                "score_cap": 0.15,
                "score": local,
            }
            global_score = 0.6
        else:
            good, inliers, ratio, local = 40, 20, 0.5, 0.5
            components = {
                "mode": "normal",
                "ransac_inlier_ratio": ratio,
                "normalized_inlier_count": 0.5,
                "inlier_ratio_weight": 0.6,
                "inlier_count_weight": 0.4,
                "score": local,
            }
            global_score = 0.8
        final = 0.45 * global_score + 0.55 * local
        rows = [
            {
                "rank": 1,
                "file_path": r"D:\素材库\第一候选<script>.jpg",
                "global_score": global_score,
                "local_score": local,
                "final_score": final,
                "visualization_path": r"D:\任务\visualizations\match.jpg",
                "query_keypoint_count": 110,
                "candidate_keypoint_count": 125,
                "good_match_count": good,
                "ransac_inlier_count": inliers,
                "ransac_inlier_ratio": ratio,
                "lowe_ratio": 0.75,
                "ransac_threshold": 5.0,
                "local_score_components": components,
                "global_weight": 0.45,
                "local_weight": 0.55,
            }
        ]
        if not insufficient_matches:
            rows.append(
                {
                    "rank": 2,
                    "file_path": r"D:\素材库\第二候选.jpg",
                    "global_score": 0.7,
                    "local_score": 0.5,
                    "final_score": 0.59,
                    "visualization_path": r"D:\任务\visualizations\match2.jpg",
                    "good_match_count": 40,
                    "ransac_inlier_count": 20,
                    "ransac_inlier_ratio": 0.5,
                    "local_score_components": components,
                    "global_weight": 0.45,
                    "local_weight": 0.55,
                }
            )
    margin = rows[0]["final_score"] - rows[1]["final_score"] if len(rows) > 1 else None
    return {
        "schema_version": "3.0",
        "status": "success",
        "query_image": r"D:\案件库\案件甲\query.jpg",
        "candidates": rows,
        "runtime_seconds": 3.75,
        "error_message": None,
        "method": {
            "global_features": ["HSV", "grayscale", "edge", "border"],
            "local_features": "SIFT",
            "matcher": "BFMatcher L2",
        },
        "metrics": {
            "query_face_detected": True,
            "query_face_bbox": [2, 3, 35, 45],
            "face_mask_applied": True,
            "masked_query_path": r"D:\任务\visualizations\masked_query.jpg",
            "library_file_count": 203,
            "library_decoded": 200,
            "library_skipped_count": 3,
            "shortlist_size": 20,
            "lowe_ratio": 0.75,
            "ransac_threshold": 5.0,
            "global_weight": 0.45,
            "local_weight": 0.55,
            "top1_top2_margin": margin,
        },
        "diagnostics": {},
        "explanation": {},
        "warnings": [],
        "limitations": [],
        "runtime": {"seconds": 3.75},
        "provenance": {"library_path": r"C:\Users\example\source_library"},
    }


def skipped_deepfake() -> dict:
    return {
        "status": "skipped",
        "query_image": r"D:\案件\query.jpg",
        "is_deepfake": None,
        "score": None,
        "label": "skipped",
        "runtime_seconds": 0.0,
        "details": {"reason": "用户未选择此模块"},
        "error_message": None,
    }


class ExplainabilityTests(unittest.TestCase):
    def test_deepfake_above_threshold_uses_actual_values(self) -> None:
        raw = deepfake_result()
        explanation = build_deepfake_explanation(raw)
        technical = explanation["technical"]
        self.assertIn("疑似AI换脸", explanation["headline"])
        self.assertAlmostEqual(technical["fusion_score"], raw["score"])
        self.assertAlmostEqual(technical["threshold_margin"], raw["score"] - 0.5)
        self.assertEqual(technical["formula"]["consistent"], True)
        self.assertIn("0.620000 × 0.800000", technical["formula"]["substitution"])
        self.assertIn("MobileNetV3-Small", technical["model_name"])
        self.assertIn("87维统计特征", technical["model_name"])
        boundary_text = " ".join(explanation["limitations"] + explanation["warnings"])
        self.assertIn("不声称高置信或低置信", boundary_text)
        self.assertNotIn("本次结果为高置信", boundary_text)

    def test_deepfake_below_threshold_explains_negative_margin(self) -> None:
        raw = deepfake_result(
            spatial=0.2,
            frequency=0.1,
            threshold=0.3,
            is_deepfake=False,
            label="real",
        )
        explanation = build_deepfake_explanation(raw)
        self.assertIn("未检出明显换脸风险", explanation["headline"])
        self.assertAlmostEqual(explanation["technical"]["threshold_margin"], -0.138)
        self.assertTrue(any("低于" in value for value in explanation["basis"]))

    def test_deepfake_insufficient_evidence_stays_uncertain(self) -> None:
        raw = deepfake_result(is_deepfake=None, label="low_confidence")
        explanation = build_deepfake_explanation(raw)
        self.assertIn("不确定", explanation["headline"])
        self.assertNotIn("未检出明显", explanation["headline"])

    def test_kernel_v2_high_band_avoids_legacy_fusion_narrative(self) -> None:
        explanation = build_deepfake_explanation(kernel_deepfake_result())
        technical = explanation["technical"]
        self.assertIn("高风险提示", explanation["headline"])
        self.assertEqual(technical["branch"], "vit_full_image")
        self.assertEqual(technical["band"], "high")
        self.assertAlmostEqual(technical["risk_score"], 0.031)
        self.assertIsNone(technical.get("fusion_score"))
        joined = " ".join(
            explanation["basis"] + explanation["limitations"] + explanation["warnings"]
        )
        self.assertIn("证据质量分级", joined)
        self.assertIn("人工复核", joined)
        self.assertIn("不产生空间域/频域加权融合分解", joined)
        self.assertNotIn("无法完整复核融合公式", joined)

    def test_kernel_v2_low_band_stays_insufficient_evidence(self) -> None:
        explanation = build_deepfake_explanation(
            kernel_deepfake_result(band="low", score=0.00004)
        )
        self.assertIn("证据不足", explanation["headline"])
        self.assertNotIn("疑似AI换脸", explanation["headline"])
        self.assertEqual(explanation["technical"]["band"], "low")
        self.assertTrue(
            any("重新标定" in value for value in explanation["limitations"])
        )

    def test_identity_candidate_math_and_closed_set_boundary(self) -> None:
        explanation = build_identity_explanation(identity_result())
        technical = explanation["technical"]
        self.assertAlmostEqual(technical["top1_top2_margin"], 0.09)
        self.assertEqual(technical["candidates"][0]["reference_count"], 3)
        self.assertEqual(technical["candidates"][0]["best_reference_name"], "best.jpg")
        joined = " ".join(explanation["basis"] + explanation["limitations"])
        self.assertIn("余弦相似度不是身份概率", joined)
        self.assertIn("不能表述为已确认身份", joined)

    def test_identity_no_candidate_and_no_face_are_structured(self) -> None:
        empty = build_identity_explanation(identity_result(candidates=False))
        self.assertIn("未返回候选", empty["headline"])
        no_face = identity_result(candidates=False)
        no_face.update(status="error", error_message="查询图无人脸", metrics={"query_face_count": 0})
        failed = build_identity_explanation(no_face)
        self.assertEqual(failed["technical"]["query_face_count"], 0)
        self.assertIn("未检测到", failed["headline"])

    def test_source_normal_formula_margin_and_evidence(self) -> None:
        raw = source_result()
        explanation = build_source_explanation(raw)
        technical = explanation["technical"]
        top1 = technical["candidates"][0]
        self.assertAlmostEqual(top1["local_formula"]["calculated_score"], 0.5)
        self.assertEqual(top1["local_formula"]["consistent"], True)
        self.assertAlmostEqual(technical["final_formula"]["calculated_score"], 0.635)
        self.assertAlmostEqual(technical["top1_top2_margin"], 0.045)
        self.assertEqual(technical["library_skipped"], 3)
        joined = " ".join(explanation["basis"] + explanation["limitations"])
        self.assertIn("RANSAC内点20个", joined)
        self.assertIn("不是图像来源概率", joined)

    def test_source_good_under_four_uses_real_special_branch(self) -> None:
        explanation = build_source_explanation(source_result(insufficient_matches=True))
        formula = explanation["technical"]["candidates"][0]["local_formula"]
        self.assertEqual(formula["mode"], "insufficient_matches")
        self.assertAlmostEqual(formula["calculated_score"], 0.05)
        self.assertIn("min(2 / 40, 0.15)", formula["substitution"])
        self.assertTrue(any("未执行RANSAC" in value for value in explanation["basis"]))

    def test_source_empty_library_and_module_failure(self) -> None:
        raw = source_result(candidates=False)
        raw.update(status="error", error_message="素材库为空", metrics={"library_decoded": 0})
        explanation = build_source_explanation(raw)
        self.assertIn("素材库为空", explanation["headline"])
        failure = build_deepfake_explanation(
            {
                "status": "error",
                "error_message": r"C:\Users\example\模型缺失\model.pt 不存在",
            }
        )
        self.assertIn("执行失败", failure["headline"])
        self.assertNotIn(r"C:\Users\example", failure["summary"])

    def test_deepfake_native_keyframe_is_task_contained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            query = task / "query.jpg"
            query.write_bytes(b"query")
            native_run = task / "visualizations" / "deepfake" / "run_001"
            keyframe = native_run / "keyframes" / "keyframe_01.jpg"
            keyframe.parent.mkdir(parents=True)
            keyframe.write_bytes(b"keyframe")
            outside = task.parent / f"{task.name}_outside_keyframe.jpg"
            outside.write_bytes(b"outside")
            try:
                raw = deepfake_result()
                raw["query_image"] = str(query)
                raw["details"].update(
                    native_run_directory=str(native_run),
                    keyframe_paths=[
                        "keyframes/keyframe_01.jpg",
                        str(outside),
                        "../outside.jpg",
                    ],
                )
                explanation = build_deepfake_explanation(raw)
                resolved_query, keyframes = _resolve_deepfake_images(raw, explanation)
                self.assertEqual(resolved_query, query.resolve())
                self.assertEqual(keyframes, [keyframe.resolve()])
            finally:
                outside.unlink(missing_ok=True)


class ReportBuilderTests(unittest.TestCase):
    def test_one_model_drives_json_txt_and_escaped_html(self) -> None:
        deepfake = deepfake_result()
        identity = identity_result()
        source = source_result()
        report = build_final_report(
            "case <script>/非法:*?",
            task_status={
                "status": "success",
                "selected_modules": ["deepfake", "identity", "source"],
                "finished_at": "2026-08-26T00:00:00+08:00",
            },
            deepfake_result=deepfake,
            identity_result=identity,
            source_result=source,
            unified_result={"runtime_seconds": {"deepfake": 1.25, "identity": 2.5, "source": 3.75, "total": 7.5}},
            product_name="<script>系统</script>",
            generated_at="2026-08-26T00:00:01+08:00",
        )
        text = render_text_report(report)
        html_report = render_html_report(report)
        self.assertEqual(report["modules"]["deepfake"]["result"]["score"], deepfake["score"])
        self.assertEqual(report["modules"]["identity"]["result"]["candidates"][0]["score"], 0.81)
        self.assertEqual(report["modules"]["source"]["result"]["candidates"][0]["final_score"], 0.635)
        self.assertEqual(report["task"]["runtime_seconds"]["total"], 7.5)
        self.assertEqual(report["input"]["width"], 112)
        self.assertEqual(report["input"]["height"], 96)
        self.assertIn(DEFAULT_DISCLAIMER, text)
        self.assertIn("0.620000 × 0.800000", text)
        self.assertIn("空间域模型：MobileNetV3-Small", text)
        self.assertIn("频域模型：FrequencyStatDetector 87维统计特征 + LogisticRegression", text)
        self.assertIn("Top1与Top2差值：0.090000", text)
        self.assertIn("RANSAC内点数：20", text)
        self.assertNotIn(r"D:\案件库", text)
        self.assertNotIn(r"C:\Users\example", text)
        self.assertIn(html.escape("<script>系统</script>"), html_report)
        self.assertNotIn("<script>系统</script>", html_report)
        self.assertIsNone(report["summary"]["cross_module_fusion"])
        self.assertFalse(report["provenance"]["external_language_model_used"])

    def test_atomic_generation_legal_names_utf8_and_bundle_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary) / "task root"
            artifacts = generate_task_reports(
                task,
                task_id="case /:*? 中文",
                task_status={"status": "success", "selected_modules": ["deepfake", "identity", "source"]},
                deepfake_result=deepfake_result(),
                identity_result=identity_result(),
                source_result=source_result(),
                generated_at="2026-08-26T00:00:01+08:00",
            )
            for key in ("json_path", "txt_path", "html_path"):
                path = Path(artifacts[key])
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
            names = artifacts["file_names"]
            self.assertEqual(names, report_file_names("case /:*? 中文"))
            self.assertNotRegex(names["txt"], r"[/:*?\"<>|]")
            self.assertIn("统一检测与检索结果报告", Path(artifacts["txt_path"]).read_text(encoding="utf-8"))
            bundle = load_report_bundle(task, task_id="case /:*? 中文")
            self.assertEqual(bundle.report["schema_version"], "3.0")
            self.assertEqual(bundle.txt_path.name, names["txt"])
            self.assertEqual(bundle.html_path.name, names["html"])

    def test_partial_success_and_skipped_module_still_report(self) -> None:
        source = source_result(candidates=False)
        source.update(status="error", error_message="素材库为空", metrics={"library_decoded": 0})
        with tempfile.TemporaryDirectory() as temporary:
            from PIL import Image

            query = Path(temporary) / "query.jpg"
            Image.new("RGB", (2, 3), color=(10, 20, 30)).save(query, format="JPEG")
            artifacts = generate_task_reports(
                temporary,
                task_id="partial_case",
                task_status={
                    "status": "partial_success",
                    "selected_modules": ["identity", "source"],
                    "query_image": str(query),
                },
                deepfake_result=skipped_deepfake(),
                identity_result=identity_result(),
                source_result=source,
                generated_at="2026-08-26T00:00:01+08:00",
            )
            report = json.loads(Path(artifacts["json_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["task"]["status"], "partial_success")
            self.assertEqual(report["modules"]["deepfake"]["status"], "skipped")
            self.assertEqual(report["modules"]["source"]["status"], "error")
            self.assertEqual(report["input"]["width"], 2)
            self.assertEqual(report["input"]["height"], 3)
            self.assertEqual(len(report["input"]["input_sha256"]), 64)
            self.assertEqual(report["input"]["sha256_source"], "task_query_file")
            self.assertIn("部分", "部分成功任务可生成报告")
            self.assertIn(DEFAULT_DISCLAIMER, Path(artifacts["txt_path"]).read_text(encoding="utf-8"))

    def test_report_view_rejects_requested_task_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generate_task_reports(
                temporary,
                task_id="actual_task",
                deepfake_result=deepfake_result(),
                identity_result=identity_result(),
                source_result=source_result(),
            )
            with self.assertRaises(ReportViewError):
                load_report_bundle(temporary, task_id="different_task")


class DataLoaderV3CompatibilityTests(unittest.TestCase):
    def test_v3_extensions_load_without_breaking_v2_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            query = task / "query.jpg"
            query.write_bytes(b"query")
            identity_library = task / "identity_library"
            source_library = task / "source_library"
            visualizations = task / "visualizations"
            for directory in (identity_library, source_library, visualizations):
                directory.mkdir()
            references = []
            for name in ("identity_best.jpg", "identity_second.jpg", "identity_third.jpg"):
                path = identity_library / name
                path.write_bytes(b"image")
                references.append(path)
            source_image = source_library / "source.jpg"
            source_image.write_bytes(b"image")
            match = visualizations / "match.jpg"
            match.write_bytes(b"image")
            masked = visualizations / "masked_query.jpg"
            masked.write_bytes(b"image")

            deepfake = deepfake_result()
            deepfake["query_image"] = str(query)
            identity = identity_result()
            identity["query_image"] = str(query)
            identity["candidates"] = [identity["candidates"][0]]
            candidate = identity["candidates"][0]
            candidate["rank"] = 1
            candidate["reference_image"] = str(references[0])
            candidate["best_reference_image"] = str(references[0])
            candidate["reference_scores"] = [
                {"reference_image": str(path), "similarity": score}
                for path, score in zip(references, (0.92, 0.78, 0.73), strict=True)
            ]
            source = source_result()
            source["query_image"] = str(query)
            source["explanation"] = [
                "素材模块真实解释可以按字符串数组持久化。",
            ]
            source["candidates"] = [source["candidates"][0]]
            source["candidates"][0].update(
                rank=1,
                file_path=str(source_image),
                visualization_path=str(match),
            )
            source["metrics"]["masked_query_path"] = str(masked)
            for filename, payload in (
                ("detection_result.json", deepfake),
                ("identity_result.json", identity),
                ("source_result.json", source),
            ):
                (task / filename).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            loaded = load_task_results(
                task,
                identity_roots=(identity_library,),
                source_roots=(source_library,),
            )
            self.assertEqual(loaded.deepfake_result["schema_version"], "3.0")
            self.assertEqual(loaded.identity_result["candidates"][0]["reference_count"], 3)
            self.assertEqual(loaded.source_result["candidates"][0]["good_match_count"], 40)


if __name__ == "__main__":
    unittest.main()
