from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


FRONTEND_ROOT = Path(__file__).resolve().parents[1]
if str(FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(FRONTEND_ROOT))

from components.data_loader import (
    load_detection_result,
    load_task_results,
    validate_detection_data,
    validate_result_data,
)
from components.path_security import ResourcePathPolicy
from components.live_progress import build_progress_snapshot
from components.result_views import build_evidence_summary
from components.status_adapter import device_label, module_availability
from streamlit.testing.v1 import AppTest


class DetectionSchemaTests(unittest.TestCase):
    def test_low_confidence_stays_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query = root / "query.jpg"
            query.write_bytes(b"fixture")
            policy = ResourcePathPolicy(task_roots=(root,))
            payload = {
                "status": "success",
                "query_image": str(query),
                "is_deepfake": None,
                "score": 0.51,
                "label": "low_confidence",
                "runtime_seconds": 0.1,
                "details": {"native_status": "LOW_CONFIDENCE"},
                "error_message": None,
            }
            validated = validate_detection_data(payload, policy)
            self.assertIsNone(validated["is_deepfake"])
            summary = build_evidence_summary(
                validated,
                {
                    "status": "skipped",
                    "query_image": str(query),
                    "candidates": [],
                    "runtime_seconds": 0.0,
                    "error_message": None,
                },
                {
                    "status": "skipped",
                    "query_image": str(query),
                    "candidates": [],
                    "runtime_seconds": 0.0,
                    "error_message": None,
                },
            )
            self.assertIn("不确定", summary["deepfake"]["headline"])
            self.assertIn("不能解释为 REAL", summary["deepfake"]["detail"])

    def test_skipped_schemas_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query = root / "query.jpg"
            query.write_bytes(b"fixture")
            policy = ResourcePathPolicy(task_roots=(root,))
            deepfake = {
                "status": "skipped",
                "query_image": str(query),
                "is_deepfake": None,
                "score": None,
                "label": "skipped",
                "runtime_seconds": 0.0,
                "details": {"reason": "用户未选择此模块"},
                "error_message": None,
            }
            search = {
                "status": "skipped",
                "query_image": str(query),
                "candidates": [],
                "runtime_seconds": 0.0,
                "error_message": None,
                "details": {"reason": "用户未选择此模块"},
            }
            self.assertEqual(
                validate_detection_data(deepfake, policy)["status"], "skipped"
            )
            self.assertEqual(
                validate_result_data(search, "identity", policy)["status"],
                "skipped",
            )


class TaskCompatibilityTests(unittest.TestCase):
    def test_legacy_dual_module_task_gets_skipped_deepfake(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            query = task / "query.jpg"
            query.write_bytes(b"fixture")
            common = {
                "status": "success",
                "query_image": str(query),
                "candidates": [],
                "runtime_seconds": 0.01,
                "error_message": None,
            }
            (task / "identity_result.json").write_text(
                json.dumps(common), encoding="utf-8"
            )
            (task / "source_result.json").write_text(
                json.dumps(common), encoding="utf-8"
            )
            results = load_task_results(task)
            self.assertTrue(results.legacy_dual_module)
            self.assertEqual(results.deepfake_result["status"], "skipped")
            self.assertIsNone(results.deepfake_result["is_deepfake"])

    def test_bundled_mock_has_all_three_results(self) -> None:
        sample = FRONTEND_ROOT / "sample_data"
        policy = ResourcePathPolicy(
            sample_roots=(sample,),
            identity_roots=(sample / "identity_libraries" / "demo",),
            source_roots=(sample / "source_libraries" / "demo",),
        )
        payload = load_detection_result(
            FRONTEND_ROOT / "mock_results" / "detection_result.json", policy
        )
        self.assertEqual(payload["status"], "success")

    def test_bundled_mock_task_loads_as_three_module_task(self) -> None:
        sample = FRONTEND_ROOT / "sample_data"
        results = load_task_results(
            FRONTEND_ROOT / "mock_task",
            identity_roots=(sample / "identity_libraries" / "demo",),
            source_roots=(sample / "source_libraries" / "demo",),
        )
        self.assertFalse(results.legacy_dual_module)
        self.assertEqual(results.deepfake_result["status"], "success")
        self.assertEqual(len(results.identity_result["candidates"]), 5)
        self.assertEqual(len(results.source_result["candidates"]), 5)


class SystemStatusAdapterTests(unittest.TestCase):
    def test_realtime_source_index_with_zero_persistent_features_is_runnable(self) -> None:
        snapshot = {
            "source": {
                "available": True,
                "status": "available",
                "image_count": 203,
                "feature_count": 0,
                "index_status": "not_persistent",
                "message": "实时扫描",
            }
        }
        adapted = module_availability(snapshot, "source")
        self.assertTrue(adapted.runnable)
        self.assertIn("203", adapted.value)
        self.assertIn("not_persistent", adapted.caption)

    def test_device_reports_effective_cpu_not_hardware_cuda(self) -> None:
        value, detail = device_label(
            {
                "device": {
                    "cuda_available": True,
                    "deepfake_cuda_available": False,
                    "gpu_names": ["RTX 5060"],
                    "deepfake_effective": "cpu",
                    "identity_effective": "cpu",
                    "effective": "cpu",
                }
            }
        )
        self.assertEqual(value, "CPU")
        self.assertIn("AIGC CUDA=不可用", detail)


class LiveProgressDisplayTests(unittest.TestCase):
    def test_intermediate_stage_success_does_not_finish_module_card(self) -> None:
        intermediate = {
            "module": "identity",
            "stage": "identity_cache_ready",
            "status": "success",
            "message": "缓存已校验",
            "details": {},
        }
        snapshot = build_progress_snapshot(
            [intermediate], selected_modules=["identity"], job_status="running"
        )
        self.assertEqual(snapshot.events[0]["status"], "success")
        self.assertEqual(snapshot.module_events["identity"]["status"], "running")

        finished = {
            "module": "identity",
            "stage": "identity_finished",
            "status": "success",
            "message": "身份模块完成",
            "details": {},
        }
        snapshot = build_progress_snapshot(
            [intermediate, finished],
            selected_modules=["identity"],
            job_status="success",
        )
        self.assertEqual(snapshot.module_events["identity"]["status"], "success")


class StreamlitSmokeTests(unittest.TestCase):
    def _app(self) -> AppTest:
        return AppTest.from_file(str(FRONTEND_ROOT / "app.py"), default_timeout=10)

    def test_live_home_renders_without_starting_models(self) -> None:
        config = FRONTEND_ROOT / "tests" / "disabled_services_config.json"
        with patch.dict(os.environ, {"FSTS_CONFIG_PATH": str(config)}):
            app = self._app().run()
        self.assertFalse(app.exception)
        self.assertEqual(
            app.title[0].value,
            "AI换脸涉案图像智能鉴伪与原始素材溯源系统",
        )
        labels = {button.label for button in app.button}
        self.assertTrue(
            {"AI换脸/深伪风险甄别", "人脸身份搜索", "原始底图搜索", "一键综合分析"}
            <= labels
        )
        self.assertNotIn("加载 Mock 演示", labels)

    def test_archived_task_defaults_do_not_reference_excluded_demo_assets(self) -> None:
        source = (FRONTEND_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("_archived_task_defaults", source)
        self.assertNotIn('value=str(MOCK_TASK_ROOT)', source)
        self.assertNotIn('value=str(IDENTITY_LIBRARY_ROOT / "demo")', source)
        self.assertNotIn('value=str(SOURCE_LIBRARY_ROOT / "demo")', source)

    def test_mock_mode_loads_five_evidence_pages(self) -> None:
        config = FRONTEND_ROOT / "tests" / "disabled_services_config.json"
        with patch.dict(
            os.environ,
            {"FSTS_CONFIG_PATH": str(config), "FSTS_ENABLE_DEV_MOCK": "1"},
        ):
            app = self._app().run()
            app.radio[0].set_value("Mock Demo").run()
            load = next(button for button in app.button if button.label == "加载 Mock 演示")
            load.click().run()
        self.assertFalse(app.exception)
        evidence_navigation = next(
            radio for radio in app.radio if radio.label == "研判页面"
        )
        self.assertEqual(
            evidence_navigation.options,
            [
                "研判总览",
                "AIGC鉴伪证据",
                "人脸身份检索",
                "原始底图溯源",
                "证据链与完整报告",
            ],
        )


if __name__ == "__main__":
    unittest.main()
