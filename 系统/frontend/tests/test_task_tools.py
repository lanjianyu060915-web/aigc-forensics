from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


FRONTEND_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_ROOT = FRONTEND_ROOT.parent
for import_root in (FRONTEND_ROOT, PRODUCT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from components.evidence_dashboard import build_key_findings, build_task_overview
from components.input_preview import ExampleQuery, describe_input, routing_edge_from_config
from components.status_adapter import module_availability
from components.task_history import (
    build_evidence_bundle,
    display_cell,
    execution_label,
    list_tasks,
    status_label,
    summaries_to_csv,
    summaries_to_table,
    summarize_task,
    verify_task_artifacts,
)


def _image_bytes(width: int, height: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 130, 140)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_task(root: Path, task_id: str, *, status: str = "success", name: str = "现场照片.jpg") -> Path:
    task = root / task_id
    task.mkdir(parents=True)
    (task / "query.jpg").write_bytes(b"query-bytes")
    evidence = task / "visualizations" / "evidence.jpg"
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(b"evidence-bytes")
    _write(
        task / "task_status.json",
        {
            "task_id": task_id,
            "status": status,
            "started_at": "2026-11-03T05:00:00.000+08:00",
            "finished_at": "2026-11-03T05:01:00.000+08:00",
            "runtime_seconds": 60.5,
            "selected_modules": ["deepfake", "identity", "source"],
            "input": {"type": "image", "source_file": "D:/inbox/upload_x.jpg", "original_file_name": name},
        },
    )
    _write(
        task / "detection_result.json",
        {
            "status": "success",
            "score": 0.5,
            "is_deepfake": True,
            "details": {"band": "high", "band_label": "高风险提示", "short_edge_px": 800},
        },
    )
    _write(
        task / "identity_result.json",
        {
            "status": "success",
            "candidates": [{"identity_id": "person_001", "identity_name": "person_001", "mean_similarity": 0.6}],
            "metrics": {},
            "provenance": {"library_path": "D:/lib/identity"},
        },
    )
    _write(
        task / "source_result.json",
        {
            "status": "skipped",
            "candidates": [],
            "provenance": {"library_path": "D:/lib/source"},
        },
    )
    _write(
        task / "artifact_manifest.json",
        {
            "artifacts": [
                {
                    "path": "query.jpg",
                    "sha256": hashlib.sha256(b"query-bytes").hexdigest(),
                },
                {
                    "path": "visualizations/evidence.jpg",
                    "sha256": hashlib.sha256(b"evidence-bytes").hexdigest(),
                },
                {"path": None, "sha256": None},
            ]
        },
    )
    return task


class TaskHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_summary_uses_original_file_name_and_chinese_status(self) -> None:
        task = _make_task(self.root, "case_a")
        summary = summarize_task(task)
        self.assertEqual(summary["file_name"], "现场照片.jpg")
        self.assertEqual(summary["status_label"], "完成")
        self.assertEqual(summary["deepfake_label"], "高风险提示")
        self.assertEqual(summary["identity_top"], "person_001")
        self.assertEqual(summary["source_label"], "未运行")
        self.assertEqual(summary["identity_library"], "D:/lib/identity")
        self.assertEqual(summary["started_at"], "2026-11-03 05:00:00")

    def test_summary_falls_back_to_source_file_name(self) -> None:
        task = _make_task(self.root, "case_b")
        status = json.loads((task / "task_status.json").read_text(encoding="utf-8"))
        del status["input"]["original_file_name"]
        _write(task / "task_status.json", status)
        self.assertEqual(summarize_task(task)["file_name"], "upload_x.jpg")

    def test_directory_without_status_is_skipped(self) -> None:
        (self.root / "half_written").mkdir()
        _make_task(self.root, "case_c")
        self.assertEqual([item["task_id"] for item in list_tasks(self.root)], ["case_c"])
        self.assertIsNone(summarize_task(self.root / "half_written"))

    def test_missing_tasks_root_returns_empty_list(self) -> None:
        self.assertEqual(list_tasks(self.root / "not_there"), [])

    def test_list_is_newest_first_and_respects_limit(self) -> None:
        import os

        for index, name in enumerate(("case_old", "case_mid", "case_new")):
            task = _make_task(self.root, name)
            os.utime(task / "task_status.json", (1_700_000_000 + index, 1_700_000_000 + index))
        self.assertEqual(
            [item["task_id"] for item in list_tasks(self.root)],
            ["case_new", "case_mid", "case_old"],
        )
        self.assertEqual(len(list_tasks(self.root, limit=1)), 1)

    def test_csv_has_bom_for_excel_and_keeps_chinese(self) -> None:
        data = summaries_to_csv([summarize_task(_make_task(self.root, "case_d"))])
        self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
        text = data.decode("utf-8-sig")
        self.assertIn("鉴伪结论", text)
        self.assertIn("现场照片.jpg", text)

    def test_table_has_one_row_per_task(self) -> None:
        table = summaries_to_table([summarize_task(_make_task(self.root, "case_e"))])
        self.assertEqual(table[0]["任务编号"], "case_e")

    def test_verify_passes_then_detects_change_and_missing_file(self) -> None:
        task = _make_task(self.root, "case_f")
        result = verify_task_artifacts(task)
        self.assertTrue(result["available"])
        self.assertEqual((result["checked"], result["matched"], result["problems"]), (2, 2, []))

        (task / "query.jpg").write_bytes(b"tampered")
        (task / "visualizations" / "evidence.jpg").unlink()
        result = verify_task_artifacts(task)
        self.assertEqual(result["matched"], 0)
        problems = {item["文件"]: item["问题"] for item in result["problems"]}
        self.assertIn("不一致", problems["query.jpg"])
        self.assertEqual(problems["visualizations/evidence.jpg"], "文件缺失")

    def test_verify_rejects_path_outside_task(self) -> None:
        task = _make_task(self.root, "case_g")
        _write(
            task / "artifact_manifest.json",
            {"artifacts": [{"path": "../outside.txt", "sha256": "0" * 64}]},
        )
        result = verify_task_artifacts(task)
        self.assertEqual(result["problems"][0]["问题"], "路径越出任务目录")

    def test_verify_reports_old_task_without_manifest(self) -> None:
        task = _make_task(self.root, "case_h")
        (task / "artifact_manifest.json").unlink()
        self.assertFalse(verify_task_artifacts(task)["available"])

    def test_bundle_contains_files_and_checksums_but_not_runtime(self) -> None:
        task = _make_task(self.root, "case_i")
        runtime = task / "runtime" / "cache.bin"
        runtime.parent.mkdir()
        runtime.write_bytes(b"cache")

        with zipfile.ZipFile(io.BytesIO(build_evidence_bundle(task))) as archive:
            names = archive.namelist()
            self.assertIn("case_i/query.jpg", names)
            self.assertIn("case_i/visualizations/evidence.jpg", names)
            self.assertNotIn("case_i/runtime/cache.bin", names)
            sheet = archive.read("case_i/证据包文件校验清单.tsv").decode("utf-8-sig")
        self.assertIn(hashlib.sha256(b"query-bytes").hexdigest(), sheet)

    def test_display_cell_keeps_small_scores_visible(self) -> None:
        self.assertEqual(display_cell("deepfake_score", 0.00005), "0.000050")
        self.assertEqual(display_cell("deepfake_score", None), "—")
        self.assertEqual(display_cell("runtime_seconds", 60.55), "60.5")
        self.assertEqual(display_cell("file_name", "a.jpg"), "a.jpg")

    def test_labels_fall_back_to_raw_value(self) -> None:
        self.assertEqual(status_label("partial_success"), "部分完成")
        self.assertEqual(status_label("新状态"), "新状态")
        self.assertEqual(status_label(None), "未知")
        self.assertEqual(execution_label("service"), "常驻服务")


class InputPreviewTests(unittest.TestCase):
    def test_large_image_routes_to_full_image_branch(self) -> None:
        preview = describe_input("a.jpg", _image_bytes(800, 600), routing_edge=256)
        self.assertEqual((preview.width, preview.height), (800, 600))
        self.assertIn("整图", preview.route_label)
        self.assertEqual(preview.notes, [])

    def test_small_image_routes_to_face_crop_branch_and_warns(self) -> None:
        preview = describe_input("a.jpg", _image_bytes(100, 90), routing_edge=256)
        self.assertIn("人脸裁剪", preview.route_label)
        self.assertTrue(any("分辨率" in note for note in preview.notes))

    def test_routing_boundary_uses_threshold_inclusively(self) -> None:
        self.assertIn("整图", describe_input("a.jpg", _image_bytes(256, 900)).route_label)
        self.assertIn("人脸裁剪", describe_input("a.jpg", _image_bytes(255, 900)).route_label)

    def test_long_screenshot_gets_a_note(self) -> None:
        preview = describe_input("a.jpg", _image_bytes(300, 1500))
        self.assertTrue(any("长宽比" in note for note in preview.notes))

    def test_broken_image_is_not_readable(self) -> None:
        preview = describe_input("a.jpg", b"not an image")
        self.assertFalse(preview.readable)
        self.assertIsNone(preview.route_label)

    def test_video_does_not_decode_and_keeps_hash(self) -> None:
        preview = describe_input("a.mp4", b"video-bytes")
        self.assertEqual(preview.kind, "video")
        self.assertEqual(preview.sha256, hashlib.sha256(b"video-bytes").hexdigest())
        self.assertIn("逐帧", preview.route_label)

    def test_example_query_matches_upload_interface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "demo.jpg"
            path.write_bytes(b"abc")
            example = ExampleQuery(path)
            self.assertEqual((example.name, example.size, example.getvalue()), ("demo.jpg", 3, b"abc"))

    def test_routing_edge_reads_config_and_falls_back(self) -> None:
        config = {"deepfake": {"detector_v2": {"routing_short_edge_threshold_px": 300}}}
        self.assertEqual(routing_edge_from_config(config), 300)
        self.assertEqual(routing_edge_from_config({}), 256)


class KeyFindingTests(unittest.TestCase):
    def test_skipped_modules_say_not_run(self) -> None:
        skipped = {"status": "skipped", "candidates": []}
        cards, steps = build_key_findings(skipped, skipped, skipped)
        self.assertEqual([card["value"] for card in cards], ["本次未运行"] * 3)
        self.assertEqual(steps, [])

    def test_failed_module_shows_error_message(self) -> None:
        cards, _ = build_key_findings(
            {"status": "error", "error_message": "模型文件缺失"},
            {"status": "skipped"},
            {"status": "skipped"},
        )
        self.assertEqual(cards[0]["value"], "模块未完成")
        self.assertIn("模型文件缺失", cards[0]["note"])
        self.assertEqual(cards[0]["tone"], "bad")

    def test_band_result_explains_that_score_is_not_a_probability(self) -> None:
        cards, _ = build_key_findings(
            {
                "status": "success",
                "score": 0.0137,
                "details": {"band": "high", "band_label": "高风险提示", "short_edge_px": 2000},
            },
            {"status": "skipped"},
            {"status": "skipped"},
        )
        self.assertEqual((cards[0]["value"], cards[0]["tone"]), ("高风险提示", "bad"))
        self.assertIn("2000", cards[0]["note"])
        self.assertIn("没有概率含义", cards[0]["note"])

    def test_low_band_warns_it_does_not_mean_real(self) -> None:
        _, steps = build_key_findings(
            {"status": "success", "score": 0.0001, "details": {"band": "low", "band_label": "证据不足"}},
            {"status": "skipped"},
            {"status": "skipped"},
        )
        self.assertTrue(any("不代表图像真实" in step for step in steps))

    def test_face_crop_branch_uses_threshold_wording(self) -> None:
        cards, _ = build_key_findings(
            {
                "status": "success",
                "score": 0.7,
                "is_deepfake": False,
                "details": {"decision_threshold": 0.36},
            },
            {"status": "skipped"},
            {"status": "skipped"},
        )
        self.assertEqual((cards[0]["value"], cards[0]["tone"]), ("未见明显换脸特征", "ok"))
        self.assertIn("判定阈值", cards[0]["note"])


class OverviewTests(unittest.TestCase):
    def test_overview_prefers_original_file_name(self) -> None:
        overview = build_task_overview(
            {"status": "success", "query_image": "D:/t/query.jpg"},
            {"status": "skipped"},
            {"status": "skipped"},
            task_status={"input": {"original_file_name": "现场.jpg"}},
        )
        self.assertEqual(overview["query_file"], "现场.jpg")


class StatusAdapterTests(unittest.TestCase):
    def test_unbuilt_identity_index_is_runnable(self) -> None:
        snapshot = {
            "identity": {
                "available": True,
                "person_count": 9,
                "image_count": 45,
                "embedding_count": 0,
                "index_status": "not_initialized",
            }
        }
        self.assertTrue(module_availability(snapshot, "identity").runnable)

    def test_empty_identity_library_is_not_runnable(self) -> None:
        snapshot = {"identity": {"available": True, "person_count": 0, "image_count": 0}}
        self.assertFalse(module_availability(snapshot, "identity").runnable)


if __name__ == "__main__":
    unittest.main()
