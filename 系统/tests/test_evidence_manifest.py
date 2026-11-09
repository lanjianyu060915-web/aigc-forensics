from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from orchestrator import _task_artifact_manifest


class TaskArtifactManifestTests(unittest.TestCase):
    def test_pdf_is_verified_and_reviews_join_the_evidence_bundle(self) -> None:
        from reporting.report_builder import generate_task_reports
        from reporting.review_records import append_review_record
        from frontend.components.task_history import build_evidence_bundle, verify_task_artifacts

        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            reports = generate_task_reports(task, task_id="清单验收")
            self.assertIn("pdf", reports["file_names"])
            manifest = _task_artifact_manifest(task, "清单验收", {}, report_files=reports["file_names"])
            (task / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            pdf = task / reports["file_names"]["pdf"]
            original_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
            entry = next(item for item in manifest["artifacts"] if item["path"] == pdf.name)
            self.assertEqual(entry["sha256"], original_hash)
            append_review_record(task, "复核员", "同意系统结论", "<b>&复核意见")
            verification = verify_task_artifacts(task)
            self.assertTrue(verification["available"])
            self.assertFalse(verification["problems"])
            self.assertEqual(hashlib.sha256(pdf.read_bytes()).hexdigest(), original_hash)
            with zipfile.ZipFile(io.BytesIO(build_evidence_bundle(task))) as archive:
                self.assertIn(f"{task.name}/review_records.jsonl", archive.namelist())
                self.assertIn(f"{task.name}/{pdf.name}", archive.namelist())

    def test_manifest_rehashes_task_relative_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            (task / "query.jpg").write_bytes(b"query")
            (task / "detection_result.json").write_text("{}", encoding="utf-8")
            (task / "identity_result.json").write_text("{}", encoding="utf-8")
            (task / "source_result.json").write_text("{}", encoding="utf-8")
            artifact = task / "visualizations" / "deepfake" / "overlay.png"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"real evidence")
            results = {
                "deepfake": {
                    "artifact_manifest": [
                        {
                            "kind": "spatial_overlay",
                            "path": "visualizations/deepfake/overlay.png",
                            "sha256": "0" * 64,
                        },
                        {"kind": "unsafe", "path": "../escape.png"},
                    ]
                },
                "identity": {},
                "source": {},
            }
            manifest = _task_artifact_manifest(task, "case_fixture", results)
            paths = {item["path"] for item in manifest["artifacts"]}
            self.assertIn("visualizations/deepfake/overlay.png", paths)
            self.assertNotIn("../escape.png", paths)
            overlay = next(
                item
                for item in manifest["artifacts"]
                if item["path"] == "visualizations/deepfake/overlay.png"
            )
            self.assertEqual(
                overlay["sha256"], hashlib.sha256(b"real evidence").hexdigest()
            )
            self.assertEqual(overlay["size_bytes"], len(b"real evidence"))
            self.assertEqual(manifest["path_policy"], "task_relative_posix")


if __name__ == "__main__":
    unittest.main()
