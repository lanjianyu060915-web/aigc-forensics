from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


FRONTEND_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_ROOT = FRONTEND_ROOT.parent
for import_root in (FRONTEND_ROOT, PRODUCT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import app
from components.data_loader import ResultDataError


class _Upload:
    def __init__(self, name: str, data: bytes = b"image", size: int | None = None) -> None:
        self.name = name
        self._data = data
        self.size = len(data) if size is None else size

    def getvalue(self) -> bytes:
        return self._data


def _runtime() -> dict:
    return {"executor": None, "lock": threading.RLock(), "jobs": {}}


class BatchExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _entry(self, name: str) -> tuple[str, Path]:
        path = self.root / f"inbox_{name}"
        path.write_bytes(b"image")
        return name, path

    def _fake_task(self, name: str) -> str:
        task = self.root / f"task_{name}"
        task.mkdir()
        (task / "task_status.json").write_text(
            json.dumps(
                {
                    "task_id": f"task_{name}",
                    "status": "success",
                    "selected_modules": ["deepfake"],
                    "input": {"type": "image", "original_file_name": name},
                }
            ),
            encoding="utf-8",
        )
        return str(task)

    def test_one_failure_does_not_stop_the_rest(self) -> None:
        entries = [self._entry("a.jpg"), self._entry("b.jpg"), self._entry("c.jpg")]
        runtime = _runtime()
        runtime["jobs"]["job"] = {
            "status": "queued",
            "current_index": 0,
            "items": [{"name": name, "status": "queued"} for name, _ in entries],
        }

        def fake_run_task(query, **kwargs):
            name = kwargs["original_file_name"]
            if name == "b.jpg":
                raise RuntimeError("模型崩溃")
            return {"status": "success", "task_dir": self._fake_task(name)}

        with patch.object(app, "run_task", side_effect=fake_run_task):
            app._execute_batch_job(
                runtime,
                "job",
                entries=entries,
                identity_root=None,
                source_root=None,
                modules=frozenset({"deepfake"}),
                services=None,
            )

        job = runtime["jobs"]["job"]
        self.assertEqual(job["status"], "finished")
        self.assertEqual([item["status"] for item in job["items"]], ["success", "error", "success"])
        self.assertIn("模型崩溃", job["items"][1]["error"])
        self.assertEqual(job["items"][0]["summary"]["file_name"], "a.jpg")
        self.assertIsNone(job["items"][1]["summary"])
        self.assertEqual(job["items"][2]["summary"]["file_name"], "c.jpg")
        # 收件箱里的临时副本不论成败都要清掉。
        self.assertTrue(all(not path.exists() for _, path in entries))


class BatchSubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.submitted: list[dict] = []
        self.runtime = _runtime()
        outer = self

        class _Executor:
            def submit(self, function, runtime, job_id, **kwargs):
                outer.submitted.append({"job_id": job_id, **kwargs})

        self.runtime["executor"] = _Executor()
        self.patches = [
            patch.object(app, "_live_job_runtime", return_value=self.runtime),
            patch.object(app, "_live_defaults", return_value=(str(self.root), str(self.root), self.root / "inbox")),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self) -> None:
        for item in self.patches:
            item.stop()
        self._temporary.cleanup()

    def test_rejected_files_are_listed_without_shifting_the_queue(self) -> None:
        job_id = app._start_batch_analysis(
            [
                _Upload("坏文件.exe"),
                _Upload("a.jpg"),
                _Upload("clip.mp4"),
                _Upload("大图.png", size=app.MAX_UPLOAD_BYTES + 1),
                _Upload("b.png"),
            ],
            str(self.root),
            str(self.root),
            ["deepfake"],
            None,
        )
        job = self.runtime["jobs"][job_id]
        self.assertEqual([item["name"] for item in job["items"]][:2], ["a.jpg", "b.png"])
        self.assertEqual([name for name, _ in self.submitted[0]["entries"]], ["a.jpg", "b.png"])
        self.assertEqual({item["status"] for item in job["items"][2:]}, {"error"})
        reasons = {item["name"]: item["error"] for item in job["items"][2:]}
        self.assertIn("扩展名", reasons["坏文件.exe"])
        self.assertIn("只支持图片", reasons["clip.mp4"])
        self.assertIn("20 MiB", reasons["大图.png"])

    def test_requires_files_and_modules(self) -> None:
        with self.assertRaises(ResultDataError):
            app._start_batch_analysis([], str(self.root), str(self.root), ["deepfake"], None)
        with self.assertRaises(ResultDataError):
            app._start_batch_analysis([_Upload("a.jpg")], str(self.root), str(self.root), [], None)

    def test_too_many_files_are_rejected_up_front(self) -> None:
        uploads = [_Upload(f"{index}.jpg") for index in range(app.MAX_BATCH_FILES + 1)]
        with self.assertRaises(ResultDataError):
            app._start_batch_analysis(uploads, str(self.root), str(self.root), ["deepfake"], None)
        self.assertEqual(self.submitted, [])

    def test_identity_module_requires_a_library(self) -> None:
        with self.assertRaises(ResultDataError):
            app._start_batch_analysis([_Upload("a.jpg")], "", str(self.root), ["identity"], None)


if __name__ == "__main__":
    unittest.main()
