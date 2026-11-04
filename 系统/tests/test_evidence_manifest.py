from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from orchestrator import _task_artifact_manifest


class TaskArtifactManifestTests(unittest.TestCase):
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
