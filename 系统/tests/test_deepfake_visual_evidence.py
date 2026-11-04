from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / "modules" / "deepfake_detection"
if str(ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTER_ROOT))

import run_deepfake_detection as adapter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_real_sample(tmp_path: Path, sample_name: str) -> dict[str, Any]:
    python = ROOT / "packaging_v0.3" / "resources" / "portable_python" / "python.exe"
    runtime = ROOT / "packaging_v0.3" / "resources" / "deepfake_runtime"
    query = ROOT / "examples" / sample_name
    output = tmp_path / "中文 任务目录" / "detection_result.json"
    completed = subprocess.run(
        [
            str(python),
            "-E",
            "-S",
            "-B",
            str(ADAPTER_ROOT / "run_deepfake_detection.py"),
            "--query",
            str(query),
            "--output",
            str(output),
            "--install-root",
            str(runtime),
            "--device",
            "cpu",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert output.is_file()
    return json.loads(output.read_text(encoding="utf-8"))


def test_real_fake_sample_generates_model_grounded_task_relative_evidence(
    tmp_path: Path,
) -> None:
    result = _run_real_sample(tmp_path, "aigc_fake_image.jpg")

    assert result["status"] == "success"
    assert result["label"] == "deepfake"
    assert result["score"] == pytest.approx(0.7451661368925726, abs=1e-12)
    assert result["metrics"]["spatial_score"] == pytest.approx(
        0.8250169157981873, abs=1e-12
    )
    assert result["metrics"]["frequency_score"] == pytest.approx(
        0.6148832870992015, abs=1e-12
    )
    assert result["metrics"]["decision_threshold"] == pytest.approx(
        0.36360106194654707, abs=1e-15
    )

    deepfake = result["visual_evidence"]["deepfake"]
    assert deepfake["status"] == "generated"
    assert deepfake["face_detection"]["detected_face_count"] >= 1
    assert deepfake["face_detection"]["selected_face_index"] == 1
    assert deepfake["face_detection"]["model_input_size"] == [224, 224]
    assert deepfake["spatial"]["method"] == "occlusion_sensitivity"
    assert deepfake["spatial"]["grid_shape"] == [6, 6]
    assert deepfake["spatial"]["baseline_recompute_absolute_error"] < 1e-5
    assert "不等于像素级伪造定位" in deepfake["spatial"]["disclaimer"]
    frequency = deepfake["frequency"]
    assert len(frequency["radial_profile"]) == 32
    contributions = frequency["feature_contributions"]
    assert contributions["status"] == "supported"
    assert contributions["feature_dimension"] == 87
    assert contributions["preprocessing_steps"] == ["scaler"]
    assert contributions["reconstruction_absolute_error"] < 1e-8
    assert len(contributions["top_by_absolute_contribution"]) == 10
    assert contributions["top_positive_contributions"]
    assert contributions["top_negative_contributions"]
    fusion = deepfake["fusion"]
    assert fusion["spatial_weight"] == pytest.approx(result["metrics"]["spatial_weight"])
    assert fusion["frequency_weight"] == pytest.approx(result["metrics"]["frequency_weight"])
    assert fusion["fusion_score"] == pytest.approx(result["score"])
    assert "0.62" in fusion["formula_substitution"]

    task_root = tmp_path / "中文 任务目录"
    manifest = result["artifact_manifest"]
    assert len(manifest) >= 12
    assert len({item["id"] for item in manifest}) == len(manifest)
    for item in manifest:
        relative = item["path"]
        assert "\\" not in relative
        assert not PurePosixPath(relative).is_absolute()
        artifact = task_root / Path(*PurePosixPath(relative).parts)
        assert artifact.is_file(), artifact
        assert artifact.stat().st_size == item["size_bytes"] > 0
        assert _sha256(artifact) == item["sha256"]

    events = result["progress_events"]
    stages = [event["stage"] for event in events]
    assert stages[:9] == [
        "input_hash_validated",
        "image_decoded",
        "face_detection_completed",
        "face_crop_preprocessed",
        "spatial_model_inference_completed",
        "frequency_features_extracted",
        "frequency_model_inference_completed",
        "fusion_score_computed",
        "decision_threshold_compared",
    ]
    assert stages[-1] == "visual_evidence_completed"
    assert all(event["elapsed_seconds"] >= 0.0 for event in events)
    progress = [event["progress_percent"] for event in events]
    assert progress == sorted(progress)
    assert progress[-1] == 100.0


def test_real_no_face_sample_has_explicit_visual_failure_state(tmp_path: Path) -> None:
    result = _run_real_sample(tmp_path, "no_face_image.jpg")

    assert result["status"] == "success"
    assert result["label"] == "no_face"
    assert result["score"] is None
    deepfake = result["visual_evidence"]["deepfake"]
    assert deepfake["status"] == "partial"
    assert deepfake["face_detection"]["status"] == "no_face"
    assert deepfake["face_detection"]["detected_face_count"] == 0
    assert deepfake["spatial"]["status"] == "unavailable"
    assert deepfake["frequency"]["status"] == "unavailable"
    assert any(
        item["kind"] == "face_detection_failure_image"
        for item in result["artifact_manifest"]
    )


def test_persistent_worker_returns_the_same_visual_evidence_contract(
    tmp_path: Path,
) -> None:
    python = ROOT / "packaging_v0.3" / "resources" / "portable_python" / "python.exe"
    runtime = ROOT / "packaging_v0.3" / "resources" / "deepfake_runtime"
    query = ROOT / "examples" / "aigc_fake_image.jpg"
    output = tmp_path / "case_persistent_中文" / "detection_result.json"
    protocol = "face-source-trace-runtime-v1"
    requests = [
        {
            "protocol": protocol,
            "type": "request",
            "request_id": "visual-run",
            "ready": True,
            "health": "ok",
            "action": "run",
            "query_path": str(query),
            "output_path": str(output),
        },
        {
            "protocol": protocol,
            "type": "request",
            "request_id": "visual-shutdown",
            "ready": True,
            "health": "ok",
            "action": "shutdown",
        },
    ]
    completed = subprocess.run(
        [
            str(python),
            "-E",
            "-S",
            "-B",
            str(ADAPTER_ROOT / "persistent_worker.py"),
            "--install-root",
            str(runtime),
            "--device",
            "cpu",
            "--writable-root",
            str(tmp_path / "persistent service"),
        ],
        cwd=str(ADAPTER_ROOT),
        input="".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    messages = []
    for line in completed.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict) and message.get("protocol") == protocol:
            messages.append(message)
    ready = next(item for item in messages if item.get("type") == "ready")
    response = next(item for item in messages if item.get("request_id") == "visual-run")
    assert ready["ready"] is True
    assert response["ok"] is True
    payload = response["payload"]
    assert payload["status"] == "success"
    assert payload["score"] == pytest.approx(0.7451661368925726, abs=1e-12)
    assert payload["visual_evidence"]["deepfake"]["status"] == "generated"
    assert len(payload["artifact_manifest"]) >= 12
    assert output.is_file()


def test_visual_explanation_failure_does_not_change_authoritative_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_module = ModuleType("src")
    reporting_module = ModuleType("src.reporting")
    evidence_module = ModuleType("src.reporting.visual_evidence")

    def fail_evidence(**_kwargs: Any) -> Any:
        raise RuntimeError("synthetic explanation-only failure")

    evidence_module.generate_deepfake_visual_evidence = fail_evidence  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src", source_module)
    monkeypatch.setitem(sys.modules, "src.reporting", reporting_module)
    monkeypatch.setitem(sys.modules, "src.reporting.visual_evidence", evidence_module)
    payload = {
        "status": "success",
        "label": "deepfake",
        "score": 0.75,
        "metrics": {"decision_threshold": 0.36},
        "details": {},
        "warnings": [],
        "progress_events": [],
    }
    query = tmp_path / "query.jpg"
    query.write_bytes(b"fixture")

    adapter._attach_visual_evidence(
        pipeline=SimpleNamespace(detector=object()),
        native={},
        query=query,
        output=tmp_path / "result.json",
        payload=payload,
    )

    assert payload["status"] == "success"
    assert payload["label"] == "deepfake"
    assert payload["score"] == 0.75
    assert payload["metrics"]["decision_threshold"] == 0.36
    assert payload["visual_evidence"]["deepfake"]["status"] == "unavailable"
    assert payload["artifact_manifest"] == []
    assert "未受影响" in payload["warnings"][0]
