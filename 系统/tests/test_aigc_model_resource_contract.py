from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / "modules" / "deepfake_detection"
if str(ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTER_ROOT))

from orchestrator import load_config, resolve_path
import run_deepfake_detection as deepfake_adapter
from run_deepfake_detection import (
    ModelResourceError,
    _classify_error,
    _map_native_result,
    _validate_resources,
    run as run_deepfake,
)
from system_status import inspect_system_status


OFFICIAL_WEIGHT_FILENAME = "mobilenet_v3_small-047dcff4.pth"
EXPECTED_COMPONENTS = {
    "spatial_model",
    "frequency_model",
    "fusion_config",
    "decision_threshold",
    "python",
    "torch",
    "torchvision",
    "sklearn",
    "opencv",
    "cuda",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_install_root() -> Path:
    config, config_path = load_config(ROOT / "config.json")
    return resolve_path(config["deepfake"]["install_root"], config_path)


def _write_fake_install(install_root: Path) -> dict[str, Path]:
    resources = {
        "executable": install_root / "深度伪造识别系统.exe",
        "spatial_model": (
            install_root
            / "models"
            / "spatial_mobilenet_finetuned_gpu"
            / "spatial_mobilenet_finetuned_gpu.pt"
        ),
        "spatial_metadata": (
            install_root
            / "models"
            / "spatial_mobilenet_finetuned_gpu"
            / "metadata.json"
        ),
        "spatial_hash": (
            install_root
            / "models"
            / "spatial_mobilenet_finetuned_gpu"
            / "model_file_sha256.json"
        ),
        "frequency_model": (
            install_root
            / "models"
            / "frequency_stat_formal"
            / "frequency_stat_detector.joblib"
        ),
        "frequency_metadata": (
            install_root / "models" / "frequency_stat_formal" / "metadata.json"
        ),
        "fusion_config": (
            install_root / "models" / "fusion_formal" / "fusion_config.json"
        ),
        "runtime_config": install_root / "config" / "runtime_pipeline.json",
    }
    for path in resources.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    resources["fusion_config"].write_text(
        json.dumps({"frame_threshold": 0.5}), encoding="utf-8"
    )
    return resources


def _native_result(
    *,
    status: str = "SUCCESS",
    actual_device: str = "cpu",
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "final_label": "REAL",
        "final_fake_score": 0.21,
        "decision_threshold": 0.5,
        "spatial_weight": 0.62,
        "frequency_weight": 0.38,
        "actual_device": actual_device,
        "input_sha256": "a" * 64,
        "valid_frame_count": 1,
        "invalid_frame_count": 0,
        "candidate_frame_count": 1,
        "spatial_score_summary": {"median": 0.2},
        "frequency_score_summary": {"median": 0.2263157894736842},
        "fusion_score_summary": {"median": 0.21},
        "warnings": list(warnings or []),
        "model_paths": {},
        "timing": {},
    }


def _function_source(path: Path, function_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    pytest.fail(f"{path} 中缺少函数 {function_name}()")


def test_formal_deepfake_config_has_no_official_weight_or_torch_cache_dependency() -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    deepfake = config["deepfake"]

    assert "base_weight" not in deepfake
    assert "torch_home" not in deepfake
    assert "base_weight" not in deepfake.get("known_sha256", {})

    serialized = json.dumps(deepfake, ensure_ascii=False).casefold()
    for forbidden in (
        OFFICIAL_WEIGHT_FILENAME.casefold(),
        r".cache\torch".casefold(),
        r"c:\users\example".casefold(),
        r"d:\刑侦查竞赛".casefold(),
    ):
        assert forbidden not in serialized


def test_resource_validator_requires_packaged_project_models_only(tmp_path: Path) -> None:
    install_root = tmp_path / "formal runtime"
    expected = _write_fake_install(install_root)

    executable, resources = _validate_resources(install_root)

    assert executable == expected["executable"]
    assert resources
    assert expected["spatial_model"] in resources.values()
    assert expected["frequency_model"] in resources.values()
    assert expected["fusion_config"] in resources.values()
    all_paths = "\n".join(str(path) for path in resources.values()).casefold()
    assert OFFICIAL_WEIGHT_FILENAME.casefold() not in all_paths
    assert "base_weight" not in " ".join(resources).casefold()
    assert all(path.is_relative_to(install_root) for path in resources.values())


def test_missing_spatial_model_is_a_model_resource_error_not_a_gpu_error(
    tmp_path: Path,
) -> None:
    install_root = tmp_path / "runtime"
    resources = _write_fake_install(install_root)
    resources["spatial_model"].unlink()
    query = tmp_path / "query.jpg"
    query.write_bytes(b"not-read-because-preflight-fails")
    output = tmp_path / "result.json"
    args = argparse.Namespace(
        query=str(query),
        output=str(output),
        install_root=str(install_root),
        device="cpu",
        self_check=False,
    )

    returncode, payload = run_deepfake(args)

    assert returncode == 1
    assert payload["status"] == "error"
    assert payload["details"]["error_code"] == "MODEL_RESOURCE_MISSING"
    assert payload["details"]["error_category"] == "模型资源缺失"
    assert payload["details"]["model_load_status"] == "failed"
    assert "spatial_mobilenet_finetuned_gpu.pt" in payload["details"]["missing_resources"]
    assert "请执行资源检查或重新安装完整模型资源" in payload["error_message"]
    assert "GPU微调空间模型加载失败" not in payload["error_message"]
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_resource_validator_exposes_all_missing_resource_names(tmp_path: Path) -> None:
    with pytest.raises(ModelResourceError) as caught:
        _validate_resources(tmp_path / "missing runtime")

    error = caught.value
    assert "spatial_mobilenet_finetuned_gpu.pt" in error.missing
    assert "frequency_stat_detector.joblib" in error.missing
    assert "fusion_config.json" in error.missing
    classified = _classify_error(error)
    assert classified["error_code"] == "MODEL_RESOURCE_MISSING"
    assert classified["user_message"] == str(error)


def test_checkpoint_parse_failure_has_its_own_error_contract(tmp_path: Path) -> None:
    native = _native_result(status="ERROR")
    native["error_message"] = "RuntimeError: checkpoint解析失败：invalid load key"

    payload = _map_native_result(native, tmp_path / "query.jpg", 0.01, tmp_path)

    assert payload["status"] == "error"
    assert payload["details"]["error_code"] == "MODEL_CHECKPOINT_INVALID"
    assert payload["details"]["error_category"] == "模型加载失败"
    assert payload["details"]["model_load_status"] == "failed"
    assert payload["error_message"] == "空间域模型加载失败：模型文件存在，但checkpoint解析失败。"
    assert "invalid load key" in payload["details"]["technical_error"]


@pytest.mark.parametrize(
    "warning",
    [
        "未检测到可用CUDA设备，系统已自动使用CPU进行推理。",
        "GPU推理失败，已自动降级至CPU继续检测。",
    ],
)
def test_cuda_unavailable_or_failed_can_finish_successfully_on_cpu(
    tmp_path: Path,
    warning: str,
) -> None:
    native = _native_result(actual_device="cpu", warnings=[warning])

    payload = _map_native_result(native, tmp_path / "query.jpg", 0.25, tmp_path)

    assert payload["status"] == "success"
    assert payload["label"] == "real"
    assert payload["error_message"] is None
    assert payload["runtime"]["actual_device"] == "cpu"
    assert payload["details"]["device_status"] == "CPU"
    assert payload["details"]["gpu_status"] == "未启用/不可用，已使用CPU"
    assert payload["details"]["model_load_status"] == "loaded"
    assert warning in payload["warnings"]


def test_cuda_inference_failure_retries_the_same_query_on_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = tmp_path / "query.jpg"
    query.write_bytes(b"fixture")
    output = tmp_path / "result.json"
    install_root = tmp_path / "runtime"
    executable = install_root / "深度伪造识别系统.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fixture")
    created_devices: list[str] = []

    class FakePipeline:
        def __init__(self, *, device: str, **_kwargs: Any) -> None:
            self.device = device
            created_devices.append(device)

        def detect_media(self, _query: Path) -> dict[str, Any]:
            if self.device == "cuda":
                raise RuntimeError("CUDA out of memory during inference")
            return _native_result(
                actual_device="cpu",
                warnings=["GPU推理失败，已自动降级至CPU继续检测。"],
            )

    fake_pipeline_module = type(sys)("src.pipeline")
    fake_pipeline_module.DeepfakeDetectionPipeline = FakePipeline
    fake_src_module = type(sys)("src")
    fake_src_module.pipeline = fake_pipeline_module
    monkeypatch.setitem(sys.modules, "src", fake_src_module)
    monkeypatch.setitem(sys.modules, "src.pipeline", fake_pipeline_module)
    monkeypatch.setattr(
        deepfake_adapter,
        "_validate_resources",
        lambda _root: (executable, {"AIGC运行时入口": executable}),
    )
    monkeypatch.setattr(
        deepfake_adapter,
        "_configure_installed_runtime",
        lambda *_args, **_kwargs: [],
    )
    args = argparse.Namespace(
        query=str(query),
        output=str(output),
        install_root=str(install_root),
        device="cuda",
        self_check=False,
    )

    returncode, payload = run_deepfake(args)

    assert returncode == 0
    assert created_devices == ["cuda", "cpu"]
    assert payload["status"] == "success"
    assert payload["runtime"]["actual_device"] == "cpu"
    assert payload["details"]["model_load_status"] == "loaded"
    assert "GPU推理失败，已自动降级至CPU继续检测。" in payload["warnings"]


@pytest.mark.parametrize(
    ("technical_error", "expected_code", "expected_message"),
    [
        (
            "ValueError: 无法读取图片：解码失败",
            "INPUT_IMAGE_INVALID",
            "输入图片无法读取或格式不支持。",
        ),
        (
            "RuntimeError: 不支持的图片格式",
            "INPUT_IMAGE_INVALID",
            "输入图片无法读取或格式不支持。",
        ),
        (
            "RuntimeError: 未检测到有效人脸",
            "NO_VALID_FACE",
            "未检测到有效人脸，当前图片无法完成空间域人脸伪造分析。",
        ),
    ],
)
def test_input_failures_are_classified_without_becoming_insufficient_evidence(
    technical_error: str,
    expected_code: str,
    expected_message: str,
) -> None:
    classified = _classify_error(technical_error)

    assert classified["error_code"] == expected_code
    assert classified["user_message"] == expected_message
    assert classified["error_category"] != "证据不足"


def test_no_face_native_result_is_successful_but_has_no_truth_claim(tmp_path: Path) -> None:
    native = _native_result(status="INSUFFICIENT_EVIDENCE")
    native.update(
        {
            "final_label": None,
            "final_fake_score": None,
            "valid_frame_count": 0,
            "invalid_frame_count": 1,
            "spatial_score_summary": {},
            "frequency_score_summary": {},
            "fusion_score_summary": {},
        }
    )

    payload = _map_native_result(native, tmp_path / "no-face.png", 0.1, tmp_path)

    assert payload["status"] == "success"
    assert payload["label"] == "no_face"
    assert payload["is_deepfake"] is None
    assert payload["score"] is None
    assert payload["details"]["error_code"] == "NO_VALID_FACE"
    assert payload["details"]["model_load_status"] == "loaded"
    assert payload["explanation"]["decision"] == (
        "未检测到有效人脸，当前图片无法完成空间域人脸伪造分析。"
    )


def test_system_status_reports_complete_aigc_component_readiness() -> None:
    status = inspect_system_status(ROOT / "config.json")
    aigc = status["deepfake"]

    assert EXPECTED_COMPONENTS <= aigc["components"].keys()
    assert all(
        isinstance(aigc["components"][name], dict)
        and "status" in aigc["components"][name]
        for name in EXPECTED_COMPONENTS
    )
    assert aigc["readiness"]["status"] == "ready"
    assert aigc["readiness"]["missing_resources"] == []
    assert aigc["available"] is True
    # CUDA is an optional accelerator. A complete CPU runtime remains ready.
    if not aigc["components"]["cuda"].get("available", False):
        assert aigc["readiness"]["status"] == "ready"
        assert status["device"]["deepfake_effective"] == "cpu"


def test_system_status_names_missing_aigc_resources_before_inference(
    tmp_path: Path,
) -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    config["deepfake"]["install_root"] = str(tmp_path / "missing runtime")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    status = inspect_system_status(config_path)
    aigc = status["deepfake"]

    assert aigc["available"] is False
    assert aigc["readiness"]["status"] == "unavailable"
    assert aigc["readiness"]["missing_resources"]
    assert aigc["components"]["spatial_model"]["status"] == "missing"
    assert aigc["components"]["frequency_model"]["status"] == "missing"
    assert aigc["components"]["fusion_config"]["status"] == "missing"
    assert any(
        "spatial" in str(item).casefold()
        for item in aigc["readiness"]["missing_resources"]
    )


def test_formal_spatial_loader_builds_without_official_weights_and_loads_strictly() -> None:
    install_root = _formal_install_root()
    mobilenet_source = install_root / "src" / "detectors" / "spatial_mobilenet.py"
    finetuning_source = install_root / "src" / "detectors" / "spatial_finetuning.py"
    assert mobilenet_source.is_file(), f"缺少正式空间域加载源码：{mobilenet_source}"
    assert finetuning_source.is_file(), f"缺少正式微调模型加载源码：{finetuning_source}"

    build_source = _function_source(mobilenet_source, "build_model")
    loader_source = _function_source(finetuning_source, "load_verified_finetuned_model")
    normalized_build = re.sub(r"\s+", "", build_source)
    normalized_loader = re.sub(r"\s+", "", loader_source)

    # The same architecture helper is retained for historical training, where
    # ImageNet initialization is an explicit opt-in. Formal inference must
    # take the default/explicit-false path whose local torchvision weight is
    # None; the subprocess test below proves that path does not touch cache.
    assert "use_imagenet_pretrained:bool=False" in normalized_build
    assert "weights=None" in normalized_build
    assert "mobilenet_v3_small(weights=weights)" in normalized_build
    assert "build_model(use_imagenet_pretrained=False)" in normalized_loader
    assert "load_state_dict(checkpoint[\"state_dict\"],strict=True)" in normalized_loader
    assert "official_weights_path" not in loader_source


def test_formal_checkpoint_is_complete_and_does_not_touch_torch_cache(
    tmp_path: Path,
) -> None:
    install_root = _formal_install_root()
    executable = install_root / "深度伪造识别系统.exe"
    checkpoint = (
        install_root
        / "models"
        / "spatial_mobilenet_finetuned_gpu"
        / "spatial_mobilenet_finetuned_gpu.pt"
    )
    assert executable.is_file(), f"缺少正式AIGC运行时：{executable}"
    assert checkpoint.is_file(), f"缺少正式空间域checkpoint：{checkpoint}"

    isolated_torch_home = tmp_path / "empty torch home"
    probe = f"""
import json
import os
from pathlib import Path
import sys

adapter_root = Path({str(ADAPTER_ROOT)!r})
install_root = Path({str(install_root)!r})
checkpoint_path = Path({str(checkpoint)!r})
torch_home = Path({str(isolated_torch_home)!r})
os.environ["TORCH_HOME"] = str(torch_home)
sys.path.insert(0, str(adapter_root))
sys.path.insert(0, str(install_root))
from pyinstaller_pyz_runtime import install_importer
install_importer(install_root / "深度伪造识别系统.exe", install_root)
handles = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for directory in (install_root, install_root / "torch" / "lib"):
        if directory.is_dir():
            handles.append(os.add_dll_directory(str(directory)))
import torch
from src.detectors.spatial_mobilenet import SpatialMobileNetDetector

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
state_dict = checkpoint["state_dict"]
model = SpatialMobileNetDetector.build_model()
expected = set(model.state_dict())
actual = set(state_dict)
print(json.dumps({{
    "checkpoint_keys": sorted(checkpoint),
    "state_dict_count": len(state_dict),
    "has_features": any(key.startswith("features.") for key in state_dict),
    "has_classifier": any(key.startswith("classifier.") for key in state_dict),
    "missing_keys": sorted(expected - actual),
    "unexpected_keys": sorted(actual - expected),
    "torch_home_exists": torch_home.exists(),
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-S", "-B", "-c", probe],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads([line for line in completed.stdout.splitlines() if line.strip()][-1])
    assert "state_dict" in result["checkpoint_keys"]
    assert result["state_dict_count"] >= 200
    assert result["has_features"] is True
    assert result["has_classifier"] is True
    assert result["missing_keys"] == []
    assert result["unexpected_keys"] == []
    assert result["torch_home_exists"] is False
    assert not (isolated_torch_home / "hub" / "checkpoints" / OFFICIAL_WEIGHT_FILENAME).exists()
    assert _sha256(checkpoint) == "5bb4b33936eb3adc631828ea271a466d62070a2b8de1902e5d96122317534794"
