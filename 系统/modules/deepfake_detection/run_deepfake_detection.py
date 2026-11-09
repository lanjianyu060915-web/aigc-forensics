from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from pyinstaller_pyz_runtime import PyInstallerArchiveError, install_importer


DEFAULT_INSTALL_ROOT = Path(
    os.environ.get(
        "DEEPFAKE_INSTALL_ROOT",
        str(Path(__file__).resolve().parents[2] / "deepfake_runtime"),
    )
)


class ModelResourceError(RuntimeError):
    """A required packaged AIGC resource is absent."""

    def __init__(self, message: str, *, missing: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing = list(missing or [])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="调用项目已有多域融合深度伪造检测模块并输出冻结 JSON 契约。"
    )
    parser.add_argument("--query", help="待检测的单张图片")
    parser.add_argument("--output", help="统一 detection_result.json 输出路径")
    parser.add_argument(
        "--install-root",
        default=str(DEFAULT_INSTALL_ROOT),
        help="已有深度伪造识别系统的安装目录",
    )
    parser.add_argument("--self-check", action="store_true", help="加载真实模型并输出AIGC资源自检JSON")
    parser.add_argument(
        "--device",
        choices=("cpu", "auto", "cuda"),
        default="auto",
        help="推理设备；CUDA不可用时auto会使用CPU",
    )
    return parser


def _base_payload(query: Path, runtime_seconds: float) -> dict[str, Any]:
    return {
        "schema_version": "3.0",
        "module": "deepfake",
        "status": "error",
        "query_image": str(query),
        "is_deepfake": None,
        "score": None,
        "label": None,
        "runtime_seconds": max(0.0, float(runtime_seconds)),
        "details": {},
        "error_message": None,
        "method": {
            "name": "多域融合单图深伪风险甄别",
            "spatial_model": "MobileNetV3-Small",
            "frequency_model": "FrequencyStatDetector 87维统计特征 + LogisticRegression",
            "fusion_method": "weighted_score",
        },
        "metrics": {},
        "explanation": {},
        "visual_evidence": {},
        "artifact_manifest": [],
        "progress_events": [],
        "warnings": [],
        "limitations": [
            "模型输出为深伪风险筛查分数，不是系统准确率或司法鉴定结论。",
            "三个模块的分数属于不同任务，不能相加或解释为综合概率。",
        ],
        "runtime": {"elapsed_seconds": max(0.0, float(runtime_seconds))},
        "provenance": {},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path_value: Any) -> str | None:
    """Hash an existing runtime resource without changing it."""

    if not path_value:
        return None
    path = Path(str(path_value)).expanduser().resolve(strict=False)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_resources(install_root: Path) -> tuple[Path, dict[str, Path]]:
    executable = install_root / "深度伪造识别系统.exe"
    spatial_model = (
        install_root
        / "models"
        / "spatial_mobilenet_finetuned_gpu"
        / "spatial_mobilenet_finetuned_gpu.pt"
    )
    frequency_model = (
        install_root
        / "models"
        / "frequency_stat_formal"
        / "frequency_stat_detector.joblib"
    )
    fusion_config = install_root / "models" / "fusion_formal" / "fusion_config.json"
    spatial_metadata = spatial_model.with_name("metadata.json")
    spatial_hash = spatial_model.with_name("model_file_sha256.json")
    frequency_metadata = frequency_model.with_name("metadata.json")
    runtime_config = install_root / "config" / "runtime_pipeline.json"
    required = {
        "AIGC运行时入口": executable,
        "空间域模型": spatial_model,
        "空间域模型元数据": spatial_metadata,
        "空间域模型校验记录": spatial_hash,
        "频域模型": frequency_model,
        "频域模型元数据": frequency_metadata,
        "融合配置": fusion_config,
        "运行管线配置": runtime_config,
    }
    missing = [path.name for path in required.values() if not path.is_file()]
    if missing:
        joined = "、".join(dict.fromkeys(missing))
        raise ModelResourceError(
            f"AIGC模型资源缺失：缺少 {joined}。请执行资源检查或重新安装完整模型资源。",
            missing=missing,
        )
    return executable, required


def _configure_installed_runtime(
    install_root: Path,
    executable: Path,
    writable_root: Path,
) -> list[Any]:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"鉴伪交付包要求 Python 3.12，当前为 {sys.version_info.major}.{sys.version_info.minor}。"
        )
    # Torch/torchvision caches, if any dependency consults them, are isolated
    # under the writable per-user task root.  Formal inference does not read an
    # official MobileNet checkpoint from this location.
    torch_cache = writable_root / "runtime" / "torch_cache"
    torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_cache)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.setdefault("MPLCONFIGDIR", str(writable_root / "runtime" / "matplotlib"))
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    sys.dont_write_bytecode = True
    if str(install_root) not in sys.path:
        sys.path.insert(0, str(install_root))
    install_importer(executable, install_root)
    handles: list[Any] = []
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for directory in (install_root, install_root / "torch" / "lib"):
            if directory.is_dir():
                handles.append(os.add_dll_directory(str(directory)))
    return handles


def _classify_error(error: Exception | str) -> dict[str, Any]:
    technical = f"{type(error).__name__}: {error}" if isinstance(error, Exception) else str(error)
    lowered = technical.casefold()
    if isinstance(error, ModelResourceError) or "模型资源缺失" in technical or "资源缺失" in technical:
        message = str(error) if isinstance(error, ModelResourceError) else technical.split(": ", 1)[-1]
        return {
            "error_code": "MODEL_RESOURCE_MISSING",
            "error_category": "模型资源缺失",
            "user_message": message,
            "technical_error": technical,
        }
    if "checkpoint" in lowered or "模型损坏或不兼容" in technical or "模型加载失败" in technical:
        return {
            "error_code": "MODEL_CHECKPOINT_INVALID",
            "error_category": "模型加载失败",
            "user_message": "空间域模型加载失败：模型文件存在，但checkpoint解析失败。",
            "technical_error": technical,
        }
    if "未检测到" in technical and "人脸" in technical:
        return {
            "error_code": "NO_VALID_FACE",
            "error_category": "输入未检出有效人脸",
            "user_message": "未检测到有效人脸，当前图片无法完成空间域人脸伪造分析。",
            "technical_error": technical,
        }
    if any(marker in technical for marker in ("无法读取图片", "不支持的图片格式", "query 不存在", "不是文件")):
        return {
            "error_code": "INPUT_IMAGE_INVALID",
            "error_category": "输入图片异常",
            "user_message": "输入图片无法读取或格式不支持。",
            "technical_error": technical,
        }
    if "cuda" in lowered:
        return {
            "error_code": "CUDA_RUNTIME_FAILURE",
            "error_category": "设备运行失败",
            "user_message": "GPU推理失败，CPU降级也未能完成检测。",
            "technical_error": technical,
        }
    return {
        "error_code": "AIGC_RUNTIME_FAILURE",
        "error_category": "AIGC运行失败",
        "user_message": "AIGC检测未能完成，请查看技术日志或执行系统资源检查。",
        "technical_error": technical,
    }


def _native_details(native: dict[str, Any], install_root: Path) -> dict[str, Any]:
    native_run = Path(str(native.get("run_directory", ""))).resolve(strict=False)
    model_paths = native.get("model_paths", {})
    model_paths = model_paths if isinstance(model_paths, dict) else {}
    fusion_config_path = native.get("fusion_config_path")
    return {
        "module_source": str(install_root),
        "native_status": native.get("status"),
        "native_label": native.get("final_label"),
        "score_semantics": "原模块空间域与频域分支按配置权重融合后的FAKE风险分数；不是系统准确率。",
        "decision_threshold": native.get("decision_threshold"),
        "spatial_weight": native.get("spatial_weight"),
        "frequency_weight": native.get("frequency_weight"),
        "actual_device": native.get("actual_device"),
        "device_status": (
            "CUDA / GPU" if str(native.get("actual_device", "")).startswith("cuda")
            else "CPU"
        ),
        "gpu_status": (
            "已启用" if str(native.get("actual_device", "")).startswith("cuda")
            else "未启用/不可用，已使用CPU"
        ),
        "media_type": native.get("media_type"),
        "input_sha256": native.get("input_sha256"),
        "valid_frame_count": native.get("valid_frame_count"),
        "invalid_frame_count": native.get("invalid_frame_count"),
        "spatial_score": native.get("spatial_score_summary", {}).get("median"),
        "frequency_score": native.get("frequency_score_summary", {}).get("median"),
        "fusion_score": native.get("fusion_score_summary", {}).get("median"),
        "candidate_frame_count": native.get("candidate_frame_count"),
        "model_paths": model_paths,
        "model_sha256": {
            "spatial": _sha256_file(model_paths.get("spatial")),
            "frequency": _sha256_file(model_paths.get("frequency")),
            "fusion_config": _sha256_file(fusion_config_path),
        },
        "model_versions": native.get("model_versions", {}),
        "fusion_config_path": fusion_config_path,
        "fusion_config_hash": native.get("fusion_config_hash"),
        "media_metadata": native.get("media_metadata", {}),
        "timing": native.get("timing", {}),
        "frame_results": native.get("frame_results", []),
        "keyframe_paths": native.get("keyframe_paths", []),
        "native_runtime_seconds": native.get("total_elapsed_seconds"),
        "native_run_directory": str(native_run),
        "native_report_paths": native.get("report_paths", {}),
        "native_progress_events": (
            list(native.get("progress_events", []))
            if isinstance(native.get("progress_events"), list)
            else []
        ),
        "spatial_initialization": "mobilenet_v3_small(weights=None) + 项目完整checkpoint(strict=True)",
        "official_base_weight_required": False,
        "warnings": list(native.get("warnings", [])),
        "disclaimer": native.get("disclaimer"),
    }


def _map_native_result(
    native: dict[str, Any],
    query: Path,
    runtime_seconds: float,
    install_root: Path,
) -> dict[str, Any]:
    payload = _base_payload(query, runtime_seconds)
    native_status = str(native.get("status", "ERROR"))
    details = _native_details(native, install_root)
    payload["details"] = details
    payload["warnings"] = list(details.get("warnings") or [])
    payload["progress_events"] = list(details.get("native_progress_events") or [])
    payload["runtime"] = {
        "elapsed_seconds": max(0.0, float(runtime_seconds)),
        "actual_device": details.get("actual_device"),
        "native_elapsed_seconds": details.get("native_runtime_seconds"),
        "timing": details.get("timing", {}),
    }
    payload["provenance"] = {
        "input_sha256": details.get("input_sha256"),
        "model_sha256": details.get("model_sha256", {}),
        "fusion_config_sha256": details.get("fusion_config_hash"),
        "model_paths": details.get("model_paths", {}),
        "native_run_directory": details.get("native_run_directory"),
        "native_report_paths": details.get("native_report_paths", {}),
    }
    payload["metrics"] = {
        "spatial_score": details.get("spatial_score"),
        "frequency_score": details.get("frequency_score"),
        "spatial_weight": details.get("spatial_weight"),
        "frequency_weight": details.get("frequency_weight"),
        "fusion_score": details.get("fusion_score"),
        "decision_threshold": details.get("decision_threshold"),
        "decision_margin": (
            float(details["fusion_score"]) - float(details["decision_threshold"])
            if details.get("fusion_score") is not None
            and details.get("decision_threshold") is not None
            else None
        ),
        "valid_frame_count": details.get("valid_frame_count"),
        "invalid_frame_count": details.get("invalid_frame_count"),
        "candidate_frame_count": details.get("candidate_frame_count"),
        "media_metadata": details.get("media_metadata", {}),
    }
    if native_status == "SUCCESS":
        native_label = str(native.get("final_label", "")).upper()
        if native_label not in {"FAKE", "REAL"}:
            payload["error_message"] = f"原模块 SUCCESS 但返回未知标签：{native_label!r}"
            return payload
        score = native.get("final_fake_score")
        if score is None:
            payload["error_message"] = "原模块 SUCCESS 但没有返回融合伪造风险评分。"
            return payload
        payload.update(
            {
                "status": "success",
                "is_deepfake": native_label == "FAKE",
                "score": float(score),
                "label": "deepfake" if native_label == "FAKE" else "real",
                "error_message": None,
            }
        )
        payload["explanation"] = {
            "decision_rule": "fusion_score >= decision_threshold",
            "fusion_formula": "spatial_weight × spatial_score + frequency_weight × frequency_score",
            "decision": "疑似AI换脸" if native_label == "FAKE" else "未检出明显换脸风险",
            "boundary_band_enabled": False,
            "boundary_band_reason": "当前版本没有经独立验证的决策边界带，仅报告实际阈值差值。",
        }
        payload["details"]["model_load_status"] = "loaded"
        return payload
    if native_status in {"LOW_CONFIDENCE", "INSUFFICIENT_EVIDENCE"}:
        no_face = details.get("valid_frame_count") == 0
        payload.update(
            {
                "status": "success",
                "score": (
                    float(native["final_fake_score"])
                    if native.get("final_fake_score") is not None
                    else None
                ),
                "label": "no_face" if no_face else native_status.lower(),
                "error_message": None,
            }
        )
        payload["explanation"] = {
            "decision": (
                "未检测到有效人脸，当前图片无法完成空间域人脸伪造分析。"
                if no_face
                else "证据不足，不能输出真假结论"
            ),
            "boundary_band_enabled": False,
            "boundary_band_reason": "当前版本没有经独立验证的决策边界带。",
        }
        payload["details"]["error_code"] = "NO_VALID_FACE" if no_face else native_status
        payload["details"]["model_load_status"] = "loaded"
        return payload
    failure = _classify_error(
        str(native.get("error_message") or f"原鉴伪模块返回状态 {native_status}。")
    )
    payload["error_message"] = failure["user_message"]
    payload["details"].update(failure)
    payload["details"]["model_load_status"] = (
        "failed" if failure["error_code"].startswith("MODEL_") else "not_completed"
    )
    return payload


def _attach_visual_evidence(
    *,
    pipeline: Any,
    native: dict[str, Any],
    query: Path,
    output: Path,
    payload: dict[str, Any],
) -> None:
    """Attach optional evidence while preserving the authoritative verdict.

    This function intentionally has no return value.  Every caller keeps the
    already mapped status, score, threshold and label even if any explanatory
    computation or artifact write fails.
    """

    evidence_started = time.perf_counter()
    task_root = output.parent.resolve(strict=False)
    try:
        from src.reporting.visual_evidence import generate_deepfake_visual_evidence

        visual_evidence, artifact_manifest, evidence_events = (
            generate_deepfake_visual_evidence(
                query_path=query,
                task_root=task_root,
                detector=pipeline.detector,
                result_payload=payload,
                native_result=native,
            )
        )
        task_id = task_root.name
        native_events = payload.get("progress_events")
        if not isinstance(native_events, list):
            native_events = []
        all_events = [*native_events, *evidence_events]
        for event in all_events:
            if isinstance(event, dict):
                event.setdefault("task_id", task_id)
        payload["visual_evidence"] = visual_evidence
        payload["artifact_manifest"] = artifact_manifest
        payload["progress_events"] = all_events
        details = payload.setdefault("details", {})
        if isinstance(details, dict):
            details["visual_evidence_status"] = (
                visual_evidence.get("deepfake", {}).get("status", "unknown")
            )
            details["artifact_count"] = len(artifact_manifest)
            details["evidence_generation_seconds"] = time.perf_counter() - evidence_started
    except Exception as exc:
        warning = (
            "AIGC检测已完成，但本次可视化证据生成失败；检测分数、阈值和结论未受影响。"
        )
        warnings = payload.setdefault("warnings", [])
        if isinstance(warnings, list) and warning not in warnings:
            warnings.append(warning)
        payload["visual_evidence"] = {
            "deepfake": {
                "schema_version": "1.0",
                "status": "unavailable",
                "message": "本次任务未生成该证据。",
                "technical_error": f"{type(exc).__name__}: {exc}",
                "failure_isolation": "检测状态、分数、阈值和结论保持原始真实结果。",
            }
        }
        payload.setdefault("artifact_manifest", [])
        details = payload.setdefault("details", {})
        if isinstance(details, dict):
            details["visual_evidence_status"] = "unavailable"
            details["evidence_generation_seconds"] = time.perf_counter() - evidence_started
            details["visual_evidence_error"] = f"{type(exc).__name__}: {exc}"


def _build_pipeline(
    *,
    install_root: Path,
    device: str,
    output_root: Path,
    writable_root: Path,
) -> tuple[Any, str | None]:
    from src.pipeline import DeepfakeDetectionPipeline

    warning: str | None = None
    try:
        pipeline = DeepfakeDetectionPipeline(
            project_root=install_root,
            device=device,
            output_root=output_root,
            writable_root=writable_root,
        )
    except Exception as exc:
        if device not in {"auto", "cuda"} or "cuda" not in str(exc).casefold():
            raise
        pipeline = DeepfakeDetectionPipeline(
            project_root=install_root,
            device="cpu",
            output_root=output_root,
            writable_root=writable_root,
        )
        warning = "未检测到可用CUDA设备，系统已自动使用CPU进行推理。"
    return pipeline, warning


def _is_cuda_runtime_failure(error: Exception | str) -> bool:
    """Return whether a failed inference is specific to the CUDA execution path."""

    text = str(error).casefold()
    return any(
        marker in text
        for marker in (
            "cuda",
            "cudnn",
            "cublas",
            "out of memory",
            "device-side assert",
            "no kernel image",
        )
    )


def _detect_with_cpu_fallback(
    *,
    pipeline: Any,
    query: Path,
    requested_device: str,
    install_root: Path,
    output_root: Path,
    writable_root: Path,
) -> tuple[dict[str, Any], str | None]:
    """Run one query and retry it on a newly constructed CPU pipeline after a CUDA failure."""

    try:
        native = pipeline.detect_media(query)
        native_error = (
            str(native.get("error_message") or "")
            if isinstance(native, dict) and str(native.get("status", "")).upper() == "ERROR"
            else ""
        )
        if (
            requested_device in {"auto", "cuda"}
            and native_error
            and _is_cuda_runtime_failure(native_error)
        ):
            raise RuntimeError(native_error)
        return native, None
    except Exception as exc:
        if requested_device not in {"auto", "cuda"} or not _is_cuda_runtime_failure(exc):
            raise
        cpu_pipeline, _ = _build_pipeline(
            install_root=install_root,
            device="cpu",
            output_root=output_root,
            writable_root=writable_root,
        )
        native = cpu_pipeline.detect_media(query)
        return native, "GPU推理失败，已自动降级至CPU继续检测。"


def run_self_check(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    install_root = Path(args.install_root).expanduser().resolve(strict=False)
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    writable_root = (
        Path(local_app_data) / "FaceSourceTrace" / "resource_check"
        if local_app_data
        else Path.cwd() / "runtime" / "resource_check"
    ).resolve(strict=False)
    checks: dict[str, Any] = {}
    try:
        executable, resources = _validate_resources(install_root)
        checks.update(
            {
                label: {"status": "正常", "path": str(path), "sha256": _sha256_file(path)}
                for label, path in resources.items()
            }
        )
        _dll_handles = _configure_installed_runtime(
            install_root, executable, writable_root
        )
        import cv2
        import joblib
        import sklearn
        import torch
        import torchvision

        pipeline, fallback_warning = _build_pipeline(
            install_root=install_root,
            device=args.device,
            output_root=writable_root / "runs",
            writable_root=writable_root,
        )
        load_seconds = float(pipeline._ensure_model_loaded())
        actual_device = str(getattr(pipeline.detector, "device", "unknown"))
        checks["Python解释器"] = {"status": "正常", "version": sys.version.split()[0], "path": sys.executable}
        checks["torch"] = {"status": "正常", "version": str(torch.__version__)}
        checks["torchvision"] = {"status": "正常", "version": str(torchvision.__version__)}
        checks["sklearn"] = {"status": "正常", "version": str(sklearn.__version__)}
        checks["opencv"] = {"status": "正常", "version": str(cv2.__version__)}
        checks["joblib"] = {"status": "正常", "version": str(joblib.__version__)}
        checks["CUDA"] = {
            "status": "可用" if torch.cuda.is_available() else "不可用，将使用CPU",
            "available": bool(torch.cuda.is_available()),
            "actual_device": actual_device,
        }
        payload = {
            "schema_version": "1.0",
            "module": "deepfake",
            "status": "ready",
            "available": True,
            "message": "AIGC模块：就绪",
            "model_load_status": "loaded",
            "actual_device": actual_device,
            "official_base_weight_required": False,
            "spatial_initialization": "mobilenet_v3_small(weights=None) + 项目完整checkpoint(strict=True)",
            "model_load_seconds": load_seconds,
            "warning": fallback_warning,
            "checks": checks,
            "runtime_seconds": time.perf_counter() - started,
        }
        return 0, payload
    except Exception as exc:
        failure = _classify_error(exc)
        payload = {
            "schema_version": "1.0",
            "module": "deepfake",
            "status": "unavailable",
            "available": False,
            "message": "AIGC模块：不可用",
            "model_load_status": "failed",
            "official_base_weight_required": False,
            "checks": checks,
            **failure,
            "runtime_seconds": time.perf_counter() - started,
        }
        if isinstance(exc, ModelResourceError):
            payload["missing_resources"] = exc.missing
        return 1, payload


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    query = Path(args.query).expanduser().resolve(strict=False)
    output = Path(args.output).expanduser().resolve(strict=False)
    install_root = Path(args.install_root).expanduser().resolve(strict=False)
    payload = _base_payload(query, 0.0)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not query.is_file():
            raise RuntimeError(f"query 不存在或不是文件：{query}")
        executable, _resources = _validate_resources(install_root)
        _dll_handles = _configure_installed_runtime(
            install_root,
            executable,
            output.parent,
        )
        native_output_root = output.parent / "visualizations" / "deepfake"
        pipeline, fallback_warning = _build_pipeline(
            install_root=install_root,
            device=args.device,
            output_root=native_output_root,
            writable_root=output.parent,
        )
        native, inference_warning = _detect_with_cpu_fallback(
            pipeline=pipeline,
            query=query,
            requested_device=args.device,
            install_root=install_root,
            output_root=native_output_root,
            writable_root=output.parent,
        )
        for warning in (fallback_warning, inference_warning):
            if warning:
                native.setdefault("warnings", []).append(warning)
        payload = _map_native_result(
            native,
            query,
            time.perf_counter() - started,
            install_root,
        )
        _attach_visual_evidence(
            pipeline=pipeline,
            native=native,
            query=query,
            output=output,
            payload=payload,
        )
    except (PyInstallerArchiveError, OSError, RuntimeError, ValueError, ImportError) as exc:
        payload = _base_payload(query, time.perf_counter() - started)
        failure = _classify_error(exc)
        payload["error_message"] = failure["user_message"]
        payload["details"].update(failure)
        payload["details"]["model_load_status"] = (
            "failed" if failure["error_code"].startswith("MODEL_") else "not_completed"
        )
        if isinstance(exc, ModelResourceError):
            payload["details"]["missing_resources"] = exc.missing
    payload["runtime_seconds"] = time.perf_counter() - started
    payload["runtime"] = {
        **(payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}),
        "elapsed_seconds": payload["runtime_seconds"],
        "actual_device": (payload.get("details") or {}).get("actual_device"),
        "gpu_status": (payload.get("details") or {}).get("gpu_status"),
    }
    try:
        _write_json(output, payload)
    except OSError as exc:
        print(f"无法写入鉴伪结果 JSON：{output}；{exc}", file=sys.stderr)
        return 2, payload
    return (0 if payload["status"] == "success" else 1), payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_check:
        returncode, payload = run_self_check(args)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return returncode
    if not args.query or not args.output:
        build_parser().error("普通检测必须同时提供 --query 和 --output")
    returncode, payload = run(args)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
