"""Lightweight command-line orchestration for the frozen A interface."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

from .errors import DependencyError, IdentitySearchError, InputValidationError
from .model import MODEL_NAME, default_model_root, preflight_model


PRODUCT_ROOT = Path(__file__).resolve().parents[3]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))

from progress_events import create_progress_writer


REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "cv2": "opencv-python-headless",
    "insightface": "insightface",
    "onnxruntime": "onnxruntime",
}


def _safe_progress(
    reporter, stage: str, status: str, message: str, **kwargs: Any
) -> None:
    if reporter is None:
        return
    try:
        reporter.emit(stage, status, message, **kwargs)
    except Exception:
        # Progress evidence must never change a genuine identity-search result.
        return


class StructuredArgumentParser(argparse.ArgumentParser):
    """Raise a domain error so parse failures can still be written as JSON."""

    def error(self, message: str) -> None:
        raise InputValidationError(f"命令行参数错误: {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = StructuredArgumentParser(
        description="人脸身份搜索模块（InsightFace buffalo_l，CPU 优先，离线模型）"
    )
    parser.add_argument("--query", required=True, help="查询图片路径（必须恰好一张人脸）")
    parser.add_argument("--library", required=True, help="身份库根目录（按人员子目录组织）")
    parser.add_argument("--top-k", type=int, default=5, help="返回候选人数，必须 > 0（默认 5）")
    parser.add_argument("--output", required=True, help="输出 JSON 文件路径")
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=None,
        help="推理设备（默认 CPU；GPU 仅在 CUDA provider 可用时）",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=None,
        help="兼容旧参数：-1 为 CPU，非负整数为 GPU ID",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="JSON/NPZ 特征缓存目录；默认使用系统用户缓存，绝不默认写源码目录",
    )
    parser.add_argument(
        "--model-root",
        default=str(default_model_root()),
        help="InsightFace 根目录；模型应位于 <root>/models/buffalo_l",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="显式声明离线运行（本工具无论是否指定均不会自动下载模型）",
    )
    parser.add_argument("--task-id", default=None, help="统一进度事件所属 task_id")
    parser.add_argument(
        "--progress-events",
        default=None,
        help="统一进度 JSONL 路径；须与 --task-id/--progress-latest 同时提供",
    )
    parser.add_argument(
        "--progress-latest",
        default=None,
        help="最新进度原子 JSON 路径；须与 --task-id/--progress-events 同时提供",
    )
    return parser


def check_required_dependencies() -> None:
    missing = [
        f"{distribution}（import {module}）"
        for module, distribution in REQUIRED_IMPORTS.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise DependencyError(
            "缺少运行依赖: " + "、".join(missing) + "。请在 A_env 中安装 requirements.txt。"
        )


def resolve_device(device: str | None, ctx_id: int | None) -> tuple[str, int]:
    """Resolve new and legacy device options without making GPU the default."""

    if ctx_id is not None and ctx_id < -1:
        raise InputValidationError(f"ctx 必须为 -1 或非负整数，实际值: {ctx_id}")
    if device is None:
        if ctx_id is None or ctx_id == -1:
            return "cpu", -1
        return "gpu", ctx_id
    if device == "cpu":
        if ctx_id not in (None, -1):
            raise InputValidationError("--device cpu 与非负 --ctx 冲突")
        return "cpu", -1
    if ctx_id == -1:
        raise InputValidationError("--device gpu 与 --ctx -1 冲突")
    return "gpu", 0 if ctx_id is None else ctx_id


def _failure_metrics(
    diagnostics: dict[str, Any], searcher: Any | None
) -> dict[str, Any]:
    face_count = diagnostics.get("face_count")
    detected_faces = diagnostics.get("detected_faces")
    library_identity_count = diagnostics.get("library_identity_count")
    library_image_count = diagnostics.get("library_image_count")
    valid_embedding_count = diagnostics.get("valid_embedding_count")
    skipped_library_image_count = diagnostics.get(
        "skipped_library_image_count"
    )
    if searcher is not None:
        identities = getattr(searcher, "identities", {})
        manifest = getattr(searcher, "_manifest", [])
        if isinstance(identities, dict):
            library_identity_count = len(identities)
            valid_embedding_count = sum(
                len(data.get("features", []))
                for data in identities.values()
                if isinstance(data, dict)
            )
        if isinstance(manifest, list):
            library_image_count = len(manifest)
        if library_image_count is not None and valid_embedding_count is not None:
            skipped_library_image_count = max(
                0, library_image_count - valid_embedding_count
            )
    selected_face = (
        detected_faces[0]
        if isinstance(detected_faces, list)
        and len(detected_faces) == 1
        and isinstance(detected_faces[0], dict)
        else {}
    )
    return {
        "query_face_count": face_count if isinstance(face_count, int) else None,
        "detected_faces": detected_faces if isinstance(detected_faces, list) else [],
        "selected_face_bbox": selected_face.get("bbox"),
        "face_detection_score": selected_face.get("det_score"),
        "embedding_dimension": diagnostics.get("embedding_dimension"),
        "library_identity_count": library_identity_count,
        "library_image_count": library_image_count,
        "valid_embedding_count": valid_embedding_count,
        "skipped_library_image_count": skipped_library_image_count,
        "top1_top2_margin": None,
    }


def error_result(
    query: str,
    message: str,
    started: float,
    *,
    diagnostics: dict[str, Any] | None = None,
    searcher: Any | None = None,
    device: str | None = None,
    ctx_id: int | None = None,
) -> dict[str, Any]:
    diagnostic_values = dict(diagnostics or {})
    runtime_seconds = max(0.0, time.perf_counter() - started)
    provenance: dict[str, Any] = {"model_name": MODEL_NAME}
    if searcher is not None:
        provenance.update(
            {
                "library_path": str(getattr(searcher, "library_path", "")),
                "library_fingerprint": getattr(searcher, "_fingerprint", None),
                "model_fingerprint": getattr(searcher, "model_fingerprint", None),
            }
        )
    return {
        "schema_version": "3.0",
        "status": "error",
        "query_image": query,
        "candidates": [],
        "runtime_seconds": runtime_seconds,
        "error_message": message,
        "method": {
            "name": "insightface_closed_set_person_mean_cosine",
            "model_name": MODEL_NAME,
            "embedding_dimension": 512,
            "embedding_normalization": "l2",
            "similarity": "cosine_similarity",
            "aggregation_method": "mean",
            "open_set_rejection_enabled": False,
        },
        "metrics": _failure_metrics(diagnostic_values, searcher),
        "diagnostics": diagnostic_values,
        "explanation": {
            "summary": "身份搜索未形成候选排序。",
            "reason": message,
        },
        "warnings": [],
        "visual_evidence": {
            "module": "identity",
            "status": "unavailable",
            "reason": message,
            "top_candidates": [],
            "per_reference_scores": [],
        },
        "artifact_manifest": [],
        "limitations": [
            "当前提供闭集候选排序，未启用未经独立验证的开放集拒识阈值。",
            "候选相似度不是身份概率，不能单独据此确认人物身份。",
            "查询图片必须恰好检测到一张人脸。",
        ],
        "runtime": {
            "seconds": runtime_seconds,
            "device": device,
            "ctx_id": ctx_id,
        },
        "provenance": provenance,
    }


def _write_json_atomic(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def emit_result(result: dict[str, Any], output: str) -> int:
    """Create the output parent first, write UTF-8 JSON, and mirror it to stdout."""

    try:
        _write_json_atomic(Path(output).expanduser(), result)
    except Exception as exc:
        original = result.get("error_message")
        write_error = f"无法写入输出 JSON {output}: {type(exc).__name__}: {exc}"
        result = dict(result)
        result.update(
            status="error",
            candidates=[],
            error_message=f"{original}；{write_error}" if original else write_error,
        )
        print(write_error, file=sys.stderr)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "success" else 1


def _new_searcher(**kwargs):
    # Importing this module triggers NumPy and, when not injected, InsightFace.
    # It therefore happens only after help, inputs, dependencies, and model bytes
    # have passed their lightweight checks.
    from .searcher import IdentitySearcher

    return IdentitySearcher(**kwargs)


def _validate_inputs(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise InputValidationError(f"top-k 必须为正整数，实际值: {args.top_k}")
    query = Path(args.query).expanduser()
    if not query.is_file():
        raise InputValidationError(f"查询图片不存在或不是文件: {args.query}")
    library = Path(args.library).expanduser()
    if not library.is_dir():
        raise InputValidationError(f"身份库目录不存在: {args.library}")


def _raw_option_value(arguments: Sequence[str], option: str) -> str | None:
    """Best-effort extraction used only when argparse itself rejects a request."""

    for index, item in enumerate(arguments):
        if item == option and index + 1 < len(arguments):
            return arguments[index + 1]
        prefix = option + "="
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(raw_arguments)
    except InputValidationError as exc:
        query = _raw_option_value(raw_arguments, "--query") or ""
        output = _raw_option_value(raw_arguments, "--output")
        result = error_result(query, str(exc), started)
        if output:
            return emit_result(result, output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2
    reporter = None
    searcher = None
    device: str | None = None
    ctx_id: int | None = None
    try:
        reporter = create_progress_writer(
            task_id=args.task_id,
            module="identity",
            events_path=args.progress_events,
            latest_path=args.progress_latest,
        )
        _safe_progress(
            reporter,
            "validate_identity_input",
            "running",
            "正在校验身份搜索输入与配置。",
            progress_percent=None,
            details={"query_exists": Path(args.query).expanduser().is_file()},
            force=True,
        )
        _validate_inputs(args)
        device, ctx_id = resolve_device(args.device, args.ctx)
        check_required_dependencies()

        # This verifies all model bytes before constructing FaceAnalysis, thereby
        # preventing InsightFace's model-download path from being reached.
        model = preflight_model(args.model_root, verify_hashes=True)
        searcher = _new_searcher(
            library_path=args.library,
            cache_dir=args.cache_dir,
            model_root=model.root,
            model_fingerprint=model.fingerprint,
            device=device,
            ctx_id=ctx_id,
            progress_reporter=reporter,
        )
        result = searcher.search(
            args.query,
            top_k=args.top_k,
            progress_reporter=reporter,
            artifact_root=Path(args.output).expanduser().parent,
            task_id=args.task_id,
        )
        # The public timing is end-to-end: preflight, model init, index, and query.
        result["runtime_seconds"] = time.perf_counter() - started
        if isinstance(result.get("runtime"), dict):
            result["runtime"]["seconds"] = result["runtime_seconds"]
    except IdentitySearchError as exc:
        diagnostic_values = getattr(exc, "diagnostics", {})
        result = error_result(
            args.query,
            str(exc),
            started,
            diagnostics=(
                dict(diagnostic_values) if isinstance(diagnostic_values, dict) else {}
            ),
            searcher=searcher,
            device=device,
            ctx_id=ctx_id,
        )
        _safe_progress(
            reporter,
            "identity_failed",
            "error",
            str(exc),
            progress_percent=None,
            details=result["metrics"],
            force=True,
        )
    except Exception as exc:
        # Keep the default CLI concise; unit tests and diagnostics retain the type.
        result = error_result(
            args.query,
            f"搜索过程异常: {type(exc).__name__}: {exc}",
            started,
            searcher=searcher,
            device=device,
            ctx_id=ctx_id,
        )
        _safe_progress(
            reporter,
            "identity_failed",
            "error",
            result["error_message"],
            progress_percent=None,
            details=result["metrics"],
            force=True,
        )
    return emit_result(result, args.output)
