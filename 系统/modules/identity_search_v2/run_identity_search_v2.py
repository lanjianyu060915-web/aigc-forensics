# -*- coding: utf-8 -*-
"""身份检索 v2 worker：在冻结算法之上加「检测尺度回退」预处理层。

查询沿用已存在的固定无脸回退链；参考入库使用同一链，完整策略参与缓存指纹。
实际检测器按比例缩放并补边，不将某次未检出归因为未经证实的拉伸问题。

本 worker 不修改任何检索算法：

- 特征、余弦相似度、人员均值、排序、单脸约束全部来自冻结的 identity_search 包；
- 参考照片只在无人脸时回退；多脸立刻拒绝，新策略不复用旧单尺度缓存；
- 只有「本次一个尺度都没检出人脸」时，才按尺度链重试查询侧检测；
- 实际生效尺度与全部尝试逐条写入结果 JSON，供人工复核。

命令行与冻结 worker 兼容，另加 --det-size-chain。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence


MODULE_ROOT = Path(__file__).resolve().parent
PRODUCT_ROOT = MODULE_ROOT.parents[1]
FROZEN_ROOT = PRODUCT_ROOT / "modules" / "identity_search"
for _entry in (str(MODULE_ROOT), str(FROZEN_ROOT), str(PRODUCT_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from identity_search.cli import (  # noqa: E402
    StructuredArgumentParser,
    check_required_dependencies,
    emit_result,
    error_result,
    resolve_device,
)
from identity_search.errors import (  # noqa: E402
    IdentitySearchError,
    InputValidationError,
    NoFaceError,
)
from identity_search.extractor import FeatureExtractor  # noqa: E402
from identity_search.model import default_model_root, preflight_model  # noqa: E402
from identity_search.searcher import IdentitySearcher  # noqa: E402

from progress_events import create_progress_writer  # noqa: E402

from scale_fallback import (  # noqa: E402
    DEFAULT_SCALE_CHAIN,
    FALLBACK_POLICY,
    MULTI_FACE_POLICIES,
    MULTI_FACE_STRICT,
    annotate_result,
    apply_detection_scale,
    attempt_scale_chain,
    parse_scale_chain,
)


def _emit_progress(reporter, stage: str, status: str, message: str, **kwargs: Any) -> None:
    """进度证据不得影响真实检索结果，因此与冻结实现一致地吞掉异常。"""

    if reporter is None:
        return
    try:
        reporter.emit(stage, status, message, **kwargs)
    except Exception:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = StructuredArgumentParser(
        description="人脸身份搜索 v2（尺度回退预处理 + 冻结检索算法，离线模型）"
    )
    parser.add_argument("--query", required=True, help="查询图片路径（必须恰好一张人脸）")
    parser.add_argument("--library", required=True, help="身份库根目录（按人员子目录组织）")
    parser.add_argument("--top-k", type=int, default=5, help="返回候选人数，必须 > 0（默认 5）")
    parser.add_argument("--output", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--ctx", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--model-root", default=str(default_model_root()))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--progress-events", default=None)
    parser.add_argument("--progress-latest", default=None)
    parser.add_argument("--main-face-only", action="store_true", help="只运行原有主脸流程，不追加逐脸候选（视频非首要帧使用）。")
    parser.add_argument(
        "--det-size-chain",
        default=",".join(str(value) for value in DEFAULT_SCALE_CHAIN),
        help=(
            "查询与参考检测尺度回退链，逗号分隔（默认 1280,640,320）。"
            "仅在某一档一个人脸都检不到时才尝试下一档。"
        ),
    )
    parser.add_argument(
        "--multi-face-policy",
        choices=MULTI_FACE_POLICIES,
        default=MULTI_FACE_STRICT,
        help="查询图有多张人脸时的处理：strict 报错（默认），largest_face 保留最大脸主结果并追加逐脸候选。",
    )
    return parser


def _validate_inputs(args: argparse.Namespace) -> None:
    if isinstance(args.top_k, bool) or args.top_k <= 0:
        raise InputValidationError(f"top-k 必须为正整数，实际值: {args.top_k}")
    if not Path(args.query).expanduser().is_file():
        raise InputValidationError(f"查询图片不存在或不是文件: {args.query}")
    if not Path(args.library).expanduser().is_dir():
        raise InputValidationError(f"身份库目录不存在: {args.library}")


def _raw_option_value(arguments: Sequence[str], option: str) -> str | None:
    if option in arguments:
        index = list(arguments).index(option)
        if index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()
    parser = build_parser()
    try:
        args = parser.parse_args(raw_arguments)
        scales = parse_scale_chain(args.det_size_chain)
    except IdentitySearchError as exc:
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
    attempts: list[dict[str, Any]] = []
    scale_used: int | None = None
    try:
        reporter = create_progress_writer(
            task_id=args.task_id,
            module="identity",
            events_path=args.progress_events,
            latest_path=args.progress_latest,
        )
        _validate_inputs(args)
        device, ctx_id = resolve_device(args.device, args.ctx)
        check_required_dependencies()
        # 先校验模型字节，杜绝 InsightFace 的联网下载分支。
        model = preflight_model(args.model_root, verify_hashes=True)

        # 全程只创建一个检测器实例，按档位切换输入尺度，避免重复加载模型。
        primary_extractor = FeatureExtractor(
            model_root=model.root,
            device=device,
            ctx_id=ctx_id,
            det_size=(scales[0], scales[0]),
        )

        def extractor_factory(scale: int):
            return apply_detection_scale(primary_extractor, scale, ctx_id)

        searcher = IdentitySearcher(
            library_path=args.library,
            cache_dir=args.cache_dir,
            model_root=model.root,
            model_fingerprint=model.fingerprint,
            device=device,
            ctx_id=ctx_id,
            det_size=(scales[0], scales[0]),
            extractor=primary_extractor,
            reference_scales=scales,
            progress_reporter=reporter,
        )
        # 参考缓存按完整策略区分；查询侧仍沿用原固定回退流程。
        searcher._load_or_build_index()
        result, attempts = attempt_scale_chain(
            searcher,
            args.query,
            scales,
            extractor_factory,
            top_k=args.top_k,
            progress_reporter=reporter,
            emit_progress=_emit_progress,
            multi_face_policy=args.multi_face_policy,
            main_face_dir=Path(args.output).expanduser().resolve(strict=False).parent,
            include_secondary_faces=not args.main_face_only,
            artifact_root=Path(args.output).expanduser().resolve(strict=False).parent,
            task_id=args.task_id,
        )
        if result is None:
            raise NoFaceError(
                f"图片中未检测到人脸: {args.query}",
                diagnostics={
                    "face_count": 0,
                    "detected_faces": [],
                    "det_size_chain": list(scales),
                    "attempts": attempts,
                },
            )
        scale_used = attempts[-1]["scale"]
        result["runtime_seconds"] = time.perf_counter() - started
        if isinstance(result.get("runtime"), dict):
            result["runtime"]["seconds"] = result["runtime_seconds"]
        result = annotate_result(result, scales, attempts)
        _emit_progress(
            reporter,
            "identity_search_complete",
            "success",
            f"身份检索完成：检测尺度 {scale_used} 生效。",
            progress_percent=100.0,
            details={
                "detection_scale_used": scale_used,
                "fallback_applied": scale_used != scales[0],
                "attempts": attempts,
            },
            force=True,
        )
    except IdentitySearchError as exc:
        diagnostics = dict(getattr(exc, "diagnostics", {}) or {})
        diagnostics.setdefault("det_size_chain", list(scales))
        diagnostics.setdefault("attempts", attempts)
        result = error_result(
            args.query,
            str(exc),
            started,
            diagnostics=diagnostics,
            searcher=searcher,
            device=device,
            ctx_id=ctx_id,
        )
        method = result.setdefault("method", {})
        if isinstance(method, dict):
            method["preprocessing"] = {
                "policy": FALLBACK_POLICY,
                "det_size_chain": list(scales),
                "library_det_size": scales[0],
                "detection_scale_used": None,
                "fallback_applied": False,
                "attempts": attempts,
            }
        _emit_progress(
            reporter,
            "identity_failed",
            "error",
            str(exc),
            progress_percent=None,
            details=result.get("metrics"),
            force=True,
        )
    except Exception as exc:  # pragma: no cover - 与冻结 CLI 一致的兜底
        result = error_result(
            args.query,
            f"搜索过程异常: {type(exc).__name__}: {exc}",
            started,
            searcher=searcher,
            device=device,
            ctx_id=ctx_id,
        )
    return emit_result(result, args.output)


if __name__ == "__main__":
    sys.exit(main())
