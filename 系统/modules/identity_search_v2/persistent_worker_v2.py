# -*- coding: utf-8 -*-
"""身份检索常驻 worker v2：冻结常驻协议 + 检测尺度回退预处理。

与冻结常驻 worker（modules/identity_search/persistent_worker.py）的关系：

- 协议、握手、request/response 结构、错误 JSON、进度事件全部沿用冻结实现，
  直接复用其 `_serve`/`_emit`/`_safe_progress`，前端与 orchestrator 无需改动；
- 模型只在工作进程启动时准备一次，尺度切换只修改检测器输入尺寸，
  不重建执行会话、不重复加载权重；
- 查询侧和参考入库复用固定无脸回退链；参考多脸立即拒绝，查询主脸策略不变。

见 docs/M2_人脸预处理_尺度回退实验.md。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parent
PRODUCT_ROOT = MODULE_ROOT.parents[1]
FROZEN_MODULE_ROOT = PRODUCT_ROOT / "modules" / "identity_search"
for _entry in (str(MODULE_ROOT), str(FROZEN_MODULE_ROOT), str(PRODUCT_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from identity_search.cli import _write_json_atomic, error_result  # noqa: E402
from identity_search.errors import (  # noqa: E402
    IdentitySearchError,
    InputValidationError,
    NoFaceError,
)
from identity_search.model import default_model_root  # noqa: E402

from progress_events import create_progress_writer  # noqa: E402
from persistent_worker import (  # noqa: E402
    PersistentIdentityRuntime,
    _emit,
    _parser as frozen_parser,
    _safe_progress,
    _serve,
)
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


def _parser() -> argparse.ArgumentParser:
    parser = frozen_parser()
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


class ScaleFallbackIdentityRuntime(PersistentIdentityRuntime):
    """冻结常驻运行时 + 查询侧检测尺度回退。"""

    def __init__(self, args: argparse.Namespace, scales: tuple[int, ...]) -> None:
        self.scales = tuple(scales)
        super().__init__(args)
        self.multi_face_policy = getattr(args, "multi_face_policy", MULTI_FACE_STRICT)
        # 参考提取会恢复尺度，查询仍从主尺度开始。
        apply_detection_scale(self._extractor, self.scales[0], self.ctx_id)

    def _reference_searcher_options(self) -> dict[str, Any]:
        return {"det_size": (self.scales[0], self.scales[0]),
                "reference_scales": self.scales}

    def health_details(self) -> dict[str, Any]:
        details = super().health_details()
        details.update(
            {
                "worker": "identity_v2",
                "detection_policy": FALLBACK_POLICY,
                "det_size_chain": list(self.scales),
                "detection_scale_current": getattr(self._extractor, "det_size", None),
            }
        )
        return details

    def run(
        self,
        query_value: Any,
        output_value: Any,
        library_value: Any,
        top_k_value: Any,
        task_id: Any = None,
        progress_events_path: Any = None,
        progress_latest_path: Any = None,
        include_secondary_faces: bool = True,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        query = Path(str(query_value)).expanduser().resolve(strict=False)
        output = Path(str(output_value)).expanduser().resolve(strict=False)
        reporter = None
        searcher = None
        attempts: list[dict[str, Any]] = []
        try:
            reporter = create_progress_writer(
                task_id=(str(task_id) if task_id is not None else None),
                module="identity",
                events_path=(
                    str(progress_events_path) if progress_events_path is not None else None
                ),
                latest_path=(
                    str(progress_latest_path) if progress_latest_path is not None else None
                ),
            )
            _safe_progress(
                reporter,
                "validate_identity_input",
                "running",
                "正在校验身份搜索输入与配置。",
                progress_percent=None,
                details={"query_exists": query.is_file()},
                force=True,
            )
            if not query.is_file():
                raise InputValidationError(f"查询图片不存在或不是文件: {query}")
            if isinstance(top_k_value, bool):
                raise InputValidationError(f"top-k 必须为正整数，实际值: {top_k_value}")
            top_k = int(top_k_value)
            if top_k <= 0:
                raise InputValidationError(f"top-k 必须为正整数，实际值: {top_k_value}")
            # 查询仍从主尺度开始；新库与首库使用相同参考回退策略。
            apply_detection_scale(self._extractor, self.scales[0], self.ctx_id)
            searcher = self._searcher_for(library_value, reporter)
            result, attempts = attempt_scale_chain(
                searcher,
                str(query),
                self.scales,
                lambda scale: apply_detection_scale(self._extractor, scale, self.ctx_id),
                top_k=top_k,
                progress_reporter=reporter,
                emit_progress=_safe_progress,
                multi_face_policy=self.multi_face_policy,
                main_face_dir=output.parent,
                include_secondary_faces=include_secondary_faces,
                artifact_root=output.parent,
                task_id=(str(task_id) if task_id is not None else None),
            )
            if result is None:
                raise NoFaceError(
                    f"图片中未检测到人脸: {query}",
                    diagnostics={
                        "face_count": 0,
                        "detected_faces": [],
                        "det_size_chain": list(self.scales),
                        "attempts": attempts,
                    },
                )
            result["runtime_seconds"] = time.perf_counter() - started
            if isinstance(result.get("runtime"), dict):
                result["runtime"]["seconds"] = result["runtime_seconds"]
            result = annotate_result(result, self.scales, attempts)
            _safe_progress(
                reporter,
                "identity_search_complete",
                "success",
                f"身份检索完成：检测尺度 {attempts[-1]['scale']} 生效。",
                progress_percent=100.0,
                details={
                    "detection_scale_used": attempts[-1]["scale"],
                    "fallback_applied": attempts[-1]["scale"] != self.scales[0],
                    "attempts": attempts,
                },
                force=True,
            )
        except IdentitySearchError as exc:
            diagnostic_values = dict(getattr(exc, "diagnostics", {}) or {})
            diagnostic_values.setdefault("det_size_chain", list(self.scales))
            diagnostic_values.setdefault("attempts", attempts)
            result = error_result(
                str(query),
                str(exc),
                started,
                diagnostics=diagnostic_values,
                searcher=searcher,
                device=self.device,
                ctx_id=self.ctx_id,
            )
            method = result.setdefault("method", {})
            if isinstance(method, dict):
                method["preprocessing"] = {
                    "policy": FALLBACK_POLICY,
                    "det_size_chain": list(self.scales),
                    "library_det_size": self.scales[0],
                    "detection_scale_used": None,
                    "fallback_applied": False,
                    "attempts": attempts,
                }
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
            result = error_result(
                str(query),
                f"搜索过程异常: {type(exc).__name__}: {exc}",
                started,
                searcher=searcher,
                device=self.device,
                ctx_id=self.ctx_id,
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
        try:
            _write_json_atomic(output, result)
        except Exception as exc:
            original = result.get("error_message")
            write_error = f"无法写入输出 JSON {output}: {type(exc).__name__}: {exc}"
            result = dict(result)
            result.update(
                status="error",
                candidates=[],
                error_message=f"{original}；{write_error}" if original else write_error,
            )
        return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        scales = parse_scale_chain(args.det_size_chain)
    except IdentitySearchError as exc:
        _emit(
            "ready",
            request_id=None,
            ready=False,
            health="error",
            ok=False,
            error=str(exc),
        )
        return 2
    try:
        runtime = ScaleFallbackIdentityRuntime(args, scales)
    except Exception as exc:
        _emit(
            "ready",
            request_id=None,
            ready=False,
            health="error",
            ok=False,
            error=f"身份模型初始化失败：{type(exc).__name__}: {exc}",
            details={"worker": "identity_v2", "pid": os.getpid()},
        )
        return 1
    return _serve(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
