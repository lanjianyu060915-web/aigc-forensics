from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time


PRODUCT_ROOT = Path(__file__).resolve().parents[2]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))

from progress_events import create_progress_writer


RESULT_SCHEMA_VERSION = "3.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search a candidate library for the original source image."
    )
    parser.add_argument("--query", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--top-k", default="5")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task-id")
    parser.add_argument(
        "--progress-events",
        type=Path,
        help="统一进度 JSONL 路径；须与 --task-id/--progress-latest 同时提供",
    )
    parser.add_argument(
        "--progress-latest",
        type=Path,
        help="最新进度原子 JSON 路径；须与 --task-id/--progress-events 同时提供",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        help="库侧特征缓存目录；只省去重复计算，不改变检索分数",
    )
    return parser.parse_args()


def _is_within(path: Path, directory: Path) -> bool:
    """Return whether *path* resolves to *directory* or one of its children."""

    try:
        path.resolve(strict=False).relative_to(directory.resolve(strict=False))
    except ValueError:
        return False
    return True


def _write_payload(output: Path, payload: dict) -> None:
    """Atomically publish a complete result object."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _query_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query_digest(path: Path) -> str:
    return _query_sha256(path)[:12]


def _base_method() -> dict:
    return {
        "name": (
            "face-masked global retrieval plus SIFT geometric verification"
        ),
        "global_features": ["HSV", "grayscale", "edge", "border"],
        "local_features": "SIFT (nfeatures=2500)",
        "matcher": "BFMatcher L2",
        "lowe_ratio": 0.75,
        "ransac_threshold": 5.0,
        "local_score_formula": (
            "0.6 * ransac_inlier_ratio + "
            "0.4 * min(ransac_inlier_count / 40, 1)"
        ),
        "insufficient_match_formula": (
            "min(good_match_count / 40, 0.15)"
        ),
        "final_score_formula": (
            "0.45 * global_score + 0.55 * local_score"
        ),
        "global_weight": 0.45,
        "local_weight": 0.55,
    }


def _base_metrics() -> dict:
    return {
        "query_face_detected": None,
        "query_face_bbox": None,
        "face_mask_applied": None,
        "masked_query_path": None,
        "library_file_count": None,
        "library_decoded": None,
        "library_skipped_count": None,
        "library_skipped_paths": [],
        "shortlist_size": 0,
        "lowe_ratio": 0.75,
        "ransac_threshold": 5.0,
        "global_weight": 0.45,
        "local_weight": 0.55,
        "top1_top2_margin": None,
    }


def _base_payload(query: Path) -> dict:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "module": "source",
        # Frozen v2 fields.
        "status": "error",
        "query_image": str(query),
        "candidates": [],
        "runtime_seconds": 0.0,
        "error_message": None,
        # v3 blocks.
        "method": _base_method(),
        "metrics": _base_metrics(),
        "explanation": [],
        "warnings": [],
        "limitations": [
            "结果是当前素材库内的候选排序，不构成原始素材确认。",
            "分数是检索相似度，不是来源概率。",
        ],
        "runtime": {"seconds": 0.0},
        "provenance": {
            "schema_version": RESULT_SCHEMA_VERSION,
            "query_path": str(query),
            "input_sha256": None,
        },
        "visual_evidence": {"source": {}},
        "artifact_manifest": [],
        "diagnostics": {},
    }


def _safe_emit(reporter, stage: str, status: str, message: str, **kwargs) -> None:
    if reporter is None:
        return
    try:
        reporter.emit(stage, status, message, **kwargs)
    except Exception:
        return


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    payload = _base_payload(args.query)
    exit_code = 0
    reporter = None
    output_is_safe = not _is_within(args.output, args.library)
    try:
        reporter = create_progress_writer(
            task_id=args.task_id,
            module="source",
            events_path=args.progress_events,
            latest_path=args.progress_latest,
        )
        _safe_emit(
            reporter,
            "validate_source_input",
            "running",
            "正在校验素材搜索输入、素材库和输出位置。",
            progress_percent=None,
            details={
                "query_exists": args.query.is_file(),
                "library_exists": args.library.is_dir(),
            },
            force=True,
        )
        if not output_is_safe:
            raise ValueError(
                "output JSON must not be located inside the source library"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=args.output.parent / f"{args.output.stem}.runtime.log",
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            encoding="utf-8",
            force=True,
        )
        logging.info(
            "source search started query=%s library=%s top_k=%s",
            args.query,
            args.library,
            args.top_k,
        )
        try:
            top_k = int(args.top_k)
        except (TypeError, ValueError) as exc:
            raise ValueError("top-k must be an integer") from exc
        if top_k < 1:
            raise ValueError("top-k must be at least 1")
        if not args.query.is_file():
            raise ValueError(f"query image does not exist: {args.query}")
        if not args.library.is_dir():
            raise ValueError(
                f"library directory does not exist: {args.library}"
            )

        # Keep argparse and lightweight validation usable even when OpenCV is
        # unavailable. A formal invocation still reaches the structured error
        # payload below if this delayed import fails.
        from source_search import search_source_images

        input_sha256 = _query_sha256(args.query)
        candidates, diagnostics = search_source_images(
            args.query,
            args.library,
            top_k,
            args.output.parent,
            visualization_stem=(
                f"{args.output.stem}_{input_sha256[:12]}"
            ),
            progress_reporter=reporter,
            feature_cache_dir=args.feature_cache_dir,
        )
        runtime_seconds = diagnostics["runtime_seconds"]
        payload.update(
            status="success",
            candidates=candidates,
            runtime_seconds=runtime_seconds,
            error_message=None,
            method=diagnostics["method"],
            metrics=diagnostics["metrics"],
            explanation=diagnostics["explanation"],
            warnings=diagnostics["warnings"],
            limitations=diagnostics["limitations"],
            runtime={"seconds": runtime_seconds},
            provenance={
                **diagnostics["provenance"],
                "input_sha256": input_sha256,
            },
            visual_evidence=diagnostics.get(
                "visual_evidence", {"source": {}}
            ),
            artifact_manifest=diagnostics.get("artifact_manifest", []),
            diagnostics=diagnostics,
        )
        logging.info("source search success diagnostics=%s", diagnostics)
    except Exception as exc:
        runtime_seconds = round(time.perf_counter() - started, 6)
        payload.update(
            runtime_seconds=runtime_seconds,
            error_message=str(exc),
            runtime={"seconds": runtime_seconds},
            explanation=[f"素材搜索未完成：{exc}"],
            warnings=[str(exc)],
        )
        payload["diagnostics"] = {
            **payload.get("diagnostics", {}),
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
        }
        exit_code = 1
        _safe_emit(
            reporter,
            "source_search_failed",
            "error",
            str(exc),
            progress_percent=None,
            details={"failure_type": type(exc).__name__},
            force=True,
        )
        if output_is_safe:
            logging.exception("source search failed")
    if output_is_safe:
        try:
            _write_payload(args.output, payload)
        except OSError as exc:
            print(f"failed to write output JSON: {exc}", file=sys.stderr)
            return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
