from __future__ import annotations

import argparse
import sys
from pathlib import Path

from orchestrator import DEFAULT_CONFIG, MODULE_ORDER, OrchestrationError, run_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行一次鉴伪 + 身份搜索 + 原始底图搜索三分支集成任务。"
    )
    parser.add_argument("--query", required=True, help="查询图片路径")
    parser.add_argument("--identity-library", help="可选：覆盖身份库路径")
    parser.add_argument("--source-library", help="可选：覆盖底图库路径")
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=MODULE_ORDER,
        metavar="MODULE",
        help="可选：仅运行指定模块；默认运行 deepfake identity source",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="配置 JSON 路径",
    )
    return parser


def _print_progress(event: object) -> None:
    message = event.get("message") if isinstance(event, dict) else event
    print(f"[进度] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        status = run_task(
            args.query,
            identity_library=args.identity_library,
            source_library=args.source_library,
            selected_modules=args.modules,
            config_path=args.config,
            progress_callback=_print_progress,
        )
    except (OrchestrationError, OSError, RuntimeError, ValueError) as exc:
        print(f"受控错误：{exc}", file=sys.stderr)
        return 2

    print(f"task_id: {status['task_id']}")
    print(f"task_dir: {status['task_dir']}")
    print(f"status: {status['status']}")
    print(f"selected_modules: {','.join(status['selected_modules'])}")
    print(f"deepfake_status: {status['deepfake']['status']}")
    print(f"identity_status: {status['identity']['status']}")
    print(f"source_status: {status['source']['status']}")
    print(f"deepfake_runtime: {status['deepfake']['runtime_seconds']:.6f}")
    print(f"identity_runtime: {status['identity']['runtime_seconds']:.6f}")
    print(f"source_runtime: {status['source']['runtime_seconds']:.6f}")
    print(f"total_runtime: {status['runtime_seconds']:.6f}")
    return 0 if status["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
