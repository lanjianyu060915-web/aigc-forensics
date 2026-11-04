"""Explicit real-model integration runner.

This script is intentionally excluded from the fast unit-test suite.  It starts
the configured persistent AIGC/identity workers, runs one or more genuine tasks
against the configured libraries, and archives a compact machine-readable
summary.  No mock result is accepted as a pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))

from orchestrator import MODULE_ORDER, run_task
from runtime_services import RuntimeServiceManager


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用真实模型、真实图库与持久 worker 运行三模块集成回归。"
    )
    parser.add_argument("--query", action="append", required=True, help="查询图片；可重复")
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=MODULE_ORDER,
        default=list(MODULE_ORDER),
        help="要运行的模块；默认三个模块全选",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PRODUCT_ROOT / "config.json",
        help="配置 JSON",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PRODUCT_ROOT / "runtime" / "validation" / "real_integration_summary.json",
        help="汇总 JSON 输出路径",
    )
    return parser


def _progress(event: Any) -> None:
    if isinstance(event, dict):
        message = event.get("message") or json.dumps(event, ensure_ascii=False)
    else:
        message = str(event)
    print(f"[真实回归] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=False)
    summary_path = args.summary.expanduser().resolve(strict=False)
    manager = RuntimeServiceManager(config_path, progress_callback=_progress)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "selected_modules": list(args.modules),
        "uses_mock": False,
        "service_start": {},
        "service_final": {},
        "tasks": [],
    }
    exit_code = 0
    try:
        report["service_start"] = manager.start()
        print(json.dumps(report["service_start"], ensure_ascii=False, indent=2), flush=True)
        for query_value in args.query:
            query = Path(query_value).expanduser().resolve(strict=False)
            try:
                task = run_task(
                    query,
                    selected_modules=args.modules,
                    services=manager,
                    config_path=config_path,
                    progress_callback=_progress,
                )
                report["tasks"].append(task)
                if task.get("status") != "success":
                    exit_code = 1
            except Exception as exc:
                exit_code = 1
                report["tasks"].append(
                    {
                        "query": str(query),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        report["service_final"] = manager.get_status()
    finally:
        manager.close()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = summary_path.with_name(f".{summary_path.name}.tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(summary_path)
    print(f"真实回归汇总：{summary_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

