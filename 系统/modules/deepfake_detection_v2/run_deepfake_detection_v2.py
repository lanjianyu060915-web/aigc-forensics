"""二代鉴伪内核的单次运行入口。

常驻服务不可用或从命令行调用时走这里，保证与常驻路径用同一套路由和模型：
短边不低于路由阈值走整图 ViT 分支，低于阈值代理到冻结人脸裁剪分支。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from persistent_worker_v2 import DEFAULT_APP_ROOT, PersistentVitRuntime, _base_payload


def _default_writable_root() -> Path:
    data_root = os.environ.get("FSTS_DATA_ROOT")
    base = Path(data_root) / "runtime" / "persistent_services" if data_root else Path(tempfile.gettempdir())
    return base / "deepfake_v2_oneshot"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="二代鉴伪内核单次运行")
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--app-root", default=str(DEFAULT_APP_ROOT))
    parser.add_argument(
        "--install-root",
        default=str(DEFAULT_APP_ROOT / "packaging_v0.3" / "resources" / "deepfake_runtime"),
    )
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument("--writable-root", default=str(_default_writable_root()))
    parser.add_argument("--no-hash-check", action="store_true")
    parser.add_argument("--no-delegation", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    query = Path(args.query).expanduser().resolve(strict=False)
    output = Path(args.output).expanduser().resolve(strict=False)
    runtime = None
    try:
        runtime = PersistentVitRuntime(args)
        payload = runtime.run(query, output)
    except Exception as exc:  # noqa: BLE001
        payload = _base_payload(query, 0.0)
        payload["error_message"] = f"二代鉴伪内核启动失败：{type(exc).__name__}: {exc}"
        payload["details"] = {"error_type": type(exc).__name__}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        if runtime is not None and runtime.frozen is not None:
            runtime.frozen.stop()
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
