"""通用 AIGC 分支单次运行入口：常驻服务不可用时由编排层调用。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通用 AIGC 分支单次运行")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-package", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--face-crops-json", default=None, help="人脸裁剪图路径列表（JSON 文件）")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = Path(args.output)
    started = time.perf_counter()
    try:
        from runtime import DEFAULT_PACKAGE, GeneralAigcRuntime

        runtime = GeneralAigcRuntime(
            Path(args.model_package) if args.model_package else DEFAULT_PACKAGE,
            device=args.device,
        )
    except Exception as exc:  # noqa: BLE001 - 模型加载失败也要写出结构化结果
        from general_aigc_core import error_payload, write_json

        write_json(
            output,
            error_payload(
                Path(args.input),
                f"通用 AIGC 模型加载失败：{type(exc).__name__}: {exc}",
                started=started,
                error_type=type(exc).__name__,
            ),
        )
        print(f"通用 AIGC 模型加载失败：{exc}", file=sys.stderr)
        return 2
    face_crops = None
    if args.face_crops_json:
        face_crops = json.loads(Path(args.face_crops_json).read_text(encoding="utf-8"))
    payload = runtime.run(args.input, output, face_crops)
    return 0 if payload.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
