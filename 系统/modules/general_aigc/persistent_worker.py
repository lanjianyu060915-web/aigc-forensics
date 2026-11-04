"""通用 AIGC 分支常驻进程，JSON 行协议与其他模块一致（face-source-trace-runtime-v1）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

PROTOCOL_NAME = "face-source-trace-runtime-v1"
PROTOCOL_OUTPUT: TextIO = sys.stdout


def _emit(message_type: str, *, request_id: str | None, ready: bool, health: str, **fields: Any) -> None:
    message = {
        "protocol": PROTOCOL_NAME,
        "type": message_type,
        "request_id": request_id,
        "ready": bool(ready),
        "health": str(health),
        **fields,
    }
    PROTOCOL_OUTPUT.write("\n" + json.dumps(message, ensure_ascii=False) + "\n")
    PROTOCOL_OUTPUT.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通用 AIGC 分支常驻进程")
    parser.add_argument("--model-package", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _serve(runtime: Any) -> int:
    _emit("ready", request_id=None, ready=True, health="ok", ok=True, details=runtime.health_details())
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(message, dict)
            or message.get("protocol") != PROTOCOL_NAME
            or message.get("type") != "request"
        ):
            continue
        request_id = message.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        action = message.get("action")
        if action == "health":
            _emit("health", request_id=request_id, ready=True, health="ok", ok=True, details=runtime.health_details())
            continue
        if action == "shutdown":
            _emit("response", request_id=request_id, ready=True, health="ok", ok=True, shutting_down=True)
            return 0
        if action != "run":
            _emit("response", request_id=request_id, ready=True, health="ok", ok=False, error=f"不支持的 action：{action!r}")
            continue
        payload = runtime.run(
            message.get("input_path"), message.get("output_path"), message.get("face_crops")
        )
        # 单个文件解码失败是这次结果的一部分，不算服务故障，编排层按 payload 的 status 处理。
        _emit("response", request_id=request_id, ready=True, health="ok", ok=True, payload=payload)
    return 0


def main() -> int:
    args = _parser().parse_args()
    # 第三方库偶尔往 stdout 打印，协议只走保留下来的原始 stdout。
    global PROTOCOL_OUTPUT
    PROTOCOL_OUTPUT = sys.stdout
    sys.stdout = sys.stderr
    try:
        from runtime import DEFAULT_PACKAGE, GeneralAigcRuntime

        runtime = GeneralAigcRuntime(
            Path(args.model_package) if args.model_package else DEFAULT_PACKAGE,
            device=args.device,
        )
    except Exception as exc:  # noqa: BLE001 - 启动失败要作为协议消息报告，由编排层回退
        _emit("ready", request_id=None, ready=False, health="error", ok=False, error=f"{type(exc).__name__}: {exc}")
        return 2
    return _serve(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
