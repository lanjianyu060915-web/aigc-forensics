"""Model-free protocol fixture used by test_runtime_services.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any


PROTOCOL_NAME = "face-source-trace-runtime-v1"


def emit(message_type: str, request_id: str | None, **fields: Any) -> None:
    payload = {
        "protocol": PROTOCOL_NAME,
        "type": message_type,
        "request_id": request_id,
        "ready": True,
        "health": "ok",
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


# Both plain text and unrelated JSON simulate third-party import noise.
print("mock third-party banner", flush=True)
print(json.dumps({"third_party": "json noise"}), flush=True)
emit("ready", None, ok=True, details={"pid": os.getpid()})

for line in sys.stdin:
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        continue
    if request.get("protocol") != PROTOCOL_NAME:
        continue
    request_id = request.get("request_id")
    action = request.get("action")
    if action == "shutdown":
        emit("response", request_id, ok=True, shutting_down=True)
        break
    if action == "health":
        emit("health", request_id, ok=True, details={"pid": os.getpid()})
        continue
    if action != "run":
        emit("response", request_id, ok=False, error="unsupported action")
        continue

    query_path = str(request.get("query_path", ""))
    if Path(query_path).name.startswith("timeout"):
        time.sleep(2.0)
    if Path(query_path).name.startswith("exit"):
        os._exit(17)

    output = Path(str(request["output_path"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    if "library_path" in request:
        payload = {
            "status": "success",
            "query_image": query_path,
            "candidates": [],
            "runtime_seconds": 0.001,
            "error_message": None,
            "mock_top_k": request.get("top_k"),
        }
    else:
        payload = {
            "status": "success",
            "query_image": query_path,
            "is_deepfake": False,
            "score": 0.1,
            "label": "real",
            "runtime_seconds": 0.001,
            "details": {"fixture": True},
            "error_message": None,
        }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    emit("response", request_id, ok=True, payload=payload)
