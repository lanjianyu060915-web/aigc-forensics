"""Independent protocol fixture. Never loads models, libraries or media."""
import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="ready")
args = parser.parse_args()


def emit(kind, **fields):
    value = {"protocol": "face-source-trace-runtime-v1", "type": kind,
             "request_id": None, "ready": False, "health": "starting", **fields}
    print(json.dumps(value), flush=True)


emit("startup_progress", stage="importing", stage_status="running", message="synthetic import")
emit("startup_progress", stage="loading_model", stage_status="running", message="synthetic load")
if args.mode == "hang":
    time.sleep(5)
if args.mode == "fail":
    emit("ready", ready=False, health="error", error="synthetic load failure")
    raise SystemExit(2)
emit("startup_progress", stage="checking_library", stage_status="running", message="synthetic cache check")
emit("startup_progress", stage="rebuilding_index", stage_status="running", message="synthetic index",
     details={"progress_current": 1, "progress_total": 2})
emit("startup_progress", stage="warming_up", stage_status="skipped", message="no synthetic warmup")
emit("ready", ready=True, health="ok", ok=True, details={"worker": "synthetic", "pid": os.getpid()})
for line in sys.stdin:
    request = json.loads(line)
    if request.get("action") == "wait":
        time.sleep(0.1)
    emit("response", request_id=request["request_id"], ready=True, health="ok", ok=request.get("action") != "reject",
         error="synthetic request rejected" if request.get("action") == "reject" else None,
         payload={"status": "success", "synthetic": True})
    if request.get("action") == "shutdown":
        break
