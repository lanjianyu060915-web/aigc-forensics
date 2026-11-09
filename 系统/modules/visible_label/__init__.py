"""隔离运行离线画面文字识别；只给线索，不改变检测档位。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

LIMITATION = "没识别到不代表不是生成内容；字样可以被添加、裁掉，或被压缩到认不出。这里只是文字识别，不核验来源。"
WORDLIST_NOTE = "词表是起点，尚未在真实工具输出上核实。"
VERDICT_LABELS = {
    "visible_label_found": "识别到生成相关字样",
    "tool_hint_found": "识别到边缘工具名字样",
    "no_label": "未识别到词表内字样",
    "unavailable": "画面文字识别未完成",
}


def unavailable(reason: str, elapsed: float = 0.0) -> dict:
    return {"status": "error", "verdict": "unavailable", "hits": [],
            "runtime_seconds": round(elapsed, 6), "limitations": [LIMITATION, WORDLIST_NOTE],
            "error_message": reason}


def analyze_visible_labels(inputs: list[dict], *, timeout_seconds: float = 30.0) -> dict:
    """最多读取五帧。子进程超时即终止，故障不外抛给检测任务。"""
    started = time.perf_counter()
    try:
        frames = []
        seen = set()
        for item in inputs:
            path = str(Path(item["path"]).resolve())
            if path in seen:
                continue
            seen.add(path)
            frames.append({"path": path, "timestamp_seconds": item.get("timestamp_seconds")})
            if len(frames) == 5:
                break
        if not frames:
            return unavailable("没有可用画面。")
        environment = os.environ.copy()
        environment.update({"PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                            "TRANSFORMERS_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"})
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).with_name("worker.py"))],
            input=json.dumps({"inputs": frames}, ensure_ascii=False), capture_output=True,
            encoding="utf-8", errors="replace", env=environment,
            timeout=min(30.0, max(0.01, float(timeout_seconds))),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode:
            return unavailable("离线文字识别进程未完成。", time.perf_counter() - started)
        result = json.loads(completed.stdout)
        if not isinstance(result, dict) or result.get("verdict") not in VERDICT_LABELS:
            raise ValueError("识别结果格式无效")
        result["runtime_seconds"] = round(time.perf_counter() - started, 6)
        result["limitations"] = [LIMITATION, WORDLIST_NOTE]
        result["verdict_label"] = VERDICT_LABELS[result["verdict"]]
        return result
    except subprocess.TimeoutExpired:
        return unavailable("画面文字识别超时，已停止本模块。", time.perf_counter() - started)
    except Exception as error:
        return unavailable(f"画面文字识别不可用（{type(error).__name__}）。", time.perf_counter() - started)


def visible_label_lines(result: dict) -> list[str]:
    """界面和三种报告使用同一份文字，旧任务无记录时不增添区块。"""
    if not result:
        return []
    from reporting.text_safety import redact_local_paths

    lines = ["画面文字标识", VERDICT_LABELS.get(result.get("verdict"), "画面文字识别未完成")]
    if result.get("error_message"):
        lines.append(redact_local_paths(str(result["error_message"])))
    for hit in result.get("hits") or []:
        location = "画面边缘" if hit.get("in_corner") else "画面内部"
        timestamp = hit.get("timestamp_seconds")
        frame = f"；时间点 {timestamp} 秒" if timestamp is not None else ""
        lines.append(f"字样：{redact_local_paths(str(hit.get('text', '')))}；位置：{location}，"
                     f"原图坐标 {hit.get('bbox')}；识别置信度 {hit.get('confidence')}{frame}")
    lines.extend(result.get("limitations") or [LIMITATION, WORDLIST_NOTE])
    lines.append("文字识别结果不参与检测档位，也不核验工具或来源。")
    return lines
