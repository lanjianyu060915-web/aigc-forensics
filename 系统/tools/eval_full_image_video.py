# -*- coding: utf-8 -*-
"""用真实编排路径（抽帧 → 关键帧 → 逐帧鉴伪 → 聚合）批量评测整图/人脸分支在视频上的表现。

只开 deepfake 模块、关掉通用分支，结果写成和 eval_general_aigc.py 相同的 JSONL。已有结果自动跳过。
    python_runtime\\python.exe -B -X utf8 tools\\eval_full_image_video.py --manifest m.jsonl --output out.jsonl --work 工作目录
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    os.environ["FSTS_DATA_ROOT"] = str(args.work)
    sys.path.insert(0, str(ROOT))
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    config.pop("general_aigc", None)
    config.pop("provenance", None)
    config["identity"]["persistent_enabled"] = False
    # {APP_ROOT} 按配置文件所在目录解析，所以临时配置要和正式配置放在一起。
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix=".eval_config_", dir=ROOT, delete=False, encoding="utf-8"
    ) as handle:
        json.dump(config, handle, ensure_ascii=False)
        config_path = Path(handle.name)

    from orchestrator import run_task
    from runtime_services import RuntimeServiceManager

    samples = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    done = set()
    if args.output.is_file():
        done = {json.loads(line)["sample_id"] for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manager = RuntimeServiceManager(config_path)
    started = time.perf_counter()
    try:
        manager.start()
        with args.output.open("a", encoding="utf-8") as stream:
            for index, sample in enumerate(samples, start=1):
                if sample["sample_id"] in done:
                    continue
                row = {**sample, "score": None, "band": None, "error": None}
                try:
                    status = run_task(
                        sample["path"],
                        selected_modules={"deepfake"},
                        services=manager,
                        config_path=config_path,
                    )
                    task_dir = Path(status["task_dir"])
                    result = json.loads((task_dir / "detection_result.json").read_text(encoding="utf-8"))
                    details = result.get("details") or {}
                    row.update(
                        score=result.get("score"),
                        band=details.get("band"),
                        frame_band_counts=details.get("frame_band_counts"),
                        error=result.get("error_message"),
                    )
                    shutil.rmtree(task_dir, ignore_errors=True)
                except Exception as exc:  # noqa: BLE001 - 单条失败只记录
                    row["error"] = f"{type(exc).__name__}: {exc}"
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if index % 20 == 0:
                    print(f"{index}/{len(samples)}，已用 {time.perf_counter() - started:.0f} 秒", flush=True)
    finally:
        manager.close()
        config_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
