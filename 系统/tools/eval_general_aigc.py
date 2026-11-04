"""用通用 AIGC 分支批量打分并统计指标。

输入是 JSONL 清单，每行至少有 path 和 label（0 真实、1 生成），其余字段原样带进结果。
已有结果的样本自动跳过，方便中断后接着跑。用引擎环境运行：

    engine_runtime\\python.exe -B tools\\eval_general_aigc.py --manifest m.jsonl --output out\\scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "engine"))
sys.path.insert(0, str(APP_ROOT / "modules" / "general_aigc"))

from aigc_engine.media import MediaError  # noqa: E402
from aigc_engine.metrics import UndefinedMetricError, evaluate_binary  # noqa: E402
from aigc_engine.scorer import GeneralAigcScorer  # noqa: E402
from general_aigc_core import input_kind  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("logit") is not None]
    labels = [int(row["label"]) for row in ok]
    scores = [float(row["logit"]) for row in ok]
    summary: dict[str, Any] = {
        "total": len(rows),
        "scored": len(ok),
        "failed": len(rows) - len(ok),
        "real": labels.count(0),
        "fake": labels.count(1),
    }
    try:
        metrics = evaluate_binary(labels, scores)
        summary["eer"] = metrics.eer
        summary["auc"] = metrics.roc_auc
    except UndefinedMetricError as exc:
        summary["metric_error"] = str(exc)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="通用 AIGC 分支批量评测")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="逐样本结果 JSONL，同名 .summary.json 写汇总")
    parser.add_argument("--root", default=None, help="清单里相对路径的根目录，默认是清单所在目录")
    parser.add_argument("--model-package", default=str(APP_ROOT / "models" / "video-c38-refit3-cf384"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--group-by", nargs="*", default=["source"])
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root = Path(args.root) if args.root else manifest_path.parent
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = _read_jsonl(manifest_path)
    done = {row["sample_id"]: row for row in _read_jsonl(output)} if output.is_file() else {}

    package = Path(args.model_package)
    descriptor = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    scorer = GeneralAigcScorer(
        package / descriptor["checkpoint"], sha256=descriptor["checkpoint_sha256"], device=args.device
    )
    print(f"模型加载 {scorer.load_seconds:.1f} 秒，设备 {scorer.device_name}", flush=True)
    started = time.perf_counter()
    with output.open("a", encoding="utf-8") as stream:
        for index, sample in enumerate(samples, start=1):
            sample_id = str(sample.get("sample_id") or sample["path"])
            if sample_id in done:
                continue
            row = {**sample, "sample_id": sample_id, "logit": None, "error": None}
            path = Path(sample["path"])
            path = path if path.is_absolute() else root / path
            try:
                result = (
                    scorer.score_image(path)
                    if input_kind(path) == "image"
                    else scorer.score_video(path)
                )
                row.update(
                    logit=result.logit,
                    clip_logit=result.clip_logit,
                    clip_original_logit=result.clip_original_logit,
                    clip_refit_logit=result.clip_refit_logit,
                    community_logit=result.community_logit,
                    width=result.width,
                    height=result.height,
                    decode_mode=result.decode_mode,
                )
            except (MediaError, OSError, ValueError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            done[sample_id] = row
            if index % 50 == 0:
                print(f"{index}/{len(samples)}，已用 {time.perf_counter() - started:.0f} 秒", flush=True)

    rows = [done[str(sample.get("sample_id") or sample["path"])] for sample in samples]
    report: dict[str, Any] = {"overall": summarize(rows), "model_id": descriptor["model_id"]}
    for key in args.group_by:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(key))].append(row)
        report[f"by_{key}"] = {name: summarize(items) for name, items in sorted(groups.items())}
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
