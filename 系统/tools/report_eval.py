"""读逐样本分数，按来源报 EER、AUC 和固定边界下的检出与误报。

    python -B tools/report_eval.py --scores a.scores.jsonl [b.scores.jsonl ...] --group-by source
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from aigc_engine.metrics import UndefinedMetricError, evaluate_binary  # noqa: E402


def load(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def summarize(rows: list[dict[str, Any]], *, field: str, thresholds: dict[str, float] | None) -> dict[str, Any]:
    ok = [row for row in rows if isinstance(row.get(field), (int, float))]
    labels = [int(row["label"]) for row in ok]
    scores = [float(row[field]) for row in ok]
    result: dict[str, Any] = {
        "n": len(rows),
        "scored": len(ok),
        "failed": len(rows) - len(ok),
        "real": labels.count(0),
        "fake": labels.count(1),
    }
    try:
        metrics = evaluate_binary(labels, scores)
        result["eer"] = round(metrics.eer * 100, 2)
        result["auc"] = round(metrics.roc_auc, 4)
    except UndefinedMetricError:
        result["eer"] = result["auc"] = None
    if thresholds:
        real = [score for score, label in zip(scores, labels, strict=True) if label == 0]
        fake = [score for score, label in zip(scores, labels, strict=True) if label == 1]
        for name, value in thresholds.items():
            result[f"real_at_or_above_{name}"] = _share(real, value)
            result[f"fake_at_or_above_{name}"] = _share(fake, value)
    return result


def _share(values: list[float], threshold: float) -> dict[str, Any] | None:
    if not values:
        return None
    hits = sum(value >= threshold for value in values)
    return {"count": hits, "of": len(values), "percent": round(100 * hits / len(values), 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description="按来源统计评测指标")
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--group-by", default="source")
    parser.add_argument("--fields", nargs="+", default=["logit", "clip_logit", "community_logit"])
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--kind", choices=("image", "video"), default="video")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    thresholds = None
    if args.calibration is not None:
        band = json.loads(args.calibration.read_text(encoding="utf-8")).get(args.kind)
        if isinstance(band, dict):
            thresholds = {"medium": float(band["low_max"]), "high": float(band["medium_max"])}
    rows = load(args.scores)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(args.group_by))].append(row)
    report: dict[str, Any] = {"thresholds": thresholds, "groups": {}}
    for name, items in sorted(groups.items()):
        report["groups"][name] = {
            field: summarize(items, field=field, thresholds=thresholds if field == "logit" else None)
            for field in args.fields
        }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    for name, fields in report["groups"].items():
        print(f"== {name}")
        for field, values in fields.items():
            print(f"   {field:18s} n={values['scored']:4d} real={values['real']:4d} fake={values['fake']:4d} EER={values['eer']} AUC={values['auc']}")
        extra = fields.get("logit", {})
        if thresholds:
            for key in ("medium", "high"):
                print(
                    f"   >= {key:6s} real {extra.get(f'real_at_or_above_{key}')} fake {extra.get(f'fake_at_or_above_{key}')}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
