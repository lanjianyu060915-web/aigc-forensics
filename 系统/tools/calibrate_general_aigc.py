"""由真实样本的分数算通用 AIGC 分支的分档边界，写进模型包的 calibration.json。

边界取真实样本分数的 p95 和 p99：高于 p95 进“需人工复核”，高于 p99 进“高风险提示”。
多个来源按来源等权（每个来源内部取分位点后不混样本量），避免大来源主导。
用法见 --help；输入是 eval_general_aigc.py 输出的逐样本 JSONL。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import json
from pathlib import Path
import statistics


def quantile(values: list[float], q: float) -> float:
    """线性插值分位点，和 numpy 默认口径一致。"""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def pooled_quantile(groups: dict[str, list[float]], q: float) -> float:
    """按来源等权的分位点：每条样本的权重是 1 / 该来源样本数。"""
    points = sorted(
        (value, 1.0 / len(values)) for values in groups.values() for value in values
    )
    total = sum(weight for _, weight in points)
    cumulative = 0.0
    for value, weight in points:
        cumulative += weight
        if cumulative / total >= q:
            return value
    return points[-1][0]


def load_real_scores(paths: list[Path], group_key: str) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if int(row["label"]) == 0 and row.get("logit") is not None:
                    groups[str(row.get(group_key) or path.stem)].append(float(row["logit"]))
    return dict(groups)


def main() -> int:
    parser = argparse.ArgumentParser(description="计算通用 AIGC 分支的分档边界")
    parser.add_argument("--kind", choices=("image", "video", "face_crop"), required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--group-key", default="source")
    parser.add_argument("--output", type=Path, required=True, help="模型包里的 calibration.json")
    parser.add_argument("--description", required=True, help="校准样本的说明，会显示在结果页")
    parser.add_argument("--version", default=date.today().strftime("calibration-%Y%m%d"))
    args = parser.parse_args()

    groups = load_real_scores(args.scores, args.group_key)
    if not groups:
        raise SystemExit("没有可用的真实样本分数")
    low_max = pooled_quantile(groups, 0.95)
    medium_max = pooled_quantile(groups, 0.99)
    payload = json.loads(args.output.read_text(encoding="utf-8")) if args.output.is_file() else {}
    payload["version"] = args.version
    payload["semantics"] = "边界是真实样本 logit 的 p95 和 p99，按来源等权；不是概率。"
    payload[args.kind] = {
        "low_max": round(low_max, 6),
        "medium_max": round(medium_max, 6),
        "calibration_set": {
            "description": args.description,
            "n_real": sum(len(values) for values in groups.values()),
            "sources": {name: len(values) for name, values in sorted(groups.items())},
            "per_source_p95": {name: round(quantile(values, 0.95), 4) for name, values in sorted(groups.items())},
            "per_source_median": {name: round(statistics.median(values), 4) for name, values in sorted(groups.items())},
        },
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload[args.kind], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
