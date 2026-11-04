# -*- coding: utf-8 -*-
"""逐案例比对：冻结 source_search.exe 与带特征缓存的 Python 底图检索。

用法：
  python -B tools/verify_source_cache.py --queries <目录或文件...> --library <底图库> --out <输出 JSON>
只读输入，所有产物写到 --work 指定的目录。Top-K 顺序不同或分数差超过容差即判不通过。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "packaging_v0.3" / "work" / "dist" / "app" / "workers" / "source" / "source_search.exe"
RUNNER = ROOT / "modules" / "source_search" / "run_source_search.py"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TOLERANCE = 1e-5
COUNT_FIELDS = ("good_match_count", "ransac_inlier_count")
SCORE_FIELDS = ("global_score", "local_score", "final_score")


def collect_queries(values: list[str]) -> list[Path]:
    queries: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            queries.extend(
                item for item in sorted(path.rglob("*")) if item.suffix.lower() in IMAGE_SUFFIXES
            )
        else:
            queries.append(path)
    return queries


def run(command: list[str], output: Path, cwd: Path) -> tuple[dict, float]:
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=str(cwd), capture_output=True, timeout=1200,
        encoding="utf-8", errors="replace",
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(f"退出码 {completed.returncode}：{completed.stderr[-400:]}")
    return json.loads(output.read_text(encoding="utf-8")), elapsed


def compare(frozen: dict, cached: dict) -> dict:
    first = frozen.get("candidates") or []
    second = cached.get("candidates") or []
    differences: list[str] = []
    worst = 0.0
    if [Path(c["file_path"]).name for c in first] != [Path(c["file_path"]).name for c in second]:
        differences.append("Top-K 顺序不同")
    for left, right in zip(first, second):
        for field in SCORE_FIELDS:
            delta = abs(float(left[field]) - float(right[field]))
            worst = max(worst, delta)
            if delta > TOLERANCE:
                differences.append(f"第{left['rank']}名 {field} 差 {delta:.2e}")
        for field in COUNT_FIELDS:
            if int(left[field]) != int(right[field]):
                differences.append(f"第{left['rank']}名 {field}: {left[field]} != {right[field]}")
    return {"passed": not differences, "max_score_delta": worst, "differences": differences}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top-k", default="5")
    args = parser.parse_args()

    cache = args.work / "feature_cache"
    rows = []
    queries = collect_queries(args.queries)
    for index, query in enumerate(queries, 1):
        case = args.work / f"case_{index:03d}"
        frozen_out = case / "frozen" / "source_result.json"
        cached_out = case / "cached" / "source_result.json"
        warm_out = case / "warm" / "source_result.json"
        common = ["--query", str(query), "--library", str(args.library), "--top-k", args.top_k]
        for path in (frozen_out, cached_out, warm_out):
            path.parent.mkdir(parents=True, exist_ok=True)
        frozen, frozen_seconds = run(
            [str(EXE), *common, "--output", str(frozen_out)], frozen_out, EXE.parent
        )
        python_command = [sys.executable, "-B", str(RUNNER), *common, "--feature-cache-dir", str(cache)]
        cached, cached_seconds = run(
            [*python_command, "--output", str(cached_out)], cached_out, RUNNER.parent
        )
        warm, warm_seconds = run(
            [*python_command, "--output", str(warm_out)], warm_out, RUNNER.parent
        )
        result = compare(frozen, cached)
        warm_result = compare(frozen, warm)
        rows.append(
            {
                "query": str(query),
                "frozen_seconds": round(frozen_seconds, 2),
                "cached_first_seconds": round(cached_seconds, 2),
                "cached_warm_seconds": round(warm_seconds, 2),
                "passed": result["passed"] and warm_result["passed"],
                "max_score_delta": max(result["max_score_delta"], warm_result["max_score_delta"]),
                "differences": result["differences"] + warm_result["differences"],
                "cache_stats": (warm.get("metrics") or {}).get("feature_cache"),
            }
        )
        print(
            f"[{index}/{len(queries)}] {query.parent.parent.name}/{query.name} "
            f"冻结 {frozen_seconds:.1f}s 首次 {cached_seconds:.1f}s 复用 {warm_seconds:.1f}s "
            f"最大分差 {rows[-1]['max_score_delta']:.2e} {'通过' if rows[-1]['passed'] else '不通过'}",
            flush=True,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = [row for row in rows if not row["passed"]]
    print(f"合计 {len(rows)} 例，不通过 {len(failed)} 例")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
