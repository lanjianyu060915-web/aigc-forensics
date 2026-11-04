# -*- coding: utf-8 -*-
"""素材检索的开放集评测：看全局分阈值在非人脸素材上是否还能分开库内和库外。

用界面环境运行。输入是一张清单（JSONL，字段 sample_id、path、role）：
- role=library：放进临时素材库
- role=positive：库内图的退化版本，应判高可信，top1 应是 expect 字段指向的图
- role=negative：库外图，应判未发现可靠候选
输出逐条 JSONL，最后打印各类的四态分布。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "modules" / "source_search"):
    sys.path.insert(0, str(import_root))

from decision.open_set import decide_source_open_set  # noqa: E402
from source_search.pipeline import search_source_images  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True, help="临时库和缓存放这里")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    logging.disable(logging.INFO)
    samples = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    library = args.work / "library"
    library.mkdir(parents=True, exist_ok=True)
    names: dict[str, str] = {}
    for sample in samples:
        if sample["role"] != "library":
            continue
        source = Path(sample["path"])
        name = f"{sample['sample_id']}{source.suffix.lower()}"
        names[sample["sample_id"]] = name
        target = library / name
        if not target.exists():
            shutil.copyfile(source, target)
    print(f"库内 {len(names)} 张", flush=True)

    done: set[str] = set()
    if args.output.is_file():
        done = {json.loads(line)["sample_id"] for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()}
    queries = [sample for sample in samples if sample["role"] != "library" and sample["sample_id"] not in done]
    cache = args.work / "cache"
    with args.output.open("a", encoding="utf-8") as stream:
        for position, sample in enumerate(queries, start=1):
            started = time.perf_counter()
            with tempfile.TemporaryDirectory(dir=args.work) as scratch:
                candidates, diagnostics = search_source_images(
                    Path(sample["path"]), library, 5, Path(scratch), feature_cache_dir=cache
                )
            verdict = decide_source_open_set(
                {"status": "success", "candidates": candidates, "metrics": diagnostics["metrics"]}
            )
            top = candidates[0] if candidates else {}
            stream.write(
                json.dumps(
                    {
                        "sample_id": sample["sample_id"],
                        "role": sample["role"],
                        "kind": sample.get("kind"),
                        "expect": names.get(sample.get("expect", "")),
                        "top1": Path(str(top.get("file_path") or "")).name,
                        "global_score": top.get("global_score"),
                        "local_score": top.get("local_score"),
                        "final_score": top.get("final_score"),
                        "state": verdict.get("state"),
                        "seconds": round(time.perf_counter() - started, 2),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stream.flush()
            if position % 50 == 0:
                print(f"{position}/{len(queries)}", flush=True)

    rows = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
    for role in ("positive", "negative"):
        subset = [row for row in rows if row["role"] == role]
        print(role, len(subset), dict(Counter(row["state"] for row in subset)))
        if role == "positive" and subset:
            print("  top1 命中", sum(row["top1"] == row["expect"] for row in subset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
