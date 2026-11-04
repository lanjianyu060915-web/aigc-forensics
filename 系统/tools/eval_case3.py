# -*- coding: utf-8 -*-
"""在当前系统上真实跑一遍案例3 的 20 个案例（身份 + 素材），输出命中情况和四态分布。

用法：python -B tools/eval_case3.py --case3 <案例3 目录> --libraries <案例3 库目录> --work <工作目录> --out <输出 JSON>
身份真值：案例“身份图片”的文件名前缀（公众人物）或 caseN；素材真值：按内容 SHA-256 比较，
因为案例3 的原始图片存在跨案例重复，按文件名判会误判。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "frontend"):
    sys.path.insert(0, str(import_root))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PUBLIC_PREFIXES = ("特朗普", "高市早苗", "尹锡悦", "韩圣淑", "拜登", "菅义伟", "李在名")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_image(folder: Path) -> Path:
    return next(path for path in sorted(folder.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES)


def identity_key(case_dir: Path) -> str:
    names = sorted(item.name for item in (case_dir / "身份图片").iterdir() if item.is_file())
    stem = Path(names[0]).stem if names else ""
    for prefix in PUBLIC_PREFIXES:
        if stem.startswith(prefix):
            return prefix
    return f"case{case_dir.name}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case3", required=True, type=Path)
    parser.add_argument("--libraries", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    os.environ["FSTS_DATA_ROOT"] = str(args.work)

    from components.task_history import summarize_task
    from orchestrator import run_task

    identity_library = args.libraries / "identity_library_case3"
    source_library = args.libraries / "source_library_case3"
    rows = []
    for case in range(1, 21):
        case_dir = args.case3 / str(case)
        query = first_image(case_dir / "AI图片")
        truth_hash = sha256(first_image(case_dir / "原始图片"))
        expected_identity = identity_key(case_dir)
        started = time.perf_counter()
        status = run_task(
            query,
            identity_library=identity_library,
            source_library=source_library,
            selected_modules=("identity", "source"),
            config_path=ROOT / "config.json",
            original_file_name=f"案例{case}_{query.name}",
        )
        elapsed = time.perf_counter() - started
        task_dir = Path(str(status["task_dir"]))
        identity = json.loads((task_dir / "identity_result.json").read_text(encoding="utf-8"))
        source = json.loads((task_dir / "source_result.json").read_text(encoding="utf-8"))
        summary = summarize_task(task_dir) or {}
        identity_candidates = identity.get("candidates") or []
        source_candidates = source.get("candidates") or []
        top_identity = identity_candidates[0].get("identity_id") if identity_candidates else None
        top_source = source_candidates[0].get("file_path") if source_candidates else None
        rows.append(
            {
                "case": case,
                "query": query.name,
                "seconds": round(elapsed, 2),
                "identity_status": identity.get("status"),
                "identity_error": identity.get("error_message"),
                "identity_expected": expected_identity,
                "identity_top": top_identity,
                "identity_hit": top_identity == expected_identity,
                "identity_state": summary.get("identity_label"),
                "identity_score": summary.get("identity_score"),
                "query_face_count": (identity.get("metrics") or {}).get("query_face_count_original") or 1,
                "main_face_selected": bool((identity.get("metrics") or {}).get("main_face_selected")),
                "source_status": source.get("status"),
                "source_top": Path(top_source).name if top_source else None,
                "source_hit": bool(top_source) and sha256(Path(top_source)) == truth_hash,
                "source_state": summary.get("source_label"),
                "source_score": summary.get("source_score"),
                "source_cache": (source.get("metrics") or {}).get("feature_cache"),
            }
        )
        print(
            f"案例{case:>2} 身份 {top_identity}（期望 {expected_identity}，{summary.get('identity_label')}） "
            f"素材 {rows[-1]['source_top']}（命中 {rows[-1]['source_hit']}，{summary.get('source_label')}） {elapsed:.1f} 秒",
            flush=True,
        )

    def counts(key: str) -> dict:
        result: dict[str, int] = {}
        for row in rows:
            label = str(row[key])
            result[label] = result.get(label, 0) + 1
        return result

    identity_ok = [row for row in rows if row["identity_status"] == "success"]
    summary = {
        "cases": len(rows),
        "identity": {
            "completed": len(identity_ok),
            "top1_hits": sum(row["identity_hit"] for row in identity_ok),
            "not_completed": len(rows) - len(identity_ok),
            "states": counts("identity_state"),
        },
        "source": {
            "completed": sum(row["source_status"] == "success" for row in rows),
            "top1_hits": sum(row["source_hit"] for row in rows),
            "states": counts("source_state"),
        },
        "median_seconds": sorted(row["seconds"] for row in rows)[len(rows) // 2],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
