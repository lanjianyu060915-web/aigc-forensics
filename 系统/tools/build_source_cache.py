# -*- coding: utf-8 -*-
"""预先建好素材库的特征缓存（全局特征 + SIFT），之后每次检索只需几秒。

用法：python -B tools/build_source_cache.py --library <素材库> --cache-dir <缓存目录>
不改变任何检索分数；已缓存且未变化的文件直接跳过。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules" / "source_search"))

from artifact_io import atomic_write_json

from source_search.face_mask import FaceMasker  # noqa: E402
from source_search.feature_cache import SourceFeatureCache  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--progress-path", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    def progress(stage, *, current=None, total=None, status="running", error=None):
        if args.progress_path is not None:
            atomic_write_json(args.progress_path, {"schema_version": "4.0", "kind": "source_cache_build",
                              "stage": stage, "status": status, "current": current, "total": total,
                              "elapsed_seconds": time.perf_counter() - started, "error": error}, role="source_cache_progress")
    progress("checking_library")
    cache = SourceFeatureCache(args.cache_dir, args.library)
    try:
        progress("loading_face_mask")
        masker = FaceMasker()
        entries, skipped = cache.load_or_build(masker, progress_callback=lambda current, total, path, ok:
                                              progress("global_features", current=current, total=total))
        sift = cv2.SIFT_create(nfeatures=2500)
        for index, entry in enumerate(entries, 1):
            progress("local_features", current=index - 1, total=len(entries))
            cache.sift_features(entry, sift)
        progress("ready", current=len(entries), total=len(entries), status="success")
    except Exception as exc:
        progress("failed", status="error", error=type(exc).__name__)
        raise
    print(
        json.dumps(
            {
                "status": "success",
                "image_count": len(entries),
                "skipped_count": len(skipped),
                "runtime_seconds": round(time.perf_counter() - started, 2),
                **cache.stats.as_dict(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
