# -*- coding: utf-8 -*-
"""预先建好底图库的特征缓存（全局特征 + SIFT），之后每次检索只需几秒。

用法：python -B tools/build_source_cache.py --library <底图库> --cache-dir <缓存目录>
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
sys.path.insert(0, str(ROOT / "modules" / "source_search"))

from source_search.face_mask import FaceMasker  # noqa: E402
from source_search.feature_cache import SourceFeatureCache  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    cache = SourceFeatureCache(args.cache_dir, args.library)
    entries, skipped = cache.load_or_build(FaceMasker())
    sift = cv2.SIFT_create(nfeatures=2500)
    for entry in entries:
        cache.sift_features(entry, sift)
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
