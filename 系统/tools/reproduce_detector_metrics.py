# -*- coding: utf-8 -*-
"""在当前系统上重算二代检测器的公开指标，和 9 月记录的数字对照。

必须用鉴伪运行环境执行，保证和线上内核一致：
  packaging_v0.4\\resources\\portable_python_v2\\python.exe -E -S -B tools\\reproduce_detector_metrics.py
      --swan <SWAN-DF 案例目录> --case3 <案例3 目录> --out <输出 JSON>

SWAN-DF：文件名以 05 开头的是换脸图，其余是真人图。案例01-10 定标，案例11-20 留出。
案例3：每个案例一张 AI 图片、一张原始图片，按配对比较。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "deepfake_detection_v2"))

from vit_detector import CommunityForensicsDetector  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# 线上用 ONNX 模型。9 月的 PyTorch 参考值是 0.8978，ONNX 验收值是 0.8969（Pearson 0.9994）。
RECORDED = {
    "swan_all_100": 0.8969,
    "swan_case01_10": 0.8425,
    "swan_case11_20": 0.9925,
    "case3_pair": 0.6625,
}


def roc_auc(labels: list[int], scores: list[float]) -> float:
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    positive = s[y == 1]
    negative = s[y == 0]
    wins = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((wins + 0.5 * ties) / (positive.size * negative.size))


def predict(detector: CommunityForensicsDetector, path: Path) -> dict:
    result = detector.predict(path)
    return {"score": float(result["score"]), "band": str(result["band"])}


def band_counts(rows: list[dict]) -> dict:
    counts = {"low": 0, "medium": 0, "high": 0}
    for row in rows:
        counts[row["band"]] = counts.get(row["band"], 0) + 1
    return counts


def first_image(folder: Path) -> Path:
    return next(path for path in sorted(folder.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swan", required=True, type=Path)
    parser.add_argument("--case3", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    resources = ROOT / "packaging_v0.4" / "resources" / "detectors_v2" / "communityforensics_vit"
    detector = CommunityForensicsDetector(resources, resources / "calibration.json")
    started = time.perf_counter()

    swan_rows = []
    for case_dir in sorted(args.swan.glob("案例*")):
        for image in sorted(case_dir.glob("*.jpg")):
            swan_rows.append(
                {
                    "case": int(case_dir.name.replace("案例", "")),
                    "file": image.name,
                    "fake": 1 if image.name.startswith("05") else 0,
                    **predict(detector, image),
                }
            )

    def subset(low: int, high: int) -> list[dict]:
        return [row for row in swan_rows if low <= row["case"] <= high]

    def swan_summary(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "n_fake": sum(row["fake"] for row in rows),
            "auc": round(roc_auc([r["fake"] for r in rows], [r["score"] for r in rows]), 4),
            "fake_bands": band_counts([r for r in rows if r["fake"]]),
            "real_bands": band_counts([r for r in rows if not r["fake"]]),
        }

    case3_rows = []
    for case in range(1, 21):
        folder = args.case3 / str(case)
        ai = predict(detector, first_image(folder / "AI图片"))
        original = predict(detector, first_image(folder / "原始图片"))
        case3_rows.append({"case": case, "ai": ai, "original": original})
    labels = [1] * 20 + [0] * 20
    scores = [r["ai"]["score"] for r in case3_rows] + [r["original"]["score"] for r in case3_rows]

    elapsed = time.perf_counter() - started
    total_images = len(swan_rows) + 2 * len(case3_rows)
    summary = {
        "swan_all_100": swan_summary(swan_rows),
        "swan_case01_10": swan_summary(subset(1, 10)),
        "swan_case11_20": swan_summary(subset(11, 20)),
        "case3_pair": {
            "n_pairs": 20,
            "auc": round(roc_auc(labels, scores), 4),
            "ai_higher": sum(r["ai"]["score"] > r["original"]["score"] for r in case3_rows),
            "ai_bands": band_counts([r["ai"] for r in case3_rows]),
            "original_bands": band_counts([r["original"] for r in case3_rows]),
        },
        "seconds_per_image": round(elapsed / total_images, 3),
        "images": total_images,
    }
    comparison = {
        key: {"recorded": value, "now": summary[key]["auc"], "same": abs(value - summary[key]["auc"]) < 1e-4}
        for key, value in RECORDED.items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"summary": summary, "comparison": comparison, "swan_rows": swan_rows, "case3_rows": case3_rows},
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "comparison": comparison}, ensure_ascii=False, indent=1))
    return 0 if all(item["same"] for item in comparison.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
