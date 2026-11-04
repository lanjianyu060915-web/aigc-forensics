# -*- coding: utf-8 -*-
"""重算整图检测器的分带边界，写回 calibration.json。

和 M3 同样的口径：SWAN-DF 案例 01-10 的真人图分数取 p95、p99 作为边界，案例 11-20 留作验证。
必须用鉴伪运行环境执行，保证和线上内核一致：
  packaging_v0.4\\resources\\portable_python_v2\\python.exe -E -S -B tools\\recalibrate_detector_v2.py --swan <SWAN-DF 目录>
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "deepfake_detection_v2"))

from vit_detector import CommunityForensicsDetector  # noqa: E402


CALIBRATION = ROOT / "packaging_v0.4" / "resources" / "detectors_v2" / "communityforensics_vit" / "calibration.json"


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    wins = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((wins + 0.5 * ties) / (positive.size * negative.size))


def bootstrap_interval(labels: np.ndarray, scores: np.ndarray, draws: int = 2000, seed: int = 20261104) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        index = rng.integers(0, len(labels), len(labels))
        if 0 < labels[index].sum() < len(labels):
            values.append(roc_auc(labels[index], scores[index]))
    return [round(float(item), 4) for item in np.quantile(values, [0.025, 0.975])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swan", required=True, type=Path)
    args = parser.parse_args()

    resources = CALIBRATION.parent
    detector = CommunityForensicsDetector(resources, CALIBRATION)
    rows = []
    for case_dir in sorted(args.swan.glob("案例*")):
        for image in sorted(case_dir.glob("*.jpg")):
            rows.append(
                (
                    int(case_dir.name.replace("案例", "")),
                    1 if image.name.startswith("05") else 0,
                    float(detector.predict(image)["score"]),
                )
            )
    case = np.array([row[0] for row in rows])
    labels = np.array([row[1] for row in rows])
    scores = np.array([row[2] for row in rows])
    calibration_set = case <= 10
    real = scores[calibration_set & (labels == 0)]
    percentiles = {
        name: float(np.quantile(real, q))
        for name, q in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99), ("p100", 1.0))
    }
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    payload["created_at"] = date.today().isoformat()
    payload["bands"]["low_max"] = percentiles["p95"]
    payload["bands"]["medium_max"] = percentiles["p99"]
    payload["reference_real_percentiles"] = percentiles
    payload["calibration_set"]["auc"] = round(roc_auc(labels[calibration_set], scores[calibration_set]), 4)
    holdout = ~calibration_set
    target = percentiles["p95"]
    payload["validation_set"]["auc"] = round(roc_auc(labels[holdout], scores[holdout]), 4)
    payload["validation_set"]["auc_ci95"] = bootstrap_interval(labels[holdout], scores[holdout])
    payload["validation_set"]["operating_point_at_target_fpr_5pct"] = {
        "threshold": target,
        "recall": float((scores[holdout & (labels == 1)] > target).mean()),
        "fpr": float((scores[holdout & (labels == 0)] > target).mean()),
    }
    payload["all_100_auc"] = round(roc_auc(labels, scores), 4)
    CALIBRATION.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("bands", "calibration_set", "validation_set", "all_100_auc")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
