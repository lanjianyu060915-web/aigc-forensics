"""从已冻结评测汇总生成总体档位的证据强度统计。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = (
    ("image", "图片", "图片评测-20261104", "test"),
    ("image_degraded", "图片传播退化", "图片评测-20261104", "test_degraded"),
    ("video", "视频", "视频盲测-20261104", "test"),
    ("video_degraded", "视频统一转码", "视频盲测-20261104", "test_degraded"),
    ("face", "换脸", "换脸盲测-20261104", "raw"),
)


def band_statistics(real: dict, fake: dict, band: str) -> dict[str, Any]:
    """按汇总百分比恢复计数，再计算两侧落入该档的比例之比。"""
    real_total, fake_total = int(real["n"]), int(fake["n"])
    if real_total <= 0 or fake_total <= 0:
        raise ValueError("两侧总样本数必须大于零")
    real_count = round(real_total * float(real[band]) / 100)
    fake_count = round(fake_total * float(fake[band]) / 100)
    real_ratio, fake_ratio = real_count / real_total, fake_count / fake_total
    return {
        "real_count": real_count, "real_total": real_total, "real_ratio": real_ratio,
        "fake_count": fake_count, "fake_total": fake_total, "fake_ratio": fake_ratio,
        "ratio": fake_ratio / real_ratio if real_count else None,
        "small_sample": real_count < 10 or fake_count < 10,
    }


def build(root: Path = ROOT) -> dict[str, Any]:
    scenarios = []
    for key, label, directory, split in SCENARIOS:
        path = root / "文档" / "评测" / directory / "summary.json"
        raw = path.read_bytes()
        group = json.loads(raw)[split]["all"]
        scenarios.append({
            "key": key, "label": label,
            "source_file": path.relative_to(root).as_posix(),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "bands": {band: band_statistics(group["real"]["overall"], group["fake"]["overall"], band) for band in ("low", "medium", "high")},
        })
    return {"schema_version": "1.0", "scenarios": scenarios}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "系统" / "decision" / "band_evidence.json"
    data = build(args.root)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for scenario in data["scenarios"]:
        print(scenario["label"], {band: round(values["ratio"], 3) if values["ratio"] is not None else None for band, values in scenario["bands"].items()})


if __name__ == "__main__":
    main()
