"""人脸分支可行性试验：在人脸裁剪的 CLIP 特征上训练线性头，看跨数据集表现。

训练集和测试集分开，测试集只看一次；CF 分数作对照。
    python -B tools/research/face_branch_probe.py --features features.npz --out report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))

from aigc_engine.metrics import evaluate_binary  # noqa: E402


def _per_video_ff(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """FF 子集每条视频 5 帧，按视频取均值，避免同一视频的帧当成独立样本。"""
    keep = data["source"].astype(str)
    out: dict[str, list] = {key: [] for key in data}
    groups: dict[tuple[str, str], list[int]] = {}
    for index, (source, group) in enumerate(zip(keep, data["group_id"].astype(str), strict=True)):
        key = (source, group) if source.startswith("ff_") else (source, str(data["sample_id"][index]))
        groups.setdefault(key, []).append(index)
    for indices in groups.values():
        first = indices[0]
        for key, value in data.items():
            if key == "embeds":
                out[key].append(np.nanmean(value[indices], axis=0))
            elif key == "community":
                out[key].append(np.nanmean(value[indices]))
            else:
                out[key].append(value[first])
    return {key: np.asarray(value) for key, value in out.items()}


def report(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int | None]:
    mask = np.isfinite(scores)
    y = labels[mask].astype(int).tolist()
    s = scores[mask].astype(float).tolist()
    if len(set(y)) < 2:
        return {"n": len(y), "eer": None, "auc": None}
    metrics = evaluate_binary(y, s)
    return {
        "n": len(y),
        "real": y.count(0),
        "fake": y.count(1),
        "eer": round(metrics.eer * 100, 2),
        "auc": round(metrics.roc_auc, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    raw = np.load(args.features, allow_pickle=False)
    data = _per_video_ff({key: raw[key] for key in raw.files})
    source = data["source"].astype(str)
    split = data["split"].astype(str)
    usable = np.isfinite(data["embeds"]).all(axis=1)
    print("无人脸样本:", {name: int(((source == name) & ~usable).sum()) for name in sorted(set(source))})
    embeds = data["embeds"] / np.clip(np.linalg.norm(data["embeds"], axis=1, keepdims=True), 1e-6, None)
    labels = data["label"].astype(int)
    train_sets = {
        "只用 FF": source == "ff_train",
        "FF + DFDC 校准集": (source == "ff_train") | ((source == "dfdc_240") & (split == "calibration")),
    }
    tests = {
        "FF 验证": source == "ff_val",
        "FF 测试": source == "ff_test",
        "DFDC 校准集": (source == "dfdc_240") & (split == "calibration"),
        "DFDC 留出集": (source == "dfdc_240") & (split == "holdout"),
        "SWAN-DF 图片": source == "swan_df_images",
        "SWAN-DF 视频": source == "swan_df_videos",
        "案例 3": source == "case3_pairs",
    }
    result: dict[str, dict] = {"CF 对照": {}}
    for test_name, mask in tests.items():
        result["CF 对照"][test_name] = report(labels[mask], data["community"][mask])
    for train_name, train_mask in train_sets.items():
        mask = train_mask & usable
        scaler = StandardScaler().fit(embeds[mask])
        for c in (0.01, 0.1, 1.0):
            model = LogisticRegression(max_iter=3000, C=c, class_weight="balanced")
            model.fit(scaler.transform(embeds[mask]), labels[mask])
            scores = np.full(len(labels), np.nan)
            scores[usable] = model.decision_function(scaler.transform(embeds[usable]))
            name = f"{train_name} C={c}"
            result[name] = {test_name: report(labels[m], scores[m]) for test_name, m in tests.items()}
    for name, values in result.items():
        print(f"\n{name}")
        for test_name, value in values.items():
            print(f"  {test_name:12s} n={value['n']:4d} EER={value['eer']} AUC={value['auc']}")
    if args.out is not None:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
