# -*- coding: utf-8 -*-
"""把各次评测的逐样本分数汇总成 文档/评测/ 下的 JSON，报告里的数字都从这里来。

输入在仓库外的 评测集/ 目录（数据不随仓库分发）。用引擎环境或任何带 numpy 的环境运行：
    python tools/research/build_eval_summary.py
缺哪部分输入就跳过哪部分，不会用旧数字顶替。
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from aigc_engine.metrics import evaluate_binary  # noqa: E402
from decision.aigc_overall import decide_overall  # noqa: E402

DATA = ROOT.parent / "评测集"
OUT = ROOT.parent / "文档" / "评测"
CALIBRATION = json.loads((ROOT / "models/video-c38-refit3-cf384/calibration.json").read_text(encoding="utf-8"))
WHOLE_IMAGE = json.loads(
    (ROOT / "packaging_v0.4/resources/detectors_v2/communityforensics_vit/calibration.json").read_text(encoding="utf-8")
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def band(value: float | None, thresholds: dict[str, Any]) -> str | None:
    if value is None or not np.isfinite(value):
        return None
    if value >= thresholds["medium_max"]:
        return "high"
    return "medium" if value >= thresholds["low_max"] else "low"


def auc(labels: Iterable[int], scores: Iterable[float | None]) -> float | None:
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if s is not None and np.isfinite(s)]
    if len({y for y, _ in pairs}) < 2:
        return None
    result = evaluate_binary([y for y, _ in pairs], [s for _, s in pairs])
    return round(float(result.roc_auc), 4)


def eer(labels: Iterable[int], scores: Iterable[float | None]) -> float | None:
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if s is not None and np.isfinite(s)]
    if len({y for y, _ in pairs}) < 2:
        return None
    return round(float(evaluate_binary([y for y, _ in pairs], [s for _, s in pairs]).eer) * 100, 2)


def rates(bands: list[str | None]) -> dict[str, Any]:
    """档位分布；分母是全部样本，没有档位的（比如没有人脸）算在 none 里。"""
    count = Counter(bands)
    total = len(bands)
    percent = lambda n: round(100 * n / total, 1) if total else None
    return {
        "n": total,
        "low": percent(count["low"]),
        "medium": percent(count["medium"]),
        "high": percent(count["high"]),
        "none": percent(count[None]),
        "review_or_above": percent(count["medium"] + count["high"]),
    }


def face_scores(path: Path) -> dict[str, float | None]:
    raw = np.load(path, allow_pickle=False)
    return {
        str(sample): (float(score) if int(crops) > 0 else None)
        for sample, score, crops in zip(raw["sample_id"], raw["community"], raw["crop_count"])
    }


def summarize_group(rows: list[dict[str, Any]], general_key: str, thresholds: dict[str, Any]) -> dict[str, Any]:
    """rows 里每行要有 label、general_key、face（可为 None）、whole_band（可为 None）。"""
    labels = [int(row["label"]) for row in rows]
    output: dict[str, Any] = {
        "n": len(rows),
        "real": labels.count(0),
        "fake": labels.count(1),
        "general_auc": auc(labels, [row.get(general_key) for row in rows]),
        "general_eer": eer(labels, [row.get(general_key) for row in rows]),
        "community_whole_auc": auc(labels, [row.get("community_logit") for row in rows]),
    }
    with_face = [row for row in rows if row.get("face") is not None]
    if with_face:
        face_labels = [int(row["label"]) for row in with_face]
        output["faces"] = {"real": face_labels.count(0), "fake": face_labels.count(1)}
        output["face_auc"] = auc(face_labels, [row["face"] for row in with_face])
        output["general_auc_on_face_samples"] = auc(face_labels, [row.get(general_key) for row in with_face])
    without_face = [row for row in rows if row.get("face") is None]
    if with_face and without_face:
        output["general_auc_without_face"] = auc(
            [int(row["label"]) for row in without_face], [row.get(general_key) for row in without_face]
        )
    for name, value in (("real", 0), ("fake", 1)):
        subset = [row for row in rows if int(row["label"]) == value]
        if not subset:
            continue
        general_bands = [band(row.get(general_key), thresholds) for row in subset]
        face_bands = [band(row.get("face"), CALIBRATION["face_crop"]) for row in subset]
        overall = [
            decide_overall(general, face, row.get("whole_band"))[0]
            for general, face, row in zip(general_bands, face_bands, subset)
        ]
        output[name] = {"general": rates(general_bands), "face": rates(face_bands), "overall": rates(overall)}
        if any(row.get("whole_band") for row in subset):
            output[name]["whole_image"] = rates([row.get("whole_band") for row in subset])
    return output


def whole_image_band(community_logit: float | None) -> str | None:
    """整图分支和引擎的 CF 分数相同（差 0.0），可以直接用它的概率边界分档。"""
    if community_logit is None:
        return None
    probability = 1.0 / (1.0 + np.exp(-float(community_logit)))
    return band(probability, WHOLE_IMAGE["bands"])


def write(relative: str, payload: Any) -> None:
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("写入", target.relative_to(ROOT.parent))


def video_panels() -> None:
    panels = {}
    rows_c21 = face_scores(DATA / "face_branch" / "features_c21.npz") if (DATA / "face_branch/features_c21.npz").is_file() else {}
    for name in ("platform148", "first199", "genvid400", "vif310", "fresh"):
        path = DATA / "panels" / f"{name}.scores.jsonl"
        if not path.is_file():
            continue
        rows = _jsonl(path)
        for row in rows:
            row["face"] = rows_c21.get(row["sample_id"])
        panels[name] = summarize_group(rows, "logit", CALIBRATION["video"])
    write(
        "视频集合-20261104/summary.json",
        {
            "note": "这些集合选模时看过，不是盲测。边界也来自这些集合的真实样本。人脸分支只在 first199 上有裁剪。",
            "thresholds": {k: CALIBRATION["video"][k] for k in ("low_max", "medium_max")},
            "panels": panels,
        },
    )


def face_swap_sets() -> None:
    features = DATA / "face_branch" / "features.npz"
    scores = DATA / "cross" / "cross_all.scores.jsonl"
    if not (features.is_file() and scores.is_file()):
        return
    faces = face_scores(features)
    whole_videos = {}
    full = DATA / "cross" / "face_videos.fullimage.jsonl"
    if full.is_file():
        whole_videos = {row["sample_id"]: row.get("band") for row in _jsonl(full)}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in _jsonl(scores):
        if row["sample_id"] in seen:
            continue
        seen.add(row["sample_id"])
        row["face"] = faces.get(row["sample_id"])
        is_image = row["source"] in {"swan_df_images", "case3_pairs", "data_formal_test_crops"}
        if is_image:
            # 这些分数是改公式前跑的，三路原始分都在，按现行图片公式重算。
            row["logit"] = 0.5 * float(row["clip_refit_logit"]) + 0.5 * float(row["community_logit"])
            row["whole_band"] = whole_image_band(row.get("community_logit"))
        else:
            row["whole_band"] = whole_videos.get(row["sample_id"])
        groups[row["source"]].append(row)
    output = {}
    for source, rows in groups.items():
        is_image = source in {"swan_df_images", "case3_pairs", "data_formal_test_crops"}
        if is_image and "image" not in CALIBRATION:
            continue
        output[source] = summarize_group(rows, "logit", CALIBRATION["image" if is_image else "video"])
    write(
        "人脸分支-20261104/summary.json",
        {
            "note": "换脸数据都是选模和定边界时看过的，不是盲测。案例 3 没有参与人脸边界标定。FF 裁剪只有 112 像素，人脸分支不适用。",
            "face_thresholds": {k: CALIBRATION["face_crop"][k] for k in ("low_max", "medium_max")},
            "sets": output,
        },
    )


def image_splits() -> None:
    folder = DATA / "image"
    manifest_rows = _jsonl(folder / "calibration.jsonl") if (folder / "calibration.jsonl").is_file() else []
    if not manifest_rows or "image" not in CALIBRATION:
        return
    report: dict[str, Any] = {
        "thresholds": {k: CALIBRATION["image"][k] for k in ("low_max", "medium_max")},
    }
    slim: list[dict[str, Any]] = []
    for split, note in (
        ("calibration", "定公式和边界用的划分"),
        ("test", "公式和边界冻结后只跑一次"),
        ("test_degraded", "test 母图的传播退化版，同样只跑一次"),
    ):
        scores_path = folder / f"{split}.scores.jsonl"
        if not scores_path.is_file():
            continue
        rows = _jsonl(scores_path)
        features_path = folder / f"features{'_cal' if split == 'calibration' else '_' + split}.npz"
        faces = face_scores(features_path) if features_path.is_file() else {}
        for row in rows:
            row["face"] = faces.get(row["sample_id"])
            row["whole_band"] = whole_image_band(row.get("community_logit"))
            slim.append(
                {
                    key: row.get(key)
                    for key in (
                        "sample_id", "label", "source", "generator", "split", "transform", "group_id", "logit",
                        "clip_refit_logit", "clip_original_logit", "community_logit", "face",
                    )
                }
            )
        section: dict[str, Any] = {"note": note, "all": summarize_group(rows, "logit", CALIBRATION["image"])}
        real_rows = [row for row in rows if int(row["label"]) == 0]
        by_generator = {}
        for generator in sorted({row["generator"] for row in rows if int(row["label"]) == 1}):
            subset = [row for row in rows if row["generator"] == generator]
            bands = [band(row["logit"], CALIBRATION["image"]) for row in subset]
            by_generator[generator] = {
                "n": len(subset),
                "auc_against_all_real": auc(
                    [0] * len(real_rows) + [1] * len(subset),
                    [row["logit"] for row in real_rows] + [row["logit"] for row in subset],
                ),
                "general": rates(bands),
            }
        section["by_generator"] = by_generator
        by_real = {}
        for source in sorted({row["source"] for row in real_rows}):
            subset = [row for row in real_rows if row["source"] == source]
            by_real[source] = {"n": len(subset), "general": rates([band(row["logit"], CALIBRATION["image"]) for row in subset])}
        section["by_real_source"] = by_real
        transforms = sorted({row.get("transform") for row in rows})
        if len(transforms) > 1:
            section["by_transform"] = {
                str(transform): summarize_group(
                    [row for row in rows if row.get("transform") == transform], "logit", CALIBRATION["image"]
                )
                for transform in transforms
            }
        report[split] = section
    write("图片评测-20261104/summary.json", report)
    target = OUT / "图片评测-20261104" / "scores.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in slim), encoding="utf-8", newline="\n")


def source_search() -> None:
    path = DATA / "source_eval" / "results.reduced.jsonl"
    if not path.is_file():
        return
    rows = _jsonl(path)
    output: dict[str, Any] = {}
    for role in ("positive", "negative"):
        subset = [row for row in rows if row["role"] == role]
        scores = np.array([row["global_score"] for row in subset], dtype=float)
        output[role] = {
            "n": len(subset),
            "states": dict(Counter(row["state"] for row in subset)),
            "global_score_quantiles": {
                q: round(float(np.quantile(scores, float(q))), 4) for q in ("0.01", "0.05", "0.5", "0.95", "0.99")
            }
            if len(scores)
            else {},
            "by_kind": {
                kind: dict(Counter(row["state"] for row in subset if row["kind"] == kind))
                for kind in sorted({row["kind"] for row in subset})
            },
        }
        if role == "positive":
            output[role]["top1_hit"] = sum(row["top1"] == row["expect"] for row in subset)
    output["seconds_median"] = float(np.median([row["seconds"] for row in rows])) if rows else None
    write("素材检索-20261104/summary.json", output)


def main() -> int:
    video_panels()
    face_swap_sets()
    image_splits()
    source_search()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
