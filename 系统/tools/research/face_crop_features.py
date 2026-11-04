"""给人脸裁剪图提 CLIP 特征和 CommunityForensics 分数，供人脸分支训练和评测。

用引擎环境运行。输入是 extract_face_crops.py 产出的目录，输出 features.npz，
每个样本一行：多张裁剪的 CLIP 特征取均值，CF 分数取均值。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine"))

from aigc_engine.scorer import GeneralAigcScorer, normalize_rgb  # noqa: E402


def _load(path: Path, size: int, resample: Image.Resampling) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size, size), resample=resample), dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    package = ROOT / "models" / "video-c38-refit3-cf384"
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    scorer = GeneralAigcScorer(package / manifest["checkpoint"], sha256=manifest["checkpoint_sha256"])
    model = scorer.model
    rows: list[dict] = []
    for root in args.crops:
        with (root / "crops.jsonl").open(encoding="utf-8") as stream:
            rows.extend({**json.loads(line), "_root": str(root)} for line in stream if line.strip())
    index = [(i, Path(row["_root"]) / crop) for i, row in enumerate(rows) for crop in row["crops"]]
    print(f"样本 {len(rows)}，裁剪图 {len(index)}", flush=True)
    embeds: list[np.ndarray] = []
    community: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(index), args.batch):
            chunk = index[start : start + args.batch]
            clip_rgb = np.stack([_load(path, 224, Image.Resampling.BICUBIC) for _, path in chunk])
            cf_rgb = np.stack([_load(path, 384, Image.Resampling.BILINEAR) for _, path in chunk])
            clip_tensor = normalize_rgb(clip_rgb, model.clip.preprocessing).to(scorer.device)
            cf_tensor = normalize_rgb(cf_rgb, model.community.preprocessing).to(scorer.device)
            embeds.append(model.clip.backbone(pixel_values=clip_tensor).image_embeds.float().cpu().numpy())
            community.append(model.community.backbone(cf_tensor)[:, -1].float().cpu().numpy())
            if (start // args.batch) % 20 == 0:
                print(f"{start}/{len(index)}", flush=True)
    all_embeds = np.concatenate(embeds) if embeds else np.zeros((0, 768), dtype=np.float32)
    all_community = np.concatenate(community) if community else np.zeros((0,), dtype=np.float32)
    owners = np.array([owner for owner, _ in index], dtype=np.int64)
    features = np.full((len(rows), all_embeds.shape[1]), np.nan, dtype=np.float32)
    cf_scores = np.full(len(rows), np.nan, dtype=np.float32)
    for i in range(len(rows)):
        mask = owners == i
        if mask.any():
            features[i] = all_embeds[mask].mean(axis=0)
            cf_scores[i] = all_community[mask].mean()
    np.savez_compressed(
        args.out,
        sample_id=np.array([row["sample_id"] for row in rows]),
        label=np.array([row["label"] for row in rows], dtype=np.int8),
        source=np.array([str(row.get("source")) for row in rows]),
        split=np.array([str(row.get("split")) for row in rows]),
        group_id=np.array([str(row.get("group_id")) for row in rows]),
        crop_count=np.array([len(row["crops"]) for row in rows], dtype=np.int32),
        embeds=features,
        community=cf_scores,
    )
    print("完成", args.out, features.shape, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
