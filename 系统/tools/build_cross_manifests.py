"""生成交叉评测清单：用本机已有的换脸数据考通用 AIGC 分支。

数据都不进仓库，路径用参数传入。清单格式见 tools/eval_general_aigc.py。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    fake = sum(row["label"] for row in rows)
    print(f"{path.name}: {len(rows)} 条，真实 {len(rows) - fake}，伪造 {fake}")


def _first_image(folder: Path) -> Path:
    return next(item for item in sorted(folder.iterdir()) if item.suffix.lower() in IMAGE_SUFFIXES)


def swan_images(root: Path) -> list[dict]:
    rows = []
    for case_dir in sorted(root.glob("案例*")):
        if not case_dir.is_dir():
            continue
        for image in sorted(case_dir.glob("*.jpg")):
            case = int(case_dir.name.replace("案例", ""))
            rows.append(
                {
                    "sample_id": f"swan-{case:02d}-{image.stem}",
                    "path": str(image),
                    "label": 1 if image.name.startswith("05") else 0,
                    "source": "swan_df_images",
                    "split": "case01_10" if case <= 10 else "case11_20",
                    "case": case,
                }
            )
    return rows


def case3_images(root: Path) -> list[dict]:
    rows = []
    for case in range(1, 21):
        folder = root / str(case)
        for kind, label in (("AI图片", 1), ("原始图片", 0)):
            rows.append(
                {
                    "sample_id": f"case3-{case:02d}-{'ai' if label else 'orig'}",
                    "path": str(_first_image(folder / kind)),
                    "label": label,
                    "source": "case3_pairs",
                    "case": case,
                }
            )
    return rows


def data_formal_images(root: Path) -> list[dict]:
    rows = []
    for label_name, label in (("real", 0), ("fake", 1)):
        for image in sorted((root / "images" / "test" / label_name).iterdir()):
            if image.suffix.lower() in IMAGE_SUFFIXES:
                rows.append(
                    {
                        "sample_id": f"formal-{label_name}-{image.stem}",
                        "path": str(image),
                        "label": label,
                        "source": "data_formal_test_crops",
                    }
                )
    return rows


def dfdc_videos(manifest: Path) -> list[dict]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return [
        {
            "sample_id": f"dfdc-{item['split']}-{item['file_name']}",
            "path": item["path"],
            "label": 1 if item["label"] == "FAKE" else 0,
            "source": "dfdc_240",
            "split": item["split"],
        }
        for item in payload["videos"]
    ]


def swan_videos(manifest: Path) -> list[dict]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = []
    for item in payload["videos"]:
        if item["role"] == "identity_source":
            continue
        rows.append(
            {
                "sample_id": f"swanvideo-{item['pair']}-{item['file_name']}",
                "path": item["path"],
                "label": 1 if item["role"] == "deepfake_result" else 0,
                "source": "swan_df_videos",
                "pair": item["pair"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="生成交叉评测清单")
    parser.add_argument("--swan-images", type=Path, required=True)
    parser.add_argument("--case3", type=Path, required=True)
    parser.add_argument("--data-formal", type=Path, required=True)
    parser.add_argument("--dfdc-manifest", type=Path, required=True)
    parser.add_argument("--swan-video-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _write(args.out / "swan_images.jsonl", swan_images(args.swan_images))
    _write(args.out / "case3_images.jsonl", case3_images(args.case3))
    _write(args.out / "data_formal_crops.jsonl", data_formal_images(args.data_formal))
    _write(args.out / "dfdc_videos.jsonl", dfdc_videos(args.dfdc_manifest))
    _write(args.out / "swan_videos.jsonl", swan_videos(args.swan_video_manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
