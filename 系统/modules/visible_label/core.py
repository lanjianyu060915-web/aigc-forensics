"""画面坐标还原及固定词表匹配，与模型及判定规则解耦。"""

import json
from pathlib import Path
import re
import unicodedata


def normalize_text(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def load_keywords() -> dict:
    return json.loads(Path(__file__).with_name("keywords.json").read_text(encoding="utf-8"))


def scan_views(image):
    """在最长边一千六百像素的工作图上扫描整图及两倍角落，保存原图映射。"""
    width, height = image.size
    scale = min(1, 1600 / max(width, height))
    full = image.resize((max(1, round(width * scale)), max(1, round(height * scale))))
    sx, sy = full.width / width, full.height / height
    views = [(full, 0, 0, sx, sy)]
    crop_w, crop_h = max(1, round(full.width * .3)), max(1, round(full.height * .3))
    for x, y in ((0, 0), (full.width - crop_w, 0), (0, full.height - crop_h), (full.width - crop_w, full.height - crop_h)):
        crop = full.crop((x, y, x + crop_w, y + crop_h)).resize((crop_w * 2, crop_h * 2))
        views.append((crop, x / sx, y / sy, 2 * sx, 2 * sy))
    return views


def match_text(text: str, in_corner: bool, keywords: dict) -> str | None:
    normalized = normalize_text(text)
    if any(normalize_text(word) in normalized for word in keywords["generic"]):
        return "visible_label_found"
    if in_corner:
        original = unicodedata.normalize("NFKC", text).casefold()
        for word in keywords["tools"]:
            if word.isascii():
                pattern = r"(?<![a-z])" + r"\s*".join(re.escape(char) for char in word.casefold()) + r"(?![a-z])"
                matched = bool(re.search(pattern, original))
            else:
                matched = normalize_text(word) in normalized
            if matched:
                return "tool_hint_found"
    return None


def convert_hit(row, *, width, height, scale_x, scale_y, offset_x=0, offset_y=0,
                frame_index=0, timestamp_seconds=None, keywords=None):
    """缩放和角落裁剪坐标还原至原图，字框中心落在外侧百分之十五视为边缘。"""
    polygon, text, confidence = row
    points = [[round(min(width, max(0, x / scale_x + offset_x)), 2),
               round(min(height, max(0, y / scale_y + offset_y)), 2)] for x, y in polygon]
    bbox = [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]
    center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    in_corner = center_x <= width * .15 or center_x >= width * .85 or center_y <= height * .15 or center_y >= height * .85
    kind = match_text(str(text), in_corner, keywords or load_keywords())
    if not kind:
        return None
    return {"text": str(text)[:200], "confidence": round(float(confidence), 6),
            "bbox": bbox, "polygon": points, "in_corner": in_corner, "kind": kind,
            "frame_index": frame_index, "timestamp_seconds": timestamp_seconds}


def restore_exif_coordinates(hit: dict, width: int, height: int, orientation: int) -> dict:
    """识别前转正画面，输出仍还原为文件原始像素坐标。"""
    transforms = {
        2: lambda x, y: (width - x, y),
        3: lambda x, y: (width - x, height - y),
        4: lambda x, y: (x, height - y),
        5: lambda x, y: (y, x),
        6: lambda x, y: (y, height - x),
        7: lambda x, y: (width - y, height - x),
        8: lambda x, y: (width - y, x),
    }
    if orientation not in transforms:
        return hit
    result = dict(hit)
    points = [transforms[orientation](*point) for point in hit["polygon"]]
    result["polygon"] = [[round(x, 2), round(y, 2)] for x, y in points]
    result["bbox"] = [min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)]
    return result


def deduplicate(hits: list[dict]) -> list[dict]:
    result = []
    for hit in sorted(hits, key=lambda item: -item["confidence"]):
        duplicate = False
        for previous in result:
            if hit["frame_index"] != previous["frame_index"] or normalize_text(hit["text"]) != normalize_text(previous["text"]):
                continue
            a, b = hit["bbox"], previous["bbox"]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if smaller and intersection / smaller >= .5:
                duplicate = True
                break
        if not duplicate:
            result.append(hit)
    return sorted(result, key=lambda item: (item["frame_index"], item["bbox"][1], item["bbox"][0]))
