"""收集任务内图件、生成黑白说明图和上传副本，保留原始证据文件。"""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .evidence_package import mapping, rows, finite


def local_file(root: Path, value: Any) -> Path | None:
    """只接受任务目录内普通文件，阻止链接和目录穿越读取。"""
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else root / candidate
    try:
        relative = candidate.absolute().relative_to(root.absolute())
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            if part == ".." or cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
                return None
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
        return resolved if resolved.is_file() else None
    except (OSError, ValueError):
        return None


def _output_dir(root: Path, name: str) -> Path:
    path = root / name
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        raise ValueError("图件输出目录不能是链接")
    path.mkdir(parents=True, exist_ok=True)
    path.resolve().relative_to(root.resolve())
    return path


def _font(size: int):
    candidates = [Path(os.environ.get("WINDIR", "")) / "Fonts" / "simsun.ttc",
                  Path(__file__).resolve().parents[1] / "assets" / "fonts" / "NotoSansSC-Regular.ttf"]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                pass
    return ImageFont.load_default()


def _save_image(image: Image.Image, path: Path) -> None:
    if path.is_symlink():
        raise ValueError("派生图件不能覆盖链接")
    image.save(path)


def _score_chart(package: dict, output: Path) -> Path:
    entries = [v for v in rows(package.get("items")) if v.get("item_id") in {"E03", "E04", "E05"}]
    image = Image.new("RGB", (1400, 180 + 180 * len(entries)), "white")
    draw = ImageDraw.Draw(image)
    font, small = _font(30), _font(24)
    draw.text((40, 28), "各检测分支分数位置（各行独立刻度）", font=font, fill="black")
    for index, entry in enumerate(entries):
        y = 145 + index * 180
        data = mapping(entry.get("data"))
        score = finite(data.get("score"))
        thresholds = mapping(data.get("thresholds"))
        low, high = finite(thresholds.get("low_max")), finite(thresholds.get("medium_max"))
        draw.text((40, y - 35), entry.get("title", "分支"), font=small, fill="black")
        if score is None or low is None or high is None:
            draw.text((400, y - 35), "分数或阈值未记录，无法绘制位置", font=small, fill="black")
            continue
        minimum, maximum = min(score, low, high), max(score, low, high)
        span = maximum - minimum or max(abs(maximum) * 0.1, 0.1)
        start, end = minimum - span * 0.15, maximum + span * 0.15
        x = lambda value: 400 + int((value - start) / (end - start) * 920)
        draw.line((400, y + 25, 1320, y + 25), fill="black", width=3)
        for label_index, (limit, label) in enumerate(((low, "95%线"), (high, "99%线"))):
            pos = x(limit)
            for offset in range(-10, 55, 10):
                draw.line((pos, y + offset, pos, y + offset + 5), fill="black", width=2)
            label_text = f"{label} {limit:.8g}"
            bounds = draw.textbbox((0, 0), label_text, font=small)
            label_x = min(pos - 45, image.width - 30 - (bounds[2] - bounds[0]))
            draw.text((label_x, y + 55 + label_index * 30), label_text, font=small, fill="black")
        pos = x(score)
        draw.ellipse((pos - 8, y + 17, pos + 8, y + 33), fill="black")
        draw.text((400, y - 40), f"当前分数 {score:.8g}；{data.get('interval', '')}", font=small, fill="black")
    draw.text((40, image.height - 52), "模型分数与分界线只用于筛查，不是概率。", font=small, fill="black")
    path = output / "branch_scores.png"
    _save_image(image, path)
    return path


def _video_chart(package: dict, output: Path) -> Path | None:
    face = next((mapping(v.get("data")) for v in rows(package.get("items")) if v.get("item_id") == "E04"), {})
    data = [(r.get("timestamp_seconds", r.get("frame_index", i)), finite(r.get("score", r.get("logit"))))
            for i, r in enumerate(rows(face.get("frame_scores")))]
    data = [(finite(t), v) for t, v in data if finite(t) is not None and v is not None]
    if not data:
        return None
    data.sort()
    image = Image.new("RGB", (1400, 600), "white")
    draw, font = ImageDraw.Draw(image), _font(26)
    draw.text((50, 25), "已选关键帧的人脸分数（不代表未采样时间段）", font=font, fill="black")
    values = [v for _, v in data]
    lower, upper = min(values), max(values)
    span = max(upper - lower, 0.1)
    lower, upper = lower - span * .15, upper + span * .15
    start, stop = min(t for t, _ in data), max(t for t, _ in data)
    points = [(100 + (t - start) / max(stop - start, 1) * 1200, 470 - (v - lower) / (upper - lower) * 350) for t, v in data]
    draw.line((100, 100, 100, 480, 1320, 480), fill="black", width=3)
    if len(points) > 1:
        draw.line(points, fill="black", width=3)
    for (x, y), (time_value, score) in zip(points, data):
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="black")
        draw.text((x - 30, y - 40), f"{score:.3f}", font=font, fill="black")
        draw.text((x - 30, 495), f"{time_value:g}", font=font, fill="black")
    draw.text((100, 550), "横轴：记录的时间点或帧序号；纵轴：当前后端模型分数", font=font, fill="black")
    path = output / "video_face_scores.png"
    _save_image(image, path)
    return path


def collect_figures(report: dict, package: dict, task_dir: str | Path, *, config: dict | None = None) -> list[dict]:
    """图件顺序固定；缺失的图只记缺口，不另造检测结果。"""
    root = Path(task_dir).resolve()
    output = _output_dir(root, "report_figures")
    uploads = _output_dir(root, "llm_upload")
    settings = mapping(mapping(config).get("llm"))
    max_side = int(settings.get("upload_max_side", 1280))
    if not 32 <= max_side <= 16384:
        raise ValueError("上传图件长边范围无效")
    module_blocks = mapping(report.get("modules"))
    raw = {name: mapping(mapping(module_blocks.get(name)).get("result")) for name in ("deepfake", "source", "identity")}
    face = mapping(mapping(raw["deepfake"].get("general_aigc")).get("face_branch"))
    task_input = mapping(mapping(mapping(report.get("provenance")).get("task_status")).get("input"))
    input_data = mapping(report.get("input"))
    frames = rows(task_input.get("key_frames")) or rows(mapping(input_data.get("video")).get("key_frames"))
    entries, omissions, seen = [], [], set()
    def add(value: Any, kind: str, module: str, section: str, description: str, required: bool = False, **extra) -> None:
        path = local_file(root, value)
        if path is None:
            if value:
                omissions.append({"type": kind, "reason": "任务内图件缺失或路径不受信任", "required": required})
            return
        identity = str(path)
        if identity in seen:
            existing = next(v for v in entries if v["original_path"] == identity)
            existing["required"] = existing["required"] or required
            return
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
        except (OSError, ValueError):
            omissions.append({"type": kind, "reason": "图件无法解码", "required": required})
            return
        seen.add(identity)
        entries.append({"type": kind, "module": module, "section": section, "description": description,
                        "required": required, "original_path": identity, "original_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "width": width, "height": height, **extra})
    highest = mapping(face.get("highest_score_face", face.get("highest_face")))
    annotation = {key: highest[key] for key in ("bbox", "face_id", "face_index", "frame_index") if key in highest}
    if frames:
        for index, frame in enumerate(frames[:6]):
            add(frame.get("path"), "key_frame", "input", "fixation", "视频已选关键帧，只代表此时间点", True,
                frame_index=frame.get("frame_rank", index), timestamp_seconds=frame.get("timestamp_seconds"))
    else:
        query = input_data.get("query_image") or task_input.get("query_image")
        query = local_file(root, query) or next((root / name for name in ("query.jpg", "query.png") if (root / name).is_file()), None)
        add(query, "original", "input", "fixation", "本次固定的输入图片", True, **annotation)
    per_face = rows(face.get("per_face")) or rows(face.get("records"))
    def highest_match(record: dict) -> bool:
        if not highest:
            return False
        if highest.get("file"):
            return record.get("file", record.get("crop_path")) == highest["file"]
        number_key = "face_id" if highest.get("face_id") is not None else "face_index"
        return (highest.get(number_key) is not None and record.get(number_key) == highest.get(number_key)
                and ("frame_index" not in highest or record.get("frame_index") == highest["frame_index"]))
    selected_faces = per_face[:8]
    highest_record = next((v for v in per_face if highest_match(v)), None)
    if highest_record is not None and not any(highest_match(v) for v in selected_faces):
        selected_faces = per_face[:7] + [highest_record]
    if len(per_face) > len(selected_faces):
        omissions.append({"type": "face_crop", "reason": "图件最多收录八张裁剪；最高分人脸优先保留，完整逐脸数值仍在结果中",
                          "omitted_count": len(per_face) - len(selected_faces), "required": False})
    for index, record in enumerate(selected_faces):
        name = record.get("file", record.get("crop_path"))
        path = local_file(root, name) or local_file(root, "face_crops/" + str(name or ""))
        is_highest = highest_match(record)
        add(path, "face_crop", "face", "detection", "模型实际分析的人脸裁剪，结合逐脸表阅读", bool(is_highest or (not highest and index == 0)),
            face_id=record.get("face_id"), face_index=record.get("face_index"), bbox=record.get("bbox"),
            score=record.get("score", record.get("logit")))
    for artifact in rows(raw["deepfake"].get("artifact_manifest")):
        if str(artifact.get("path", "")).lower().endswith((".png", ".jpg", ".jpeg")):
            add(artifact.get("path"), "detection_visual", "deepfake", "detection", "检测可视化；敏感图不是像素级伪造定位")
    for candidate in rows(raw["source"].get("candidates"))[:3]:
        add(candidate.get("candidate_image_artifact"), "source_candidate", "source", "source", "素材候选缩略图，不等于确认来源")
    pair = mapping(report.get("pair_compare"))
    for index, candidate in enumerate(rows(pair.get("candidates"))[:3]):
        for key, value in mapping(candidate.get("artifacts")).items():
            note = {"reference": "候选底图", "aligned": "几何对齐后的查询图", "difference": "对齐后的像素差异，亮处差异较大",
                    "change_mask": "规则阈值下的变化掩码"}.get(key, "同位置人脸或配对画面对照")
            add(value, "pair_" + key, "pair_compare", "source", note, index == 0 and key in {"reference", "aligned", "difference"}, candidate_rank=candidate.get("rank", index + 1))
    for candidate in rows(raw["source"].get("candidates"))[:3]:
        add(mapping(candidate.get("evidence_artifacts")).get("matches_inliers") or candidate.get("visualization_relative_path"),
            "sift_matches", "source", "source", "局部特征匹配与内点；不能单独确认同源")
    identity_visual = mapping(raw["identity"].get("visual_evidence"))
    add(identity_visual.get("face_crop_before_alignment"), "identity_query", "identity", "identity", "身份特征查询脸")
    for card in rows(identity_visual.get("candidate_card_images"))[:3]:
        add(card.get("path"), "identity_candidate", "identity", "identity", "身份候选最佳参考照，不等于身份确认")
    add(identity_visual.get("best_reference_comparison_image"), "identity_comparison", "identity", "identity", "查询脸与第一候选最佳参考照的并排对照", True)
    for index, hit in enumerate(rows(mapping(report.get("visible_label")).get("hits"))):
        frame_index = hit.get("frame_index", 0)
        source_path = frames[frame_index].get("path") if isinstance(frame_index, int) and 0 <= frame_index < len(frames) else input_data.get("query_image")
        source = local_file(root, source_path) or local_file(root, "query.jpg") or local_file(root, "query.png")
        box = hit.get("bbox")
        if source is None or not isinstance(box, list) or len(box) != 4 or any(finite(v) is None for v in box):
            omissions.append({"type": "visible_text", "reason": "命中字框或来源帧缺失", "required": True})
            continue
        with Image.open(source) as image:
            # 文字模块给出的坐标已还原到文件原始像素坐标，此处不转正后再裁。
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, y1, x2, y2 = max(0, x1 - 8), max(0, y1 - 8), min(image.width, x2 + 8), min(image.height, y2 + 8)
            if x2 <= x1 or y2 <= y1:
                continue
            path = output / f"visible_text_{index + 1:02d}.png"
            _save_image(image.crop((x1, y1, x2, y2)).convert("RGB"), path)
        add(path, "visible_text", "visible_label", "visible_text", "画面文字识别的命中区域，文字本身可能被添加", True)
    add(_score_chart(package, output), "branch_scores", "reporting", "detection", "各分支独立刻度的当前分数和两条分界线", True)
    if frames:
        video_chart = _video_chart(package, output)
        if video_chart:
            add(video_chart, "video_scores", "reporting", "detection", "实际记录的关键帧人脸分数曲线，未采样区间未知", True)
    for index, figure in enumerate(entries, 1):
        figure["figure_id"] = f"F{index:02d}"
        with Image.open(figure["original_path"]) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            for _ in range(12):
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=90)
                data = buffer.getvalue()
                if (len(data) + 2) // 3 * 4 <= int(settings.get("max_image_base64_bytes", 7 * 1024 * 1024)):
                    break
                image = image.resize((max(1, image.width * 3 // 4), max(1, image.height * 3 // 4)), Image.Resampling.LANCZOS)
            else:
                raise ValueError("图件缩小后仍超过上传大小上限")
            upload = uploads / (figure["figure_id"] + ".jpg")
            if upload.is_symlink():
                raise ValueError("图件上传副本不能是链接")
            upload.write_bytes(data)
            figure.update(upload_path=str(upload), sha256=hashlib.sha256(data).hexdigest(), upload_sha256=hashlib.sha256(data).hexdigest(),
                          upload_width=image.width, upload_height=image.height, upload_bytes=len(data), uploaded=False)
    manifest = {"figures": entries, "omissions": omissions}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entries
