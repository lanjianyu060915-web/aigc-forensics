"""验证原图保留、上传缩略图、图件固定顺序与越界隔离。"""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from reporting.evidence_figures import collect_figures, local_file


def package():
    return {"items": [{"item_id": key, "title": title, "data": {"score": value, "thresholds": {"low_max": -2, "medium_max": 1}}}
                      for key, title, value in (("E03", "通用", 2), ("E04", "人脸", 10), ("E05", "整图", .7))]}


def task(tmp_path):
    query = tmp_path / "query.jpg"
    Image.new("RGB", (2400, 1200), "red").save(query)
    return {"input": {"query_image": str(query)}, "modules": {}, "visible_label": {}}, query


def test_figures_have_separate_original_and_upload_hashes(tmp_path):
    report, query = task(tmp_path)
    before = query.read_bytes()
    figures = collect_figures(report, package(), tmp_path)
    assert [v["figure_id"] for v in figures] == ["F01", "F02"]
    original = figures[0]
    assert original["required"]
    assert original["original_sha256"] == hashlib.sha256(before).hexdigest()
    assert original["sha256"] == hashlib.sha256(Path(original["upload_path"]).read_bytes()).hexdigest()
    assert original["original_sha256"] != original["sha256"]
    assert max(original["upload_width"], original["upload_height"]) <= 1280
    assert query.read_bytes() == before
    with Image.open(figures[-1]["original_path"]) as image:
        rgb = image.convert("RGB")
        assert rgb.getchannel("R").tobytes() == rgb.getchannel("G").tobytes() == rgb.getchannel("B").tobytes()


def test_visible_text_crop_is_a_separate_required_evidence(tmp_path):
    report, query = task(tmp_path)
    report["visible_label"] = {"hits": [{"bbox": [10, 20, 110, 80], "frame_index": 0}]}
    figures = collect_figures(report, package(), tmp_path)
    text = next(v for v in figures if v["type"] == "visible_text")
    assert text["required"] and text["width"] == 116 and text["height"] == 76
    assert text["original_path"] != str(query)


def test_path_escape_is_never_read(tmp_path):
    root = tmp_path / "task"
    root.mkdir()
    outside = tmp_path / "secret.png"
    Image.new("RGB", (20, 20)).save(outside)
    assert local_file(root, outside) is None
    assert local_file(root, "../secret.png") is None
    report = {"input": {"query_image": str(outside)}, "modules": {}}
    figures = collect_figures(report, package(), root)
    assert all(v["type"] != "original" for v in figures)


def test_missing_video_frame_scores_do_not_create_fictitious_curve(tmp_path):
    report, query = task(tmp_path)
    report["input"]["video"] = {"key_frames": [{"path": str(query), "frame_rank": 0, "timestamp_seconds": 1.25}]}
    figures = collect_figures(report, package(), tmp_path)
    assert figures[0]["type"] == "key_frame"
    assert all(v["type"] != "video_scores" for v in figures)


def test_existing_per_frame_scores_make_curve(tmp_path):
    report, query = task(tmp_path)
    report["input"]["video"] = {"key_frames": [{"path": str(query)}]}
    evidence = package()
    evidence["items"][1]["data"]["frame_scores"] = [{"timestamp_seconds": .5, "score": .1}, {"timestamp_seconds": 2.5, "score": .9}]
    figures = collect_figures(report, evidence, tmp_path)
    assert next(v for v in figures if v["type"] == "video_scores")["required"]


def test_upload_size_limit_is_respected(tmp_path):
    report, _ = task(tmp_path)
    figures = collect_figures(report, package(), tmp_path, config={"llm": {"upload_max_side": 64, "max_image_base64_bytes": 10000}})
    assert all(max(v["upload_width"], v["upload_height"]) <= 64 for v in figures)
    assert all((v["upload_bytes"] + 2) // 3 * 4 <= 10000 for v in figures)


def test_highest_face_from_late_video_frame_is_not_truncated(tmp_path):
    report, _ = task(tmp_path)
    folder = tmp_path / "face_crops"
    folder.mkdir()
    faces = []
    for index in range(12):
        name = f"frame_{index}_face.jpg"
        Image.new("RGB", (70, 80), "gray").save(folder / name)
        faces.append({"file": name, "face_index": 0, "frame_index": index, "score": index})
    report["modules"] = {"deepfake": {"result": {"general_aigc": {"face_branch": {
        "per_face": faces, "highest_score_face": faces[-1]}}}}}
    figures = collect_figures(report, package(), tmp_path)
    crops = [v for v in figures if v["type"] == "face_crop"]
    assert len(crops) == 8
    assert [v["score"] for v in crops if v["required"]] == [11]
    assert Path(crops[-1]["original_path"]).name == faces[-1]["file"]
