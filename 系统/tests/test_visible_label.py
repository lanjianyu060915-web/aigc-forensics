"""固定词表、坐标、故障隔离及真实离线文字识别的合成检验。"""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageFont
import pytest

from decision.aigc_evidence import attach_evidence
from modules.visible_label import analyze_visible_labels, visible_label_lines, LIMITATION
from modules.visible_label.core import convert_hit, deduplicate, match_text, load_keywords, normalize_text, restore_exif_coordinates, scan_views

ROOT = Path(__file__).resolve().parents[1]


def test_normalization_and_tool_boundary():
    words = load_keywords()
    assert normalize_text("ＡＩ 生 成") == "ai生成"
    assert match_text("由 ＡＩ 生成", False, words) == "visible_label_found"
    assert match_text("Sora", False, words) is None
    assert match_text("Sora", True, words) == "tool_hint_found"
    assert match_text("sorafenib", True, words) is None
    assert match_text("Made with S o r a", True, words) == "tool_hint_found"


def test_large_working_image_corner_coordinates_return_to_original():
    image = Image.new("RGB", (3000, 2000))
    views = scan_views(image)
    assert len(views) == 5
    assert max(views[0][0].size) == 1600
    view, x, y, sx, sy = views[-1]
    assert max(view.size) <= 960
    assert x + view.width / sx == pytest.approx(image.width)
    assert y + view.height / sy == pytest.approx(image.height)
    row = [[[view.width - 20, view.height - 10], [view.width, view.height - 10],
            [view.width, view.height], [view.width - 20, view.height]], "AI生成", .9]
    hit = convert_hit(row, width=image.width, height=image.height, scale_x=sx, scale_y=sy, offset_x=x, offset_y=y)
    assert hit["bbox"][2:] == [3000, 2000]
    assert hit["in_corner"]


def test_original_coordinates_and_duplicate_suppression():
    row = [[[0, 0], [100, 0], [100, 30], [0, 30]], "AI生成", .9]
    hit = convert_hit(row, width=1000, height=500, scale_x=2, scale_y=2, offset_x=700, offset_y=450)
    assert hit["bbox"] == [700, 450, 750, 465]
    assert hit["in_corner"] is True
    assert len(deduplicate([hit, deepcopy(hit)])) == 1
    center = convert_hit(row, width=1000, height=500, scale_x=1, scale_y=1, offset_x=400, offset_y=200)
    assert center["in_corner"] is False
    restored = restore_exif_coordinates(hit, 500, 1000, 6)
    assert restored["bbox"] == [450, 250, 465, 300]


@pytest.mark.parametrize("verdict", ["visible_label_found", "tool_hint_found", "no_label", "unavailable"])
@pytest.mark.parametrize("band", ["low", "medium", "high"])
def test_only_one_conflict_combination_and_no_band_change(verdict, band):
    original = {"status": "success", "overall": {"band": band}, "score": .2}
    before = deepcopy(original)
    visible = {"verdict": verdict, "hits": [{"text": "AI生成"}]}
    result = attach_evidence(original, {}, data={}, visible_label=visible)
    hints = [x for x in result["overall"]["hints"] if x["code"] == "visible_label_conflict"]
    assert bool(hints) is (verdict == "visible_label_found" and band == "low")
    assert result["overall"]["band"] == band
    assert original == before


@pytest.mark.parametrize("fault", ["missing", "timeout", "invalid"])
def test_unavailable_is_returned_without_exception(monkeypatch, tmp_path, fault):
    def broken(*args, **kwargs):
        if fault == "missing":
            raise FileNotFoundError()
        if fault == "timeout":
            raise subprocess.TimeoutExpired("文字识别", .1)
        return subprocess.CompletedProcess([], 0, stdout="不是结果", stderr="")
    monkeypatch.setattr(subprocess, "run", broken)
    result = analyze_visible_labels([{"path": tmp_path / "示例.png"}], timeout_seconds=.1)
    assert result["verdict"] == "unavailable"
    assert result["hits"] == []
    assert LIMITATION in result["limitations"]


def test_maximum_five_frames_and_paths_not_returned(monkeypatch, tmp_path):
    def fake(*args, **kwargs):
        request = json.loads(kwargs["input"])
        assert len(request["inputs"]) == 5
        assert kwargs["timeout"] == 30
        return subprocess.CompletedProcess([], 0, stdout='{"status":"success","verdict":"no_label","hits":[]}', stderr="")
    monkeypatch.setattr(subprocess, "run", fake)
    result = analyze_visible_labels([{"path": tmp_path / f"{i}.png"} for i in range(8)], timeout_seconds=60)
    assert result["status"] == "success"
    assert "path" not in json.dumps(result)
    assert visible_label_lines({}) == []
    assert "不参与检测档位" in visible_label_lines(result)[-1]


def test_real_child_process_is_terminated_on_timeout(monkeypatch, tmp_path):
    original_run, original_popen = subprocess.run, subprocess.Popen
    children = []
    def tracked(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child
    def sleeping(command, **kwargs):
        return original_run([sys.executable, "-B", "-c", "import time; time.sleep(20)"], **kwargs)
    monkeypatch.setattr(subprocess, "Popen", tracked)
    monkeypatch.setattr(subprocess, "run", sleeping)
    started = time.perf_counter()
    result = analyze_visible_labels([{"path": tmp_path / "示例.png"}], timeout_seconds=.2)
    assert result["verdict"] == "unavailable"
    assert time.perf_counter() - started < 5
    assert children and all(child.poll() is not None for child in children)


def _labelled(tmp_path, location, variation="original", text="AI生成"):
    image = Image.open(ROOT / "examples" / "no_face_image.jpg").convert("RGB")
    font = ImageFont.truetype(str(ROOT / "assets" / "fonts" / "LXGWWenKai-Regular.ttf"), max(12, round(min(image.size) * .05)))
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    pad = max(3, round(min(image.size) * .02))
    positions = {"tl": (pad, pad), "tr": (image.width - tw - pad, pad),
                 "bl": (pad, image.height - th - pad), "br": (image.width - tw - pad, image.height - th - pad),
                 "center": ((image.width - tw) // 2, (image.height - th) // 2)}
    x, y = positions[location]
    draw.rectangle((x - 2, y - 2, x + tw + 2, y + th + 2), fill="white")
    draw.text((x - box[0], y - box[1]), text, font=font, fill="black")
    if variation == "half":
        image = image.resize((image.width // 2, image.height // 2))
    path = tmp_path / f"{location}_{variation}.jpg"
    image.save(path, quality=60 if variation == "jpeg60" else 95)
    return path


@pytest.mark.parametrize("location", ["tl", "tr", "bl", "br"])
@pytest.mark.parametrize("variation", ["original", "jpeg60", "half"])
def test_real_ocr_synthetic_corners(tmp_path, location, variation):
    path = _labelled(tmp_path, location, variation)
    result = analyze_visible_labels([{"path": path}])
    assert result["verdict"] == "visible_label_found", result
    assert any(hit["in_corner"] for hit in result["hits"])


def test_real_ocr_center_and_tool_name(tmp_path):
    result = analyze_visible_labels([{"path": _labelled(tmp_path, "center")}])
    assert result["verdict"] == "visible_label_found", result
    assert not any(hit["in_corner"] for hit in result["hits"])
    result = analyze_visible_labels([{"path": _labelled(tmp_path, "center", text="Sora")}])
    assert result["verdict"] == "no_label", result


def test_real_ocr_exif_rotation_restores_raw_coordinates(tmp_path):
    source = _labelled(tmp_path, "tl")
    image = Image.open(source).transpose(Image.Transpose.ROTATE_90)
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "方向标记.jpg"
    image.save(path, exif=exif, quality=95)
    result = analyze_visible_labels([{"path": path}])
    assert result["verdict"] == "visible_label_found", result
    assert all(hit["bbox"][1] > image.height * .7 for hit in result["hits"])


def test_real_ocr_large_label_keeps_original_corner_coordinates(tmp_path):
    source = _labelled(tmp_path, "br")
    path = tmp_path / "大图角落.jpg"
    Image.open(source).resize((3000, 2000)).save(path, quality=60)
    result = analyze_visible_labels([{"path": path}])
    assert result["verdict"] == "visible_label_found", result
    assert all(hit["bbox"][0] > 2100 and hit["bbox"][1] > 1400 for hit in result["hits"])


@pytest.mark.parametrize("path", sorted((ROOT / "examples").glob("*.jpg")), ids=lambda p: p.name)
def test_real_ocr_existing_images_do_not_match(path):
    result = analyze_visible_labels([{"path": path}])
    assert result["verdict"] == "no_label", result


@pytest.mark.parametrize("failure", [False, True])
def test_module_failure_does_not_change_task_and_reports_remain(tmp_path, monkeypatch, failure):
    import orchestrator
    from test_general_aigc_orchestration import _config, _section, _write_runner, _query

    def recognize(inputs):
        if failure:
            raise RuntimeError("替身模拟识别不可用")
        return {"status": "success", "verdict": "visible_label_found", "hits": [{"text": "AI生成", "bbox": [1, 2, 3, 4], "in_corner": True}], "limitations": [LIMITATION]}
    monkeypatch.setattr(orchestrator, "analyze_visible_labels", recognize)
    config = _config(tmp_path, general=_section(_write_runner(tmp_path / "runner.py")))
    status = orchestrator.run_task(_query(tmp_path), selected_modules={"deepfake"}, config_path=config)
    task = Path(status["task_dir"])
    assert status["status"] == "success"
    assert status["visible_label"]["verdict"] == ("unavailable" if failure else "visible_label_found")
    report = json.loads((task / "final_report.json").read_text(encoding="utf-8"))
    assert report["modules"]["deepfake"]["result"]["overall"]["band"] == "medium"
    manifest = json.loads((task / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert "visible_label_result.json" in {item["path"] for item in manifest["artifacts"]}
    assert list(task.glob("*.pdf"))
    assert "画面文字标识" in next(task.glob("*.txt")).read_text(encoding="utf-8")
    assert "画面文字标识" in next(task.glob("*.html")).read_text(encoding="utf-8")
