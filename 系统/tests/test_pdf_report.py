"""PDF 生成、失败降级和只追加复核的报告契约。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from PIL import Image
import pytest

from decision.aigc_evidence import attach_evidence
from decision.aigc_overall import attach_general_aigc
from reporting import pdf_report
from reporting.report_builder import build_final_report, generate_task_reports
from reporting.review_records import ReviewRecordError, append_review_record, load_review_records


def sample_report(tmp_path: Path, *, kind="image", face=True, evidence=True):
    Image.new("RGB", (120, 180), (45, 90, 160)).save(tmp_path / "query.jpg")
    calibration = json.loads((Path(__file__).resolve().parents[1] / "models/video-c38-refit3-cf384/calibration.json").read_text(encoding="utf-8"))["face_crop"]
    general = {
        "status": "success", "input_kind": kind, "band": "low", "band_label": "证据不足", "logit": -3,
        "media": {"width": 120, "height": 180}, "model": {"model_id": "验收模型", "calibration_version": "验收版本"},
        "band_thresholds": {"low_max": -2, "medium_max": -1},
        "face_branch": {"backend": "cf384", "band_aggregation": "largest",
                        "band_thresholds": {key: calibration[key] for key in ("low_max", "medium_max")},
                        "status": "success" if face else "no_face", "band": "low" if face else None, "band_label": "证据不足" if face else "未检出人脸", "logit": -6},
        "frames": [{"timestamp_seconds": 0, "clip_logit": -3}, {"timestamp_seconds": 1, "clip_logit": -2}],
    }
    (tmp_path / "general_aigc_result.json").write_text(json.dumps(general), encoding="utf-8")
    if face:
        (tmp_path / "face_crops").mkdir(exist_ok=True)
        Image.new("RGB", (100, 100), (120, 80, 30)).save(tmp_path / "face_crops/人脸.jpg")
    original = {"status": "success", "query_image": "query.jpg", "score": 0.1, "details": {"band": "low"}, "method": {"branch": "vit_full_image"}}
    merged = attach_general_aigc(original, general)
    if evidence:
        merged = attach_evidence(merged, {})
    status = {"status": "success"}
    if kind == "video":
        status["input"] = {"type": "video", "video_file": "原视频.mp4", "video_sha256": "a" * 64, "key_frames": [{"path": "query.jpg", "timestamp_seconds": 0}]}
    return build_final_report("报告验收", task_status=status, deepfake_result=merged, identity_result={"status": "skipped"}, source_result={"status": "success", "candidates": []})


@pytest.mark.parametrize(("kind", "face", "evidence"), [("image", True, True), ("video", True, True), ("image", False, True), ("image", False, False)])
def test_image_video_no_face_no_candidates_and_old_results(tmp_path, kind, face, evidence):
    report = sample_report(tmp_path, kind=kind, face=face, evidence=evidence)
    before = deepcopy(report)
    raw = pdf_report.render_pdf_report(report, tmp_path)
    assert raw.startswith(b"%PDF-") and raw.rstrip().endswith(b"%%EOF")
    assert len(re.findall(rb"/Type\s*/Page(?!s)", raw)) >= 2
    assert report == before


def test_markup_in_filename_and_opinion_is_plain_text(tmp_path):
    report = sample_report(tmp_path)
    report["input"]["file_name"] = "查询<b>&.jpg"
    records = [{"reviewer": "复核<b>&", "conclusion": "无法判断", "opinion": "<b>&不是标签\n仍需复核", "reviewed_at": "2026-11-05T10:00:00+08:00", "report_sha256": "a" * 64}]
    raw = pdf_report.render_pdf_report(report, tmp_path, records)
    assert raw.startswith(b"%PDF-")


def test_no_face_identity_keeps_not_applicable_explanation(tmp_path):
    report = sample_report(tmp_path, face=False)
    reason = "未检测到人脸，人脸身份比对不适用"
    report["modules"]["identity"]["result"].update(status="skipped", details={"reason": reason})
    section = pdf_report._source_section(report, tmp_path, pdf_report._styles())
    text = "\n".join(item.getPlainText() for item in section if hasattr(item, "getPlainText"))
    assert reason in text
    assert "身份比对未运行" not in text


@pytest.mark.parametrize("local_path", ["C:\\private\\outside.jpg", "C:/private/outside.jpg", "\\\\server\\share\\outside.jpg"])
def test_path_after_chinese_text_is_not_printed(local_path):
    assert pdf_report._text("请参考" + local_path) == "请参考outside.jpg"


def test_web_link_in_review_is_not_treated_as_local_file():
    opinion = "参考 https://example.org/review?a=1&b=2"
    assert pdf_report._text(opinion) == opinion


def test_reviews_are_append_only_and_do_not_replace_original_reports(tmp_path):
    output = generate_task_reports(tmp_path, task_id="复核验收")
    assert output["pdf"]["status"] == "success"
    original = {name: (tmp_path / name).read_bytes() for name in output["file_names"].values()}
    digest = hashlib.sha256(original["final_report.json"]).hexdigest()
    first = append_review_record(tmp_path, "复核甲", "不同意", "<b>&需要核查")
    first_bytes = (tmp_path / "review_records.jsonl").read_bytes()
    second = append_review_record(tmp_path, "复核乙", "无法判断", "补充意见")
    assert (tmp_path / "review_records.jsonl").read_bytes().startswith(first_bytes)
    assert load_review_records(tmp_path) == [first, second]
    assert first["report_sha256"] == second["report_sha256"] == digest
    assert first["reviewed_at"]
    pdf_report.render_pdf_report(output["report"], tmp_path, [first, second])
    assert {name: (tmp_path / name).read_bytes() for name in original} == original


@pytest.mark.parametrize(("reviewer", "conclusion", "opinion"), [("", "同意系统结论", "意见"), ("甲", "同意系统结论", "  "), ("甲", "未提供的选项", "意见")])
def test_incomplete_review_is_not_saved(tmp_path, reviewer, conclusion, opinion):
    (tmp_path / "final_report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReviewRecordError):
        append_review_record(tmp_path, reviewer, conclusion, opinion)
    assert not (tmp_path / "review_records.jsonl").exists()


def test_missing_font_preserves_task_status_and_three_original_formats(tmp_path, monkeypatch):
    from test_general_aigc_orchestration import _config, _query, _section, _write_runner
    from orchestrator import run_task

    def missing():
        raise FileNotFoundError("验收使用的字体替身不可用")

    monkeypatch.setattr(pdf_report, "check_pdf_ready", missing)
    config = _config(tmp_path, general=_section(_write_runner(tmp_path / "runner.py")))
    status = run_task(_query(tmp_path), selected_modules={"deepfake"}, config_path=config)
    assert status["status"] == "success"
    assert status["reports"]["pdf"]["status"] == "unavailable"
    assert "PDF 报告暂不可用" in status["reports"]["pdf"]["warning"]
    assert set(status["reports"]["files"]) == {"json", "txt", "html"}
    root = Path(status["task_dir"])
    assert all((root / name).is_file() for name in status["reports"]["files"].values())
    assert not list(root.glob("*.pdf"))
    result = json.loads((root / "detection_result.json").read_text(encoding="utf-8"))
    assert result["general_aigc"]["logit"] == 1.4 and result["overall"]["band"] == "medium"
