"""历史结果补生成、不可覆盖原件、上传降级与独立证据完整性。"""
import hashlib
import json
from copy import deepcopy

import pytest

from reporting import analysis_pipeline as pipeline
from reporting.llm_client import LLMSettings
from reporting.report_builder import generate_task_reports
from reporting.review_records import append_review_record
from frontend.components.analysis_views import load_analysis
from frontend.components.task_history import verify_task_artifacts


def prepare(tmp_path):
    result = generate_task_reports(tmp_path, task_id="历史报告")
    (tmp_path / "detection_result.json").write_text("{}", encoding="utf-8")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir() if p.is_file()}
    report = deepcopy(result["report"])
    saved = pipeline.generate_analysis_report(tmp_path, report, config={"llm": {}}, settings=LLMSettings())
    return report, saved, before


def test_legacy_append_and_cached_reuse_do_not_call_again(tmp_path, monkeypatch):
    report, saved, before = prepare(tmp_path)
    assert saved["status"] == "success"
    assert saved["final_decision"]["final_band"] == "undetermined"
    assert saved["external_language_model_used"] is False
    assert before == {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in before}
    assert not pipeline.verify_analysis_artifacts(tmp_path)
    verification = verify_task_artifacts(tmp_path)
    assert verification["available"] and not verification["problems"]
    assert load_analysis(tmp_path, report)["source_report_sha256"] == saved["source_report_sha256"]
    monkeypatch.setattr(pipeline, "run_judge", lambda *a, **k: pytest.fail("缓存不应再次请求研判"))
    monkeypatch.setattr(pipeline, "run_writer", lambda *a, **k: pytest.fail("缓存不应再次请求撰写"))
    assert pipeline.generate_analysis_report(tmp_path, report, config={}) == saved


def test_changed_input_or_force_cannot_overwrite_original(tmp_path):
    report, saved, before = prepare(tmp_path)
    original = (tmp_path / saved["pdf_file"]).read_bytes()
    with pytest.raises(ValueError, match="不能覆盖"):
        pipeline.generate_analysis_report(tmp_path, report, config={}, force=True)
    changed = deepcopy(report)
    changed["task"]["status"] = "changed"
    with pytest.raises(ValueError, match="不同来源"):
        pipeline.generate_analysis_report(tmp_path, changed, config={})
    with pytest.raises(ValueError, match="不一致"):
        load_analysis(tmp_path, changed)
    assert (tmp_path / saved["pdf_file"]).read_bytes() == original
    (tmp_path / "detection_result.json").write_text('{"tampered":true}', encoding="utf-8")
    assert pipeline.verify_analysis_artifacts(tmp_path)
    assert verify_task_artifacts(tmp_path)["problems"]
    with pytest.raises(ValueError, match="证据缺失或发生变化"):
        pipeline.generate_analysis_report(tmp_path, report, config={})


def test_review_pdf_is_separate_and_verified(tmp_path):
    report, saved, before = prepare(tmp_path)
    original_pdf = (tmp_path / saved["pdf_file"]).read_bytes()
    record = append_review_record(tmp_path, "测试复核员", "无法判断", "模拟复核，需核查输入。")
    raw = pipeline.generate_review_analysis(tmp_path, reviews=[record])
    assert raw.startswith(b"%PDF-")
    assert (tmp_path / saved["pdf_file"]).read_bytes() == original_pdf
    assert not verify_task_artifacts(tmp_path)["problems"]
    (tmp_path / "研判分析报告（复核版）.pdf").write_bytes(raw + b"changed")
    assert verify_task_artifacts(tmp_path)["problems"]


@pytest.mark.parametrize("target", ["analysis_report.json", "研判分析报告.pdf", "manifest_entry"])
def test_ui_rejects_altered_report_pdf_or_missing_manifest_entry(tmp_path, target):
    report, saved, before = prepare(tmp_path)
    if target == "analysis_report.json":
        changed = deepcopy(saved)
        changed["final_decision"].update(final_band="obvious", score=99)
        (tmp_path / target).write_text(json.dumps(changed), encoding="utf-8")
    elif target == "manifest_entry":
        path = tmp_path / "analysis_artifact_manifest.json"
        manifest = json.loads(path.read_text("utf-8"))
        manifest["artifacts"] = [r for r in manifest["artifacts"] if r["path"] != "analysis_report.json"]
        path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        path = tmp_path / target
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="证据清单"):
        load_analysis(tmp_path, report)


def test_settings_write_only_password_and_environment_precedence(tmp_path):
    from frontend.components.llm_views import save_llm_profile, upload_notice
    from reporting.llm_client import load_llm_settings
    path = tmp_path / "profile" / "llm.json"
    kwargs = dict(base_url="https://example.test/v1", model="test-model", upload_scope="text_only",
                  decision_mode="rule", judge_send_images=True)
    save_llm_profile(path, api_key="dummy-test-only", **kwargs)
    save_llm_profile(path, api_key="", **kwargs)
    state = load_llm_settings({}, profile_path=path, environ={"AIGC_LLM_MODEL": "environment-model"})
    assert state.api_key == "dummy-test-only" and state.model == "environment-model"
    assert "api_key" not in state.public_status()
    assert "dummy-test-only" not in repr(state)
    assert "不上传图片" in upload_notice(state)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_llm_profile(path, api_key="", **{**kwargs, "base_url": "https://username:secret@example.test/v1"})
    assert path.read_bytes() == before


def test_video_evidence_uses_video_hash_size_fps_and_bound_sampling(tmp_path):
    from reporting.evidence_package import build_evidence_package
    raw = b"fixture-video-bytes"
    (tmp_path / "query_video.mp4").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    frame_sha = "a" * 64
    report = {"input": {"query_image": "query.jpg", "input_sha256": frame_sha,
                        "video": {"video_file": "query_video.mp4", "video_sha256": sha}},
              "provenance": {"task_status": {"input": {"type": "video", "video_file": "query_video.mp4", "video_sha256": sha,
                    "video_info": {"width": 640, "height": 360, "fps": 25.0, "duration_seconds": 6.0},
                    "frame_selection": "frame_selection.json"}}}}
    selection = {"video_sha256": sha, "status": "success", "sampling": {"candidate_frames_decoded": 24},
                 "selection": {"selected_frames": 5}}
    path = tmp_path / "frame_selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    package = build_evidence_package(report)
    pipeline._complete_input_evidence(tmp_path, report, package)
    material, process = package["items"][8]["data"], package["items"][9]["data"]
    assert material["input_sha256"] == sha != frame_sha
    assert material["primary_frame_sha256"] == frame_sha
    assert material["hash_mismatch"] is False
    assert material["size_bytes"] == len(raw) and material["fps"] == 25.0
    assert process["candidate_frame_count"] == 24 and process["selected_frame_count"] == 5
    selection["video_sha256"] = frame_sha
    path.write_text(json.dumps(selection), encoding="utf-8")
    pipeline._complete_input_evidence(tmp_path, report, package)
    assert process["frame_selection"]["status"] == "unavailable"


def test_figure_transmission_distinguishes_response_unknown_and_not_sent(tmp_path):
    folder = tmp_path / "report_figures"
    folder.mkdir()
    (folder / "manifest.json").write_text('{"figures":[],"omissions":[]}', encoding="utf-8")
    figures = [{"figure_id": "F01"}, {"figure_id": "F02"}, {"figure_id": "F03"}]
    final = {"uploaded_figure_ids": ["F01"], "attempted_figure_ids": ["F01"], "transmission_status": "response_received"}
    writer = {"delivery_unknown_figure_ids": ["F02"], "attempted_figure_ids": ["F02"],
              "transmission_status": "delivery_unknown", "omitted_for_request_size": ["F03"]}
    result = pipeline._record_figure_transmission(tmp_path, figures, final, writer, LLMSettings())
    assert result["upload_scope"] == "full"
    assert figures[0]["uploaded"] and not figures[0]["delivery_unknown"]
    assert not figures[1]["uploaded"] and figures[1]["delivery_unknown"]
    assert not figures[2]["uploaded"] and not figures[2]["delivery_unknown"]
    assert figures[2]["transmission"]["writer"]["omitted_for_request_size"]


@pytest.mark.parametrize("url", [
    "https://test-user:dummy-url-password@example.test/v1",
    "https://example.test/v1?token=dummy-url-password",
])
def test_invalid_endpoint_credentials_are_not_echoed_by_settings_ui(monkeypatch, tmp_path, url):
    from contextlib import nullcontext
    from frontend.components import llm_views
    from reporting.llm_client import load_llm_settings

    class UI:
        session_state = {}
        def __init__(self):
            self.inputs = []
            self.buttons = []
        def text_input(self, label, **kwargs):
            self.inputs.append({"label": label, **kwargs})
            return kwargs.get("value", "")
        def form(self, *args, **kwargs):
            return nullcontext()
        def selectbox(self, label, options, **kwargs):
            return options[kwargs.get("index", 0)]
        def checkbox(self, *args, **kwargs):
            return kwargs.get("value", False)
        def form_submit_button(self, *args, **kwargs):
            return False
        def button(self, label, **kwargs):
            self.buttons.append({"label": label, **kwargs})
            return False
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    ui = UI()
    settings = load_llm_settings({"base_url": url, "model": "test-model", "api_key": "dummy-api-key"}, environ={})
    assert settings.configuration_error and not settings.configured
    monkeypatch.setattr(llm_views, "st", ui)
    monkeypatch.setattr(llm_views, "llm_context", lambda _: ({}, tmp_path/'llm.json', settings))
    llm_views.render_llm_settings('unused')
    assert next(row for row in ui.inputs if row["label"] == "兼容接口地址")["value"] == ""
    assert next(row for row in ui.inputs if row.get("type") == "password")["value"] == ""
    assert "dummy-url-password" not in json.dumps(ui.inputs)
    assert "dummy-api-key" not in json.dumps(ui.inputs)
    assert next(row for row in ui.buttons if row["label"] == "测试连接")["disabled"] is True
