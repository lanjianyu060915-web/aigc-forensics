"""研判、重试、回退和撰写坏段隔离的本地模拟测试。"""

from copy import deepcopy
import json

import pytest

from reporting.llm_client import DEFAULT_LLM_CONFIG, LLMClientError, LLMSettings
from reporting import llm_service


def evidence():
    return {"rule_band": "review", "constraints": {"floor_band": "review", "allow_obvious": False},
            "items": [{"item_id": "E03", "score": 0.75}], "figures": []}


def judgment():
    return {"final_band": "review", "score": 50,
            "evidence_assessment": [{"item_id": "E03", "observation": "分数0.75", "direction": "supports_aigc", "weight": "medium", "reason": "需要人工核查"}],
            "visual_observations": [], "band_reason": "检测分支提示风险", "score_reason": "单项中等证据", "relation_to_rule": "same",
            "adjustment_basis_ids": [], "adjustment_reason": "", "uncertainties": ["需人工核查"]}


def settings(mode="llm"):
    options = deepcopy(DEFAULT_LLM_CONFIG)
    options["decision_mode"] = mode
    return LLMSettings("http://127.0.0.1:1/v1", "mock", "local-test-secret", options)


def fake_trace(content):
    return {"request": {"model": "mock"}, "request_sha256": "a" * 64, "host": "127.0.0.1",
            "transmission_status": "response_received", "request_attempted": True, "included_request_figure_ids": [],
            "elapsed_seconds": 0.01, "response": {"model": "mock-actual", "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False), "reasoning_content": "不进入正文的模拟思考"}}]},
            "charged_usage": {"total_tokens": 15, "completion_tokens_details": {"reasoning_tokens": 3}}}


def test_valid_judge_and_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_service, "complete_json", lambda *args, **kwargs: fake_trace(judgment()))
    result = llm_service.run_judge(evidence(), settings=settings(), task_dir=tmp_path)
    assert (result["final_band"], result["score"], result["source"]) == ("review", 50, "llm")
    assert result["actual_model"] == "mock-actual"
    assert result["usage"]["reasoning_tokens"] == 3
    assert all((tmp_path / name).is_file() for name in result["audit_files"])
    assert "不进入正文的模拟思考" not in json.dumps(result, ensure_ascii=False)
    assert "local-test-secret" not in "".join(path.read_text(encoding="utf-8") for path in tmp_path.glob("*.json"))


@pytest.mark.parametrize("stage", ["judge", "writer"])
def test_both_stages_send_shared_model_view_without_mutating_local_evidence(monkeypatch, tmp_path, stage):
    view = evidence()
    view["figures"] = [{"figure_id": "F01", "sha256": "a" * 64, "width": 1280,
                        "required": True, "related_evidence_ids": ["E03"]}]
    original = deepcopy(view)
    calls = []
    def respond(*args, **kwargs):
        calls.append(kwargs["user_payload"]["evidence_package"])
        return fake_trace(judgment() if stage == "judge" else {})
    monkeypatch.setattr(llm_service, "complete_json", respond)
    configured = settings()
    configured.options["max_retries"] = 0
    if stage == "judge":
        llm_service.run_judge(view, settings=configured, task_dir=tmp_path)
    else:
        llm_service.run_writer(view, {"final_band": "review", "score": None, "source": "rule"}, settings=configured, task_dir=tmp_path)
    assert view == original and len(calls) == 1
    assert calls[0]["items"] == view["items"]
    assert "sha256" not in calls[0]["figures"][0] and "width" not in calls[0]["figures"][0]
    assert calls[0]["figures"][0]["related_evidence_ids"] == ["E03"]


def test_one_retry_then_success(monkeypatch, tmp_path):
    calls = []
    def respond(*args, **kwargs):
        calls.append(kwargs)
        content = judgment()
        if len(calls) == 1:
            content["score"] = 90
        return fake_trace(content)
    monkeypatch.setattr(llm_service, "complete_json", respond)
    result = llm_service.run_judge(evidence(), settings=settings(), task_dir=tmp_path)
    assert result["source"] == "llm" and len(calls) == 2
    assert calls[1]["retry_feedback"]
    assert result["usage"]["total_tokens"] == 30


def test_invalid_twice_falls_back_and_writer_not_called(monkeypatch, tmp_path):
    calls = []
    def respond(*args, **kwargs):
        calls.append(1)
        return fake_trace({"final_band": "none"})
    monkeypatch.setattr(llm_service, "complete_json", respond)
    result = llm_service.run_judge(evidence(), settings=settings(), task_dir=tmp_path)
    assert result["source"] == "rule_fallback" and result["score"] is None and result["final_band"] == "review"
    writer = llm_service.run_writer(evidence(), result, settings=settings(), task_dir=tmp_path)
    assert writer["status"] == "fallback" and len(calls) == 2


@pytest.mark.parametrize("kind", ["rule", "undetermined", "not_configured"])
def test_skip_paths_do_not_call_network(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(llm_service, "complete_json", lambda *args, **kwargs: pytest.fail("不应联网"))
    view = evidence()
    config = settings("rule" if kind == "rule" else "llm")
    if kind == "undetermined":
        view["rule_band"] = "undetermined"
    if kind == "not_configured":
        config = LLMSettings(options=config.options)
    result = llm_service.run_judge(view, settings=config, task_dir=tmp_path)
    assert result["score"] is None
    assert result["final_band"] == ("undetermined" if kind == "undetermined" else "review")


def test_authentication_failure_stops_after_one_request(monkeypatch, tmp_path):
    calls = []
    def failure(*args, **kwargs):
        calls.append(1)
        raise LLMClientError("authentication_failed", "接口鉴权失败", status=401)
    monkeypatch.setattr(llm_service, "complete_json", failure)
    result = llm_service.run_judge(evidence(), settings=settings(), task_dir=tmp_path)
    assert result["source"] == "rule_fallback" and len(calls) == 1


def test_connection_failure_is_bounded(monkeypatch, tmp_path):
    calls = []
    def failure(*args, **kwargs):
        calls.append(1)
        raise LLMClientError("connection_failed", "接口连接失败或超时")
    monkeypatch.setattr(llm_service, "complete_json", failure)
    result = llm_service.run_judge(evidence(), settings=settings(), task_dir=tmp_path)
    assert result["source"] == "rule_fallback" and len(calls) == 2


def test_writer_bad_section_is_removed(monkeypatch, tmp_path):
    phrase = "本次分析依据现有证据展开，检测结论的适用范围需要核查。各模块未完成的事项均不能推定为已完成，后续应由人员核实原始材料并确认结论。"
    value = {"summary": "建议人工复核，评分50。", "process": {key: "本步骤需要核查。" for key in ("fixation", "provenance", "visible_text", "detection", "source", "identity", "clues")},
             "synthesis": phrase * 14, "uncertainties": ["来源待核实", "存在适用范围", "需人工复核"], "next_steps": ["调取原件", "核实来源", "复核结论"], "figures": []}
    value["process"]["source"] = "确认伪造"
    monkeypatch.setattr(llm_service, "complete_json", lambda *args, **kwargs: fake_trace(value))
    result = llm_service.run_writer(evidence(), {"final_band": "review", "score": 50, "source": "rule"}, settings=settings("rule"), task_dir=tmp_path)
    assert result["status"] == "partial"
    assert "source" not in result["content"]["process"]
    assert "确认伪造" not in json.dumps(result["content"], ensure_ascii=False)


def test_parent_evidence_package_contract(monkeypatch, tmp_path):
    from reporting.evidence_package import build_evidence_package, to_llm_payload
    report = {"input": {"file_name": "synthetic.jpg"}, "task": {"task_id": "synthetic"}, "modules": {}}
    package = build_evidence_package(report, config={})
    assert all("item_id" in item and "data" in item for item in package["items"])
    view = to_llm_payload(package, "text_only")
    monkeypatch.setattr(llm_service, "complete_json", lambda *args, **kwargs: pytest.fail("没有可用档位不能调用研判"))
    result = llm_service.run_judge(view, settings=settings(), task_dir=tmp_path)
    assert result["final_band"] == "undetermined" and result["score"] is None


@pytest.mark.parametrize("content", ["非JSON", "```json\n{}\n```", "[]", '{"score":NaN}', "null"])
def test_non_json_or_wrong_structure_falls_back(monkeypatch, tmp_path, content):
    def respond(*args, **kwargs):
        trace = fake_trace({})
        trace["response"]["choices"][0]["message"]["content"] = content
        return trace
    monkeypatch.setattr(llm_service, "complete_json", respond)
    assert llm_service.run_judge(evidence(), settings=settings(), task_dir=tmp_path)["source"] == "rule_fallback"


def test_text_only_writer_never_opens_image(monkeypatch, tmp_path):
    received = []
    def respond(*args, **kwargs):
        received.append(kwargs)
        return fake_trace({})
    monkeypatch.setattr(llm_service, "complete_json", respond)
    config = settings("rule")
    config.options["upload_scope"] = "text_only"
    llm_service.run_writer(evidence(), {"final_band": "review", "score": None, "source": "rule"}, settings=config,
                           task_dir=tmp_path, figures=[{"figure_id": "F01", "upload_path": str(tmp_path / "不存在.jpg")}])
    assert received and all(not call["images"] for call in received)


def test_rule_without_endpoint_does_not_claim_writer_participated(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_service, "complete_json", lambda *args, **kwargs: pytest.fail("不应联网"))
    config = LLMSettings()
    result = llm_service.run_judge(evidence(), settings=config, task_dir=tmp_path)
    assert result["reason"] == "配置为规则定档"
    assert "撰写" not in result["reason"]
    assert llm_service.run_writer(evidence(), result, settings=config, task_dir=tmp_path)["status"] == "fallback"


def test_request_limit_removes_last_figures_before_network_retry(monkeypatch, tmp_path):
    pictures = [{"figure_id": key} for key in ("F01", "F02", "F03")]
    monkeypatch.setattr(llm_service, "prepare_images", lambda *args, **kwargs: deepcopy(pictures))
    checked, sent = [], []
    def respond(*args, **kwargs):
        ids = [item["figure_id"] for item in kwargs["images"]]
        checked.append(ids)
        view = kwargs["user_payload"]["evidence_package"]
        assert view["uploaded_figure_ids"] == ids
        assert [item["figure_id"] for item in view["figures"] if item["uploaded"]] == ids
        if len(ids) > 1:
            raise LLMClientError("request_too_large", "请求超过限制")
        sent.append(ids)
        trace = fake_trace(judgment())
        trace["included_request_figure_ids"] = ids
        return trace
    monkeypatch.setattr(llm_service, "complete_json", respond)
    view = evidence()
    view["figures"] = deepcopy(pictures)
    config = settings()
    config.options["max_retries"] = 0
    result = llm_service.run_judge(view, settings=config, task_dir=tmp_path)
    assert checked == [["F01", "F02", "F03"], ["F01", "F02"], ["F01"]]
    assert len(sent) == 1 and result["source"] == "llm"
    assert result["planned_figure_ids"] == ["F01", "F02", "F03"]
    assert result["uploaded_figure_ids"] == result["included_request_figure_ids"] == ["F01"]
    assert result["omitted_for_request_size"] == ["F02", "F03"]


@pytest.mark.parametrize("code", ["unsafe_payload", "request_too_large"])
def test_preflight_failure_never_claims_uploaded(monkeypatch, tmp_path, code):
    monkeypatch.setattr(llm_service, "prepare_images", lambda *args, **kwargs: [{"figure_id": "F01"}])
    def failure(*args, **kwargs):
        raise LLMClientError(code, "前置检查失败")
    monkeypatch.setattr(llm_service, "complete_json", failure)
    result = llm_service.run_judge(evidence(), settings=settings(), task_dir=tmp_path)
    assert result["source"] == "rule_fallback"
    assert result["planned_figure_ids"] == ["F01"]
    assert result["uploaded_figure_ids"] == result["attempted_figure_ids"] == []
    assert result["transmission_status"] == "not_sent"


def test_network_failure_distinguishes_unknown_delivery(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_service, "prepare_images", lambda *args, **kwargs: [{"figure_id": "F01"}])
    def failure(*args, **kwargs):
        error = LLMClientError("connection_failed", "连接失败")
        error.trace = {"transmission_status": "delivery_unknown", "request_attempted": True,
                       "included_request_figure_ids": ["F01"]}
        raise error
    monkeypatch.setattr(llm_service, "complete_json", failure)
    result = llm_service.run_judge(evidence(), settings=settings(), task_dir=tmp_path)
    assert result["uploaded_figure_ids"] == []
    assert result["attempted_figure_ids"] == result["delivery_unknown_figure_ids"] == ["F01"]
    assert result["transmission_note"] == "已尝试发送，送达未知"
