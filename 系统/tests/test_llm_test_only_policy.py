"""仅用合成配置验证测试资料云端保护，不发送网络请求或读取图片。"""

from contextlib import nullcontext
import json

import pytest

from reporting.llm_client import LLMClientError, complete_json, load_llm_settings, prepare_images


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
POLICY = {"cloud_text_only": True, "allowed_cloud_base_url": BASE_URL, "dataset_id": "synthetic_test_only"}
CREDENTIALS = {"base_url": BASE_URL, "model": "synthetic-model", "api_key": "synthetic-secret"}


@pytest.mark.parametrize("stage", ["judge", "writer"])
def test_root_policy_overrides_profile_without_reading_images(tmp_path, stage):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({**CREDENTIALS, "upload_scope": "full", "judge_send_images": True,
                                   "test_only_policy": {"cloud_text_only": False}}), encoding="utf-8")
    config = {"llm": {"upload_scope": "full", "judge_send_images": True}, "test_only_policy": POLICY}
    settings = load_llm_settings(config, profile_path=profile, environ={
        "AIGC_LLM_BASE_URL": BASE_URL, "AIGC_LLM_UPLOAD_SCOPE": "full", "AIGC_LLM_JUDGE_SEND_IMAGES": "true"})
    assert settings.configured
    assert settings.options["upload_scope"] == "text_only"
    assert settings.options["judge_send_images"] is False
    assert settings.options["decision_mode"] == "rule"
    # 不存在的上传副本确保只要尝试读图就会失败。
    assert prepare_images([{"figure_id": "F01", "upload_path": str(tmp_path / "absent.png")}], settings, stage=stage) == []
    assert "synthetic-secret" not in json.dumps(settings.public_status())


@pytest.mark.parametrize("source", ["config", "profile", "environment"])
@pytest.mark.parametrize("bad_url", ["https://other.invalid/v1", "https://dashscope.aliyuncs.com.evil.invalid/compatible-mode/v1",
                                    "http://dashscope.aliyuncs.com/compatible-mode/v1"])
def test_disallowed_endpoint_never_reaches_transport(tmp_path, monkeypatch, source, bad_url):
    config = {"llm": dict(CREDENTIALS), "test_only_policy": POLICY}
    profile = tmp_path / "profile.json"
    environ = {}
    if source == "config":
        config["llm"]["base_url"] = bad_url
    elif source == "profile":
        profile.write_text(json.dumps({"base_url": bad_url}), encoding="utf-8")
    else:
        environ["AIGC_LLM_BASE_URL"] = bad_url
    settings = load_llm_settings(config, profile_path=profile, environ=environ)
    assert settings.configuration_error and not settings.configured
    monkeypatch.setattr("urllib.request.build_opener", lambda *_: pytest.fail("不允许触达网络传输"))
    with pytest.raises(LLMClientError) as failure:
        complete_json(settings, system_prompt="合成测试", user_payload={}, stage="writer")
    assert failure.value.code == "not_configured"
    assert bad_url not in str(failure.value)


@pytest.mark.parametrize("allowed", ["", None, "https://other.invalid/v1", "https://dashscope.aliyuncs.com/compatible-mode/v1?token=synthetic"])
def test_invalid_policy_endpoint_is_rejected(allowed):
    settings = load_llm_settings({"llm": CREDENTIALS, "test_only_policy": {**POLICY, "allowed_cloud_base_url": allowed}}, environ={})
    assert not settings.configured and settings.configuration_error


def test_policy_normalizes_only_terminal_slash():
    settings = load_llm_settings({"llm": {**CREDENTIALS, "base_url": BASE_URL + "/"},
                                  "test_only_policy": {**POLICY, "allowed_cloud_base_url": BASE_URL + "/"}}, environ={})
    assert settings.configured and settings.base_url == BASE_URL


def test_no_policy_preserves_normal_configuration(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({**CREDENTIALS, "base_url": "https://normal.invalid/v1",
                                   "upload_scope": "full", "judge_send_images": True}), encoding="utf-8")
    settings = load_llm_settings({"llm": {}}, profile_path=profile, environ={})
    assert settings.configured and settings.base_url == "https://normal.invalid/v1"
    assert settings.options["upload_scope"] == "full" and settings.options["judge_send_images"] is True


def test_text_only_rejects_direct_images_before_transport(monkeypatch):
    settings = load_llm_settings({"llm": CREDENTIALS, "test_only_policy": POLICY}, environ={})
    monkeypatch.setattr("urllib.request.build_opener", lambda *_: pytest.fail("不允许触达网络传输"))
    with pytest.raises(LLMClientError) as failure:
        complete_json(settings, system_prompt="合成测试", user_payload={}, stage="writer",
                      images=[{"figure_id": "F01", "data_url": "data:image/png;base64,synthetic"}])
    assert failure.value.code == "unsafe_payload"


@pytest.mark.parametrize("configured", [False, True])
def test_status_separates_rule_decision_and_writer_capability(tmp_path, monkeypatch, configured):
    from frontend.components import llm_views

    class UI:
        def table(self, rows):
            self.rows = rows
        def __getattr__(self, name):
            return lambda *args: None

    config = {"llm": CREDENTIALS if configured else {}, "test_only_policy": POLICY}
    settings = load_llm_settings(config, environ={})
    ui = UI()
    monkeypatch.setattr(llm_views, "st", ui)
    monkeypatch.setattr(llm_views, "llm_context", lambda _: (config, tmp_path / "llm.json", settings))
    llm_views.render_llm_status("synthetic")
    rows = {row["项目"]: row["当前值"] for row in ui.rows}
    assert rows["研判方式"] == "规则研判"
    assert rows["报告撰写"] == ("云端辅助（已配置，按任务调用）" if configured else "本地模板")


def test_profile_save_cannot_persist_full_under_root_policy(tmp_path):
    from frontend.components.llm_views import save_llm_profile
    path = tmp_path / "llm.json"
    save_llm_profile(path, **CREDENTIALS, upload_scope="full", decision_mode="rule", judge_send_images=True,
                     config={"test_only_policy": POLICY})
    saved = json.loads(path.read_text("utf-8"))
    assert saved["upload_scope"] == "text_only" and saved["judge_send_images"] is False
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_llm_profile(path, **{**CREDENTIALS, "base_url": "https://other.invalid/v1"}, upload_scope="full",
                         decision_mode="rule", judge_send_images=True, config={"test_only_policy": POLICY})
    assert path.read_bytes() == before


def test_ui_displays_and_locks_effective_text_only_policy(tmp_path, monkeypatch):
    from frontend.components import llm_views

    class UI:
        def __init__(self):
            self.session_state = {}
            self.widgets = {}
            self.messages = []
        def text_input(self, label, **kwargs):
            self.widgets[label] = kwargs
            return kwargs.get("value", "")
        def selectbox(self, label, options, **kwargs):
            self.widgets[label] = {"options": options, **kwargs}
            return options[kwargs.get("index", 0)]
        def checkbox(self, label, **kwargs):
            self.widgets[label] = kwargs
            return kwargs.get("value", False)
        def form(self, *args, **kwargs):
            return nullcontext()
        def form_submit_button(self, *args, **kwargs):
            return False
        def button(self, *args, **kwargs):
            return False
        def __getattr__(self, name):
            return lambda text: self.messages.append(text)

    config = {"llm": {**CREDENTIALS, "upload_scope": "full", "judge_send_images": True}, "test_only_policy": POLICY}
    settings = load_llm_settings(config, environ={})
    ui = UI()
    monkeypatch.setattr(llm_views, "st", ui)
    monkeypatch.setattr(llm_views, "llm_context", lambda _: (config, tmp_path / "llm.json", settings))
    llm_views.render_llm_settings("synthetic")
    assert ui.widgets["上传范围"]["options"] == ["text_only"]
    assert ui.widgets["上传范围"]["disabled"]
    assert ui.widgets["研判时附带图件"]["disabled"] and not ui.widgets["研判时附带图件"]["value"]
    assert ui.widgets["兼容接口地址"]["disabled"]
    assert "synthetic-secret" not in json.dumps(ui.widgets, default=str)
    assert any("图片与参考库原件留在本机" in item for item in ui.messages)
