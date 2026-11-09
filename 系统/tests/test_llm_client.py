"""本地 HTTP 模拟服务验证请求、缓存与凭据不落盘。"""

from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from PIL import Image
import pytest

from reporting.llm_client import DEFAULT_LLM_CONFIG, LLMClientError, LLMSettings, complete_json, load_llm_settings, prepare_images


@pytest.fixture
def server():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append({"path": self.path, "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"model": "mock-model", "choices": [{"message": {"content": '{"ok":true}', "reasoning_content": "模拟思考"}}], "usage": {"total_tokens": 12}}).encode())
        def log_message(self, *_):
            pass
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", requests
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def test_configuration_priority_and_secret_repr(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"base_url": "https://configured.invalid/v1", "model": "profile-model", "api_key": "profile-secret", "upload_scope": "text_only"}))
    settings = load_llm_settings({"llm": {"decision_mode": "llm"}}, profile_path=profile, environ={"AIGC_LLM_MODEL": "env-model"})
    assert settings.configured and settings.model == "env-model"
    assert settings.options["upload_scope"] == "text_only"
    assert "profile-secret" not in repr(settings)
    assert "profile-secret" not in json.dumps(settings.public_status())
    assert not load_llm_settings({}, environ={}).configured


@pytest.mark.parametrize("options", [{"max_retries": 2}, {"timeout_seconds": 0}, {"judge_send_images": "false"}, {"upload_scope": "other"}, {"thinking_budget": -1}])
def test_invalid_configuration_is_inactive(options):
    assert load_llm_settings({"llm": options}, environ={}).configuration_error


def test_request_cache_and_no_key_in_audit(server, tmp_path):
    base, requests = server
    settings = LLMSettings(base, "requested-model", "local-test-secret")
    kwargs = dict(system_prompt="输出 JSON", user_payload={"items": []}, stage="judge", cache_dir=tmp_path)
    first = complete_json(settings, **kwargs)
    second = complete_json(settings, **kwargs)
    third = complete_json(settings, **kwargs, bypass_cache=True)
    assert len(requests) == 2
    assert not first["cache_hit"] and second["cache_hit"] and not third["cache_hit"]
    assert second["charged_usage"] == {}
    assert first["transmission_status"] == "response_received" and first["request_attempted"]
    assert second["transmission_status"] == "cached_response" and not second["request_attempted"]
    assert requests[0]["path"] == "/v1/chat/completions"
    assert requests[0]["body"]["enable_thinking"] is True
    assert "max_tokens" not in requests[0]["body"]
    assert first["response"]["choices"][0]["message"]["reasoning_content"] == "模拟思考"
    assert "local-test-secret" not in json.dumps(first)
    assert all("local-test-secret" not in path.read_text() for path in tmp_path.glob("*.json"))


def test_image_request_uses_json_object_and_audit_only_hash(server, tmp_path):
    base, requests = server
    path = tmp_path / "synthetic.jpg"
    Image.new("RGB", (20, 20), "white").save(path)
    options = deepcopy(DEFAULT_LLM_CONFIG)
    options["judge_send_images"] = True
    settings = LLMSettings(base, "requested-model", "local-test-secret", options)
    figures = [{"figure_id": "F01", "upload_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
    images = prepare_images(figures, settings, stage="judge")
    result = complete_json(settings, system_prompt="输出 JSON", user_payload={"items": []}, images=images, stage="judge", schema={"type": "object"})
    assert requests[0]["body"]["response_format"] == {"type": "json_object"}
    assert "data:image/jpeg;base64," not in json.dumps(result["request"])
    assert str(path) not in json.dumps(result["request"])
    options["upload_scope"] = "text_only"
    assert prepare_images(figures, settings, stage="writer") == []


@pytest.mark.parametrize("payload", [{"local_path": "secret"}, {"note": "C:\\private\\case.png"}, {"api_key": "hidden"}, {"note": "file:///tmp/private"}])
def test_path_and_credentials_rejected_before_network(payload):
    settings = LLMSettings("http://127.0.0.1:1/v1", "mock", "local-test-secret")
    with pytest.raises(LLMClientError, match="上传内容"):
        complete_json(settings, system_prompt="输出 JSON", user_payload=payload, stage="judge")


def test_payload_limit_and_hash_mismatch(tmp_path):
    options = deepcopy(DEFAULT_LLM_CONFIG)
    options["max_request_bytes"] = 10
    options["judge_send_images"] = True
    settings = LLMSettings("http://127.0.0.1:1/v1", "mock", "local-test-secret", options)
    with pytest.raises(LLMClientError) as error:
        complete_json(settings, system_prompt="输出 JSON", user_payload={}, stage="judge")
    assert error.value.code == "request_too_large"
    image = tmp_path / "synthetic.jpg"
    Image.new("RGB", (10, 10)).save(image)
    with pytest.raises(LLMClientError) as error:
        prepare_images([{"figure_id": "F01", "upload_path": str(image), "sha256": "0" * 64}], settings, stage="judge")
    assert error.value.code == "image_hash_mismatch"
