"""研判和撰写分别调用、校验与回退，并保存不含密钥的调用记录。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any
import urllib.parse

from .llm_client import LLMClientError, LLMSettings, complete_json, now_iso, prepare_images, write_json
from .llm_validation import BANDS, JUDGE_SCHEMA, WRITER_SCHEMA, PROCESS_FIELDS, validate_judge, validate_writer
from .evidence_package import to_model_view


PROMPT_ROOT = Path(__file__).with_name("prompts")


def _prompt(stage):
    path = PROMPT_ROOT / (stage + "_v1.md")
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    schema = JUDGE_SCHEMA if stage == "judge" else WRITER_SCHEMA
    return {"text": text + "\n\n输出 JSON 结构：\n" + json.dumps(schema, ensure_ascii=False),
            "version": text.splitlines()[0].split()[-1], "sha256": hashlib.sha256(raw).hexdigest(), "schema": schema}


def _view_with_uploads(evidence_view, images):
    view = deepcopy(dict(evidence_view))
    uploaded = {item["figure_id"] for item in images}
    for figure in view.get("figures", []):
        if isinstance(figure, dict):
            figure["uploaded"] = figure.get("figure_id", figure.get("id")) in uploaded
    view["uploaded_figure_ids"] = sorted(uploaded)
    return view


def _save_audit(root, stage, attempts, *, skipped_reason=None):
    names = {kind: f"llm_{stage}_{kind}.json" for kind in ("request", "response", "check")}
    write_json(root / names["request"], {"attempts": [item.get("trace", {}).get("request") for item in attempts],
        "metadata": [{key: value for key, value in item.get("trace", {}).items() if key not in ("request", "response")} for item in attempts],
        "skipped_reason": skipped_reason})
    write_json(root / names["response"], {"attempts": [item.get("trace", {}).get("response", {"error_code": item.get("error_code"), "reason": item.get("reason")}) for item in attempts], "skipped_reason": skipped_reason})
    write_json(root / names["check"], {"attempts": [item.get("check", {"valid": False, "errors": [item.get("reason", "调用失败")]}) for item in attempts], "skipped_reason": skipped_reason})
    return list(names.values())


def _usage(attempts):
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0, "image_tokens": 0}
    for attempt in attempts:
        usage = attempt.get("trace", {}).get("charged_usage", {})
        if not isinstance(usage, Mapping):
            continue
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key, 0)
            if isinstance(value, int) and not isinstance(value, bool):
                total[key] += value
        for container, key in (("completion_tokens_details", "reasoning_tokens"), ("prompt_tokens_details", "image_tokens")):
            details = usage.get(container)
            value = details.get(key) if isinstance(details, Mapping) else None
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                total[key] += value
    return total


def _invoke(stage, evidence_view, *, settings, root, figures, final_decision=None, bypass_cache=False):
    evidence_view = to_model_view(evidence_view)
    prompt = _prompt(stage)
    attempts = []
    try:
        images = prepare_images(figures, settings, stage=stage)
    except LLMClientError as error:
        attempts.append({"error_code": error.code, "reason": str(error)})
        return None, attempts, prompt, _image_audit([], [], attempts)
    planned = [image["figure_id"] for image in images]
    feedback = None
    result = None
    for _ in range(settings.options["max_retries"] + 1):
        record = {}
        try:
            while True:
                view = _view_with_uploads(evidence_view, images)
                payload = {"evidence_package": view, "band_labels": settings.options["band_labels"]}
                if final_decision is not None:
                    payload["final_decision"] = {key: deepcopy(final_decision[key]) for key in ("final_band", "score", "source", "reason", "judgment") if key in final_decision}
                try:
                    trace = complete_json(settings, system_prompt=prompt["text"], user_payload=payload, images=images,
                                          stage=stage, schema=prompt["schema"], retry_feedback=feedback,
                                          cache_dir=root / "llm_cache", bypass_cache=bypass_cache)
                    break
                except LLMClientError as error:
                    if error.code != "request_too_large" or not images:
                        raise
                    # 大小检查发生在网络请求前，按既定优先级从末尾减图，不消耗重试次数。
                    images.pop()
            record["trace"] = trace
            response = trace["response"]
            try:
                content = response["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError()
                result = json.loads(content, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            except (ValueError, KeyError, TypeError, IndexError):
                record["check"] = {"valid": False, "errors": ["返回正文不是完整 JSON 对象"]}
                result = None
            else:
                ids = [item["figure_id"] for item in images]
                record["check"] = validate_judge(result, view, uploaded_figure_ids=ids) if stage == "judge" else validate_writer(result, view, final_decision, uploaded_figure_ids=ids, band_labels=settings.options["band_labels"])
            attempts.append(record)
            if record["check"]["valid"]:
                return record["check"].get("normalized", result), attempts, prompt, _image_audit(planned, images, attempts)
            feedback = record["check"]["errors"]
        except LLMClientError as error:
            record.update(error_code=error.code, reason=str(error), trace=error.trace or {})
            attempts.append(record)
            result = None
            feedback = [str(error)]
            if error.code in ("authentication_failed", "not_configured", "unsafe_payload", "request_too_large", "redirect_refused"):
                break
    return result, attempts, prompt, _image_audit(planned, images, attempts)


def _image_audit(planned, images, attempts):
    """图区分计划、请求内容、发送尝试和收到响应；网络错误不冒称送达。"""
    attempted, confirmed, unknown = set(), set(), set()
    statuses = []
    for attempt in attempts:
        trace = attempt.get("trace", {})
        status = trace.get("transmission_status", "not_sent")
        statuses.append(status)
        ids = set(trace.get("included_request_figure_ids", []))
        if trace.get("request_attempted"):
            attempted.update(ids)
        if status in ("response_received", "cached_response"):
            confirmed.update(ids)
        elif status == "delivery_unknown":
            unknown.update(ids)
    status = statuses[-1] if statuses else "not_sent"
    return {"planned_figure_ids": list(planned), "included_request_figure_ids": [image["figure_id"] for image in images],
            "attempted_figure_ids": sorted(attempted), "uploaded_figure_ids": sorted(confirmed),
            "delivery_unknown_figure_ids": sorted(unknown - confirmed), "transmission_status": status,
            "transmission_note": {"not_sent": "未发送请求", "delivery_unknown": "已尝试发送，送达未知",
                                  "response_received": "接口已有返回", "cached_response": "复用已保存响应，本次未新发送"}.get(status, status),
            "omitted_for_request_size": [item for item in planned if item not in {image["figure_id"] for image in images}]}


def _reason(attempts):
    if not attempts:
        return "大模型未参与"
    last = attempts[-1]
    return last.get("reason") or "；".join(last.get("check", {}).get("errors", ["输出未通过校验"]))


def run_judge(evidence_view: Mapping[str, Any], *, settings: LLMSettings, task_dir: str | Path,
              figures=(), bypass_cache: bool = False) -> dict[str, Any]:
    root = Path(task_dir)
    rule = evidence_view.get("rule_band")
    rule = rule if rule in BANDS else "undetermined"
    decision = {"final_band": rule, "score": None, "source": "rule", "rule_band": rule,
                "reason": "规则定档", "requested_model": settings.model,
                "actual_model": None, "host": urllib.parse.urlsplit(settings.base_url).hostname,
                "time": now_iso(), "judgment": None, **_image_audit([], [], [])}
    skipped = None
    if rule == "undetermined":
        skipped = "检测分支没有可用档位或前置步骤失败，无法完成判断"
    elif settings.options["decision_mode"] == "rule":
        skipped = "配置为规则定档"
    elif not settings.configured:
        decision["source"] = "rule_fallback"
        skipped = settings.configuration_error or "大模型接口未完整配置"
    if skipped:
        decision["reason"] = skipped
        decision["audit_files"] = _save_audit(root, "judge", [], skipped_reason=skipped)
        decision["usage"] = _usage([])
    else:
        result, attempts, prompt, image_audit = _invoke("judge", evidence_view, settings=settings, root=root, figures=figures, bypass_cache=bypass_cache)
        successful = bool(attempts and attempts[-1].get("check", {}).get("valid"))
        trace = attempts[-1].get("trace", {}) if attempts else {}
        decision.update(prompt_version=prompt["version"], prompt_sha256=prompt["sha256"],
                        actual_model=trace.get("response", {}).get("model"), request_sha256=trace.get("request_sha256"),
                        **image_audit, usage=_usage(attempts),
                        elapsed_seconds=sum(attempt.get("trace", {}).get("elapsed_seconds", 0) for attempt in attempts),
                        audit_files=_save_audit(root, "judge", attempts))
        if successful:
            decision.update(final_band=result["final_band"], score=result["score"], source="llm", judgment=result,
                            reason=result["band_reason"])
        else:
            decision.update(source="rule_fallback", reason=_reason(attempts))
    decision["audit_files"].append("final_decision.json")
    write_json(root / "final_decision.json", decision)
    return decision


def _validated_parts(result, invalid):
    content = deepcopy(result) if isinstance(result, dict) else {}
    if "all" in invalid:
        return {}
    for key in invalid:
        if key in PROCESS_FIELDS:
            if isinstance(content.get("process"), dict):
                content["process"].pop(key, None)
        else:
            content.pop(key, None)
    return content


def run_writer(evidence_view: Mapping[str, Any], final_decision: Mapping[str, Any], *, settings: LLMSettings,
               task_dir: str | Path, figures=(), bypass_cache: bool = False) -> dict[str, Any]:
    root = Path(task_dir)
    if not settings.configured or final_decision.get("source") == "rule_fallback":
        reason = final_decision.get("reason") if final_decision.get("source") == "rule_fallback" else settings.configuration_error or "大模型接口未完整配置"
        output = {"status": "fallback", "content": {}, "invalid_sections": ["all"], "reason": reason,
                  "audit_files": _save_audit(root, "writer", [], skipped_reason=reason), "usage": _usage([]),
                  **_image_audit([], [], [])}
    else:
        result, attempts, prompt, image_audit = _invoke("writer", evidence_view, settings=settings, root=root, figures=figures, final_decision=final_decision, bypass_cache=bypass_cache)
        last = attempts[-1] if attempts else {}
        check = last.get("check", {})
        successful = check.get("valid", False)
        invalid = [] if successful else check.get("invalid_sections", ["all"])
        content = _validated_parts(check.get("normalized", result), invalid)
        trace = last.get("trace", {})
        output = {"status": "success" if successful else "partial" if content else "fallback", "content": content,
                  "invalid_sections": invalid, "reason": "撰写通过校验" if successful else _reason(attempts),
                  "prompt_version": prompt["version"], "prompt_sha256": prompt["sha256"],
                  "requested_model": settings.model, "actual_model": trace.get("response", {}).get("model"),
                  "host": trace.get("host"), "time": now_iso(), "request_sha256": trace.get("request_sha256"),
                  **image_audit, "usage": _usage(attempts),
                  "elapsed_seconds": sum(attempt.get("trace", {}).get("elapsed_seconds", 0) for attempt in attempts),
                  "audit_files": _save_audit(root, "writer", attempts)}
    output["audit_files"].append("llm_writer_result.json")
    write_json(root / "llm_writer_result.json", output)
    return output
