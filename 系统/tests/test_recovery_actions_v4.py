"""Independent marker artifacts and mocked network/model dispatch only.

Run with -p v4_test_isolation. No existing task, media, score, library, model or
real cloud endpoint is opened; frozen prompt/validator source hashes are code.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from artifact_io import atomic_write_json
from contracts.task_contract_v4 import canonical_digest, normalize_final_decision, rule_source_hashes
from reporting.analysis_recovery import report_config_digest, save_report_checkpoint
from reporting.llm_client import LLMSettings
from reporting import recovery_actions_v4 as actions


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_task(root, *, initial_calls=1, writer_status="fallback", complete_report=False, initial_transmission="response_received"):
    root.mkdir(parents=True, exist_ok=True)
    settings = LLMSettings(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="synthetic-model",
                           api_key="synthetic-secret-not-a-real-key")
    settings = replace(settings, options={**settings.options, "upload_scope": "text_only"})
    config = {"synthetic": True}
    report = {"task": {"task_id": "synthetic-recovery-original", "status": "partial_success"}, "input": {}}
    view = {"rule_band": "review", "items": [], "figures": []}
    final = normalize_final_decision({"final_band": "review", "score": None, "source": "rule", "rule_band": "review", "reason": "synthetic frozen reason"},
        view, requested_mode="rule", source_hashes=rule_source_hashes())
    package = {"rule_band": "review", "items": [], "figures": []}
    prompt_sha = digest(Path(actions.__file__).with_name("prompts") / "writer_v1.md")
    writer = {"status": writer_status, "content": {}, "prompt_sha256": prompt_sha,
              "audit_files": ["llm_writer_request.json", "llm_writer_response.json", "llm_writer_check.json", "llm_writer_result.json"]}
    values = {"final_decision.json": final, "analysis_source_report.json": report,
              "llm_writer_request.json": {"metadata": [{"request_attempted": True, "transmission_status": initial_transmission} for _ in range(initial_calls)], "attempts": [{} for _ in range(initial_calls)]},
              "llm_writer_response.json": {"synthetic": True}, "llm_writer_check.json": {"synthetic": True}, "llm_writer_result.json": writer}
    for name, value in values.items():
        atomic_write_json(root / name, value, overwrite=False)
    source_sha, config_sha = canonical_digest(report), report_config_digest(config, settings)
    save_report_checkpoint(root, stage="report_ready", source_digest=source_sha, config_digest=config_sha,
        payload={"package": package, "decision_input": view, "final_decision": final, "writer": writer, "content": {}}, artifact_names=list(values))
    if complete_report:
        atomic_write_json(root / "evidence_package.json", package, overwrite=False)
        atomic_write_json(root / "analysis_report.json", {"status": "success", "source_report_sha256": source_sha,
            "final_decision": final, "evidence_package": package, "pdf_file": "研判分析报告.pdf"}, overwrite=False)
        (root / "研判分析报告.pdf").write_bytes(b"%PDF synthetic marker, never rendered or decoded")
        names = [*values, "evidence_package.json", "analysis_report.json", "研判分析报告.pdf"]
        atomic_write_json(root / "analysis_artifact_manifest.json", {"artifacts": [{"path": name, "sha256": digest(root/name)} for name in names]}, overwrite=False)
    return report, config, settings, prompt_sha


def original_hashes(root):
    # The root is a freshly constructed pytest fixture, never an external task.
    return {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()}


def assert_originals(root, before):
    assert all(digest(root / name) == value for name, value in before.items())


def call_writer(root, fixture, **kwargs):
    report, config, settings, prompt = fixture
    return actions.retry_report_writer(root, report, config=config, settings=settings,
        expected_writer_prompt_sha256=prompt, authorized=True, cloud_authorized=True, **kwargs)


def fake_writer(monkeypatch, *, statuses=("fallback",), callback=None, transmission="response_received"):
    calls = []
    def write(view, final, *, settings, task_dir, figures, bypass_cache):
        calls.append((deepcopy(view), deepcopy(final)))
        assert settings.options["max_retries"] == 0 and settings.options["upload_scope"] == "text_only"
        assert figures == () and bypass_cache is True
        if callback:
            callback()
        status = statuses[min(len(calls)-1, len(statuses)-1)]
        result = {"status": status, "content": {}, "audit_files": ["llm_writer_request.json", "llm_writer_response.json", "llm_writer_check.json", "llm_writer_result.json"]}
        values = {"llm_writer_request.json": {"metadata": [{"request_attempted": True, "transmission_status": transmission}], "attempts": [{}]},
                  "llm_writer_response.json": {"synthetic": True}, "llm_writer_check.json": {"synthetic": True}, "llm_writer_result.json": result}
        for name, value in values.items():
            atomic_write_json(Path(task_dir)/name, value, overwrite=False)
        return result
    monkeypatch.setattr(actions, "run_writer", write)
    from reporting import llm_service, analysis_pdf
    monkeypatch.setattr(llm_service, "run_judge", lambda *a, **k: pytest.fail("judge must never run"))
    monkeypatch.setattr(analysis_pdf, "render_analysis_pdf", lambda *a, **k: pytest.fail("writer/receipt actions must not render"))
    return calls


def test_receipt_only_verifies_and_appends_without_models_cloud_or_render(tmp_path, monkeypatch):
    report, config, settings, _ = fixture_task(tmp_path, complete_report=True)
    calls = fake_writer(monkeypatch)
    before = original_hashes(tmp_path)
    result = actions.write_recovery_receipt(tmp_path, report, config=config, settings=settings, authorized=True)
    assert result["status"] == "completed" and result["model_calls"] == result["cloud_calls"] == result["pdf_renders"] == 0
    assert not calls and not result["independent_sample"]
    assert_originals(tmp_path, before)
    with pytest.raises(actions.RecoveryActionError, match="receipt_already"):
        actions.write_recovery_receipt(tmp_path, report, config=config, settings=settings, authorized=True)


@pytest.mark.parametrize("mutation", ["missing_manifest", "missing_checkpoint_receipt", "changed_final", "changed_package", "bad_json", "stop", "cancel"])
def test_receipt_does_not_invent_success_without_verified_artifacts(tmp_path, mutation):
    report, config, settings, _ = fixture_task(tmp_path, complete_report=True)
    if mutation == "missing_manifest":
        (tmp_path / "analysis_artifact_manifest.json").unlink()
    elif mutation == "missing_checkpoint_receipt":
        (tmp_path / "stage_receipts/analysis.initial.report_ready.json").unlink()
    elif mutation == "changed_final":
        (tmp_path / "final_decision.json").write_text('{}', encoding="utf-8")
    elif mutation == "changed_package":
        (tmp_path / "evidence_package.json").write_text('{}', encoding="utf-8")
    elif mutation == "bad_json":
        (tmp_path / "analysis_report.json").write_text('{', encoding="utf-8")
    else:
        (tmp_path / ("STOP" if mutation == "stop" else "CANCEL_SHUTDOWN")).write_text("synthetic stop", encoding="utf-8")
    with pytest.raises(ValueError):
        actions.write_recovery_receipt(tmp_path, report, config=config, settings=settings, authorized=True)
    assert not (tmp_path / "recovery_receipts_v4").exists()


def test_two_claims_bound_retries_and_preserve_frozen_decision(tmp_path, monkeypatch):
    fixture = fixture_task(tmp_path)
    before = original_hashes(tmp_path)
    calls = fake_writer(monkeypatch)
    first = call_writer(tmp_path, fixture)
    second = call_writer(tmp_path, fixture)
    assert first["task_dir"] != second["task_dir"]
    for attempt in (first, second):
        assert attempt["result"]["cloud_calls"] == 1 and attempt["result"]["render_ready"]
        assert attempt["result"]["judge_calls"] == attempt["result"]["model_calls"] == attempt["result"]["pdf_renders"] == 0
        assert (Path(attempt["task_dir"]) / "stage_receipts/analysis.initial.report_ready.json").is_file()
        assert (Path(attempt["task_dir"]) / "final_decision.json").read_bytes() == (tmp_path / "final_decision.json").read_bytes()
    with pytest.raises(actions.RecoveryActionError, match="budget_exhausted"):
        call_writer(tmp_path, fixture)
    assert len(calls) == 2
    assert_originals(tmp_path, before)


def test_original_internal_retry_consumes_one_of_two_total_retries(tmp_path, monkeypatch):
    fixture = fixture_task(tmp_path, initial_calls=2)
    calls = fake_writer(monkeypatch)
    call_writer(tmp_path, fixture)
    with pytest.raises(actions.RecoveryActionError, match="budget_exhausted"):
        call_writer(tmp_path, fixture)
    assert len(calls) == 1


def test_success_stops_further_retry_without_selecting_another_answer(tmp_path, monkeypatch):
    fixture = fixture_task(tmp_path)
    calls = fake_writer(monkeypatch, statuses=("success",))
    result = call_writer(tmp_path, fixture)
    assert result["result"]["status"] == "completed"
    with pytest.raises(actions.RecoveryActionError, match="already_succeeded"):
        call_writer(tmp_path, fixture)
    assert len(calls) == 1


def test_reentrant_attempt_is_blocked_by_pending_immutable_claim(tmp_path, monkeypatch):
    fixture = fixture_task(tmp_path)
    def concurrent():
        with pytest.raises(actions.RecoveryActionError, match="pending_or_delivery_unknown"):
            call_writer(tmp_path, fixture)
    calls = fake_writer(monkeypatch, callback=concurrent)
    call_writer(tmp_path, fixture)
    assert len(calls) == 1


def test_stop_arriving_during_writer_is_retained_and_blocks_followup(tmp_path, monkeypatch):
    fixture = fixture_task(tmp_path)
    calls = fake_writer(monkeypatch, callback=lambda: (tmp_path / "STOP").write_text("synthetic stop", encoding="utf-8"))
    result = call_writer(tmp_path, fixture)
    assert result["result"]["status"] == "stopped" and result["result"]["cloud_calls"] == 1
    assert (tmp_path / "STOP").exists() and len(calls) == 1
    with pytest.raises(actions.RecoveryActionError, match="stopped"):
        call_writer(tmp_path, fixture)


def test_receipt_disk_failure_never_creates_completion_marker(tmp_path, monkeypatch):
    report, config, settings, _ = fixture_task(tmp_path, complete_report=True)
    before = original_hashes(tmp_path)
    def fail(*a, **k):
        raise OSError("synthetic full disk")
    monkeypatch.setattr(actions, "write_stage_receipt", fail)
    with pytest.raises(OSError):
        actions.write_recovery_receipt(tmp_path, report, config=config, settings=settings, authorized=True)
    assert not (tmp_path / "recovery_receipts_v4/completed.json").exists()
    assert_originals(tmp_path, before)


def test_writer_outcome_disk_failure_keeps_claim_and_blocks_unknown_retry(tmp_path, monkeypatch):
    fixture = fixture_task(tmp_path)
    calls = fake_writer(monkeypatch)
    original_write = actions.atomic_write_json
    def fail(path, *a, **k):
        if Path(path).name == "recovery_result.json":
            raise OSError("synthetic full disk")
        return original_write(path, *a, **k)
    monkeypatch.setattr(actions, "atomic_write_json", fail)
    with pytest.raises(OSError):
        call_writer(tmp_path, fixture)
    assert (tmp_path / "writer_retry_claims_v4/slot-1.json").is_file()
    with pytest.raises(actions.RecoveryActionError, match="pending_or_delivery_unknown"):
        call_writer(tmp_path, fixture)
    assert len(calls) == 1


def test_writer_attempt_can_be_rendered_later_without_any_further_cloud(tmp_path, monkeypatch):
    from reporting import analysis_pdf, analysis_pipeline
    from reporting.analysis_recovery import recover_analysis_report
    render = analysis_pdf.render_analysis_pdf
    fixture = fixture_task(tmp_path)
    calls = fake_writer(monkeypatch, statuses=("success",))
    retried = call_writer(tmp_path, fixture)
    assert not (Path(retried["task_dir"]) / "研判分析报告.pdf").exists()
    monkeypatch.setattr(analysis_pdf, "render_analysis_pdf", render)
    report, config, settings, _ = fixture
    recovered = recover_analysis_report(retried["task_dir"], report, config=config, settings=settings)
    assert len(calls) == 1 and recovered["result"]["recovery"]["cloud_calls"] == 0
    assert not analysis_pipeline.verify_analysis_artifacts(recovered["task_dir"])


def test_receipt_repairs_actual_synthetic_pipeline_receipt_interruption(tmp_path):
    from reporting import analysis_pipeline
    report = {"task": {"task_id": "synthetic-receipt", "status": "partial_success"}, "input": {}, "detection": {}, "identity": {}, "source": {}}
    settings = LLMSettings()
    analysis_pipeline.generate_analysis_report(tmp_path, report, config={}, settings=settings)
    # Only this freshly constructed receipt is removed to inject interruption.
    (tmp_path / "stage_receipts/analysis.initial.report.json").unlink()
    result = actions.write_recovery_receipt(tmp_path, report, config={}, settings=settings, authorized=True)
    assert result["status"] == "completed" and result["pdf_renders"] == 0


def test_uncertain_request_failure_is_recorded_without_secret_and_not_retried(tmp_path, monkeypatch):
    fixture = fixture_task(tmp_path)
    def fail(*a, **kw):
        raise RuntimeError("synthetic-secret-not-a-real-key")
    monkeypatch.setattr(actions, "run_writer", fail)
    result = call_writer(tmp_path, fixture)
    assert result["result"]["cloud_calls"] is None and result["result"]["status"] == "failed"
    assert "synthetic-secret" not in json.dumps(result)
    with pytest.raises(actions.RecoveryActionError, match="delivery_unknown"):
        call_writer(tmp_path, fixture)


def test_caught_timeout_with_delivery_unknown_also_blocks_another_claim(tmp_path, monkeypatch):
    fixture = fixture_task(tmp_path)
    calls = fake_writer(monkeypatch, transmission="delivery_unknown")
    result = call_writer(tmp_path, fixture)
    assert result["result"]["cloud_calls"] == 1 and result["result"]["delivery_unknown"] is True
    with pytest.raises(actions.RecoveryActionError, match="delivery_unknown"):
        call_writer(tmp_path, fixture)
    assert len(calls) == 1


def test_original_delivery_unknown_cannot_be_treated_as_safe_retry(tmp_path, monkeypatch):
    fixture = fixture_task(tmp_path, initial_transmission="delivery_unknown")
    calls = fake_writer(monkeypatch)
    with pytest.raises(actions.RecoveryActionError, match="original_writer_delivery_unknown"):
        call_writer(tmp_path, fixture)
    assert not calls and not (tmp_path / "writer_retry_claims_v4").exists()


@pytest.mark.parametrize("case", ["independent", "unknown_profile", "images", "endpoint", "prompt", "unauthorized", "unknown_audit"])
def test_cloud_recovery_rejects_unapproved_paths_before_claim(tmp_path, monkeypatch, case):
    report, config, settings, prompt = fixture_task(tmp_path)
    calls = fake_writer(monkeypatch)
    if case == "independent":
        config["execution_profile"] = {"schema_version": "execution_profile.independent.v1"}
    elif case == "unknown_profile":
        config["execution_profile"] = {"schema_version": "unknown"}
    elif case == "images":
        settings = replace(settings, options={**settings.options, "upload_scope": "full"})
    elif case == "endpoint":
        settings = replace(settings, base_url="https://different.invalid")
    elif case == "prompt":
        prompt = "0" * 64
    elif case == "unknown_audit":
        # A damaged audit cannot be treated as zero earlier requests.
        (tmp_path / "llm_writer_request.json").write_text('{"metadata": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        actions.retry_report_writer(tmp_path, report, config=config, settings=settings, expected_writer_prompt_sha256=prompt,
                                    authorized=case != "unauthorized", cloud_authorized=True)
    assert not calls and not (tmp_path / "writer_retry_claims_v4").exists()


def test_rerun_uses_current_gate_and_dispatches_one_new_task(tmp_path, monkeypatch):
    import orchestrator
    from contracts import execution_profile_v4 as profile
    old, new_parent = tmp_path / "old", tmp_path / "new"
    old.mkdir(); new_parent.mkdir()
    digest_value = hashlib.sha256(b"independent synthetic query marker").hexdigest()
    atomic_write_json(old / "execution_binding.json", {"usage": "independent_model_selection", "input_digest": digest_value, "task_id": "old-synthetic"})
    config = {"execution_profile": {"schema_version": "execution_profile.independent.v1", "independent_root": str(tmp_path)}, "runtime": {"tasks_root": str(new_parent)}}
    calls = []
    monkeypatch.setattr(orchestrator, "load_config", lambda path: (config, path))
    monkeypatch.setattr(profile, "trusted_policy_authority", lambda: object())
    monkeypatch.setattr(profile, "audit_execution_profile", lambda *a, **k: {"config_digest": "a" * 64, "policy_binding": {"version": 17}})
    def admit(*a, **k):
        calls.append("current_admission")
        return SimpleNamespace(sha256=digest_value)
    monkeypatch.setattr(profile, "admit_query_snapshot", admit)
    def run(*a, **kw):
        calls.append("new_task")
        assert kw["services"] is None
        assert kw["config_path"].is_file() and kw["config_path"] != tmp_path / "config.json"
        assert json.loads(kw["config_path"].read_text("utf-8")) == config
        path = new_parent / "fresh-task"
        path.mkdir()
        return {"task_dir": str(path), "status": "success"}
    monkeypatch.setattr(orchestrator, "run_task", run)
    result = actions.rerun_as_new_task(old, tmp_path / "query.marker", config_path=tmp_path / "config.json", authorized=True)
    assert calls == ["current_admission", "new_task"]
    assert result["recovery"]["batch_resume"] is False and result["recovery"]["independent_sample"] is False
    assert (Path(result["task_dir"]) / "rerun_lineage_v4.json").is_file()


@pytest.mark.parametrize("usage", ["test_only", "unrecorded"])
def test_old_team_or_unrecorded_batch_cannot_rerun(tmp_path, usage):
    atomic_write_json(tmp_path / "execution_binding.json", {"usage": usage})
    with pytest.raises(actions.RecoveryActionError, match="batch_rerun_forbidden"):
        actions.rerun_as_new_task(tmp_path, tmp_path / "never-opened", config_path=tmp_path / "never-opened.json", authorized=True)
