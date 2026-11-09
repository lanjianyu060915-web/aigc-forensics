"""Full task/report wiring with new pixels, empty libraries and fake services.

Run with -p v4_test_isolation. No inference, external request, shared library,
historical score source or old result is allowed by these fixtures.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading

from PIL import Image
import pytest

import orchestrator as backend
from contracts.task_contract_v4 import canonical_digest
from reporting import analysis_pdf, analysis_pipeline, llm_client, llm_service, pdf_report, report_builder
from reporting.final_decision_view import saved_decision_view


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


@pytest.fixture(autouse=True)
def no_external_state(tmp_path, monkeypatch, request):
    assert request.config.pluginmanager.hasplugin("v4_test_isolation"), "Run this audited suite with -p v4_test_isolation"
    monkeypatch.setenv("FSTS_DATA_ROOT", str(tmp_path / "new-user-data"))
    for key in list(os.environ):
        if key.startswith("AIGC_LLM_"):
            monkeypatch.delenv(key)
    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic end-to-end tests must not load models, spawn runners or call cloud")
    monkeypatch.setattr(backend, "_run_subprocess", forbidden)
    monkeypatch.setattr(llm_client.urllib.request, "build_opener", forbidden)
    monkeypatch.setattr(llm_service, "complete_json", forbidden)
    monkeypatch.setattr(backend, "analyze_visible_labels", lambda inputs: backend.visible_label_unavailable("synthetic OCR disabled"))


def setup_task(tmp_path, *, parallel=False, general=True, disabled=False):
    identity, source = tmp_path / "empty-identity", tmp_path / "empty-source"
    identity.mkdir()
    source.mkdir()
    query = tmp_path / "newly-constructed-query.png"
    image = Image.new("RGB", (48, 32), (17, 73, 151))
    for x in range(8, 24):
        for y in range(8, 24):
            image.putpixel((x, y), (239, 201, 47))
    image.save(query)
    config = {"schema_version": "synthetic-test-only", "usage": "test_only", "training_allowed": False,
        "runtime": {"tasks_root": str(tmp_path / "tasks"), "inbox_root": str(tmp_path / "inbox"),
                    "identity_cache": str(tmp_path / "new-cache"), "parallel_modules": parallel},
        "deepfake": {"enabled": not disabled, "timeout_seconds": 5},
        "general_aigc": {"enabled": general and not disabled, "timeout_seconds": 5,
                         "face_crops": {"enabled": False}},
        "identity": {"enabled": not disabled, "library": str(identity), "timeout_seconds": 5},
        "source": {"enabled": not disabled, "library": str(source), "timeout_seconds": 5},
        "frontend": {}, "pair_compare": {"enabled": False}, "provenance": {"enabled": False},
        "llm": {"decision_mode": "rule", "upload_scope": "text_only", "judge_send_images": False}}
    path = tmp_path / "synthetic-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return query, path, config


def services(*, corrupt=False, observed=None):
    def observe(name):
        if observed is not None:
            observed.append({"module": name, "environment": deepcopy(backend._TASK_ENVIRONMENT.get()),
                             "thread": threading.current_thread().name})
    def deepfake(*, query_path, **kwargs):
        observe("deepfake")
        result = backend._standard_deepfake_result(query_path, 0.001, None, status="success")
        result.update(score=float("nan") if corrupt else .125, is_deepfake=False, label="synthetic-low",
                      details={"band": "low"})
        return result
    def general(*, input_path, **kwargs):
        observe("general_aigc")
        return {"status": "success", "module": "general_aigc", "query_image": str(input_path),
                "score": .125, "logit": float("nan") if corrupt else -1.75, "band": "low",
                "band_label": "synthetic-low", "runtime_seconds": .001, "error_message": None,
                "face_branch": {"status": "no_face", "logit": None, "score": None, "band": None}}
    def forbidden(*args, **kwargs):
        pytest.fail("Empty reference libraries must not call retrieval services")
    return {"run_deepfake": deepfake, "run_general_aigc": general,
            "run_identity": forbidden, "run_source": forbidden}


def test_run_task_uses_saved_decision_in_all_reports_and_publishes_bound_receipts(tmp_path, monkeypatch):
    query, config_path, _ = setup_task(tmp_path)
    captured = {"analysis_pdf": [], "technical_pdf": [], "text": []}
    original_analysis = analysis_pdf.render_analysis_pdf
    original_technical = pdf_report.render_pdf_report
    original_text = report_builder.render_text_report
    def analysis_render(package, final, *args, **kwargs):
        captured["analysis_pdf"].append(deepcopy(final))
        return original_analysis(package, final, *args, **kwargs)
    def technical_render(report, *args, **kwargs):
        captured["technical_pdf"].append(deepcopy(report["analysis"]["final_decision"]))
        return original_technical(report, *args, **kwargs)
    def text_render(report):
        captured["text"].append(deepcopy(report["analysis"]["final_decision"]))
        return original_text(report)
    monkeypatch.setattr(analysis_pdf, "render_analysis_pdf", analysis_render)
    monkeypatch.setattr(pdf_report, "render_pdf_report", technical_render)
    monkeypatch.setattr(report_builder, "render_text_report", text_render)
    status = backend.run_task(query, config_path=config_path, selected_modules=["deepfake"], services=services())
    root = Path(status["task_dir"])
    assert status["status"] == status["reports"]["status"] == "success"
    final = read(root / "final_decision.json")
    assert final["final_band"] == "none" and final["source"] == "rule"
    for values in captured.values():
        assert values and all(value == final for value in values)
    checkpoint = read(root / "recovery_v4/decision.json")
    assert final["evidence_digest"] == canonical_digest(checkpoint["payload"]["decision_input"])
    files = status["reports"]["files"]
    report = read(root / files["json"])
    assert report["analysis"]["final_decision"] == final
    for kind in ("txt", "html"):
        content = (root / files[kind]).read_text("utf-8")
        assert final["rule_version"] in content and final["evidence_digest"] in content
        assert saved_decision_view(report)["label"] in content
    for kind in ("pdf", "analysis_pdf"):
        assert (root / files[kind]).read_bytes().startswith(b"%PDF-")
    assert analysis_pipeline.verify_analysis_artifacts(root) == []
    inference = read(root / "stage_receipts" / f"{status['task_id']}.{status['attempt_id']}.inference.json")
    assert inference["input_digest"] == hashlib.sha256(query.read_bytes()).hexdigest()
    assert inference["config_digest"] == status["config_digest"]
    assert inference["status"] == "completed"
    for receipt in [inference, *[read(root / "stage_receipts" / f"analysis.initial.{stage}.json") for stage in ("decision", "report_ready", "report")]]:
        for artifact in receipt["artifacts"]:
            assert hashlib.sha256((root / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]


def test_empty_library_failure_is_preserved_as_partial_success_without_retrieval_loading(tmp_path):
    query, config_path, _ = setup_task(tmp_path)
    status = backend.run_task(query, config_path=config_path, selected_modules=["deepfake", "identity"], services=services())
    root = Path(status["task_dir"])
    assert status["status"] == "partial_success"
    assert status["deepfake"]["status"] == "success" and status["identity"]["status"] == "error"
    assert status["identity"]["execution_mode"] == "preflight"
    assert status["reports"]["status"] == "success"
    assert read(root / "final_decision.json")["final_band"] == "none"
    assert read(root / "identity_result.json")["candidates"] == []


def test_nonfinite_fake_service_outputs_become_errors_not_zero_scores(tmp_path):
    query, config_path, _ = setup_task(tmp_path)
    status = backend.run_task(query, config_path=config_path, selected_modules=["deepfake"], services=services(corrupt=True))
    root = Path(status["task_dir"])
    detection = read(root / "detection_result.json")
    general = read(root / "general_aigc_result.json")
    assert "face_coverage_evidence" in detection
    assert detection["face_coverage_evidence"]["coverage"]["count_status"] == "unrecorded"
    assert detection["face_coverage_evidence"]["identity_coverage"]["processed_bound_count"] == 0
    assert status["deepfake"]["status"] == "error"
    assert detection["score"] is None and general["logit"] is None
    assert general["status"] == "error"
    assert read(root / "final_decision.json")["final_band"] == "undetermined"
    assert status["reports"]["status"] == "success"
    assert "NaN" not in (root / "detection_result.json").read_text("utf-8")


def test_general_failure_cannot_be_hidden_by_successful_auxiliary_branch(tmp_path):
    query, config_path, _ = setup_task(tmp_path)
    fake = services()
    fake["run_general_aigc"] = lambda **kwargs: {
        "status": "error", "score": None, "logit": None, "band": None,
        "error_message": "synthetic missing decoder", "runtime_seconds": 0.001}
    status = backend.run_task(query, config_path=config_path, selected_modules=["deepfake"], services=fake)
    assert status["status"] == "partial_success"
    assert status["deepfake"]["status"] == "success"  # preserved legacy auxiliary status
    assert status["inference_completion"]["general_aigc"]["status"] == "error"
    assert any("synthetic missing decoder" in error for error in status["errors"])
    assert status["reports"]["status"] == "success"  # report delivery is a separate fact
    root = Path(status["task_dir"])
    # The engineering fix changes completion, not the pre-existing fallback band.
    assert read(root / "final_decision.json")["final_band"] == "none"
    receipt = read(root / "stage_receipts" / f"{status['task_id']}.{status['attempt_id']}.inference.json")
    assert any(item["module"] == "general_aigc" and item["module_status"] == "error" for item in receipt["artifacts"])


def test_disabled_modules_never_invoke_services_or_model_runners(tmp_path):
    query, config_path, _ = setup_task(tmp_path, disabled=True, parallel=True)
    observed = []
    status = backend.run_task(query, config_path=config_path, services=services(observed=observed))
    assert observed == []
    assert all(status[name]["execution_mode"] == "disabled" for name in backend.MODULE_ORDER)


def test_parallel_module_and_nested_service_threads_keep_isolated_context(tmp_path, monkeypatch):
    query, config_path, _ = setup_task(tmp_path, parallel=True)
    from contracts import execution_profile_v4
    # This tests context transport only. Independent admission is tested in the
    # admission suite; this synthetic config has no data-admission profile.
    expected = {"SYNTHETIC_ONLY": "context-marker"}
    monkeypatch.setattr(execution_profile_v4, "is_independent_profile", lambda config: True)
    monkeypatch.setattr(execution_profile_v4, "isolated_environment", lambda config: dict(expected))
    observed = []
    backend.run_task(query, config_path=config_path, selected_modules=["deepfake", "source"], services=services(observed=observed))
    assert {row["module"] for row in observed} == {"deepfake", "general_aigc"}
    assert all(row["environment"] == expected for row in observed)
    assert all(row["thread"] == "fsts-service-call" for row in observed)
    assert backend._TASK_ENVIRONMENT.get() is None


def test_report_render_interruption_retains_decision_but_does_not_claim_complete_report(tmp_path, monkeypatch):
    query, config_path, config = setup_task(tmp_path)
    original_render = analysis_pdf.render_analysis_pdf
    def fail_render(*args, **kwargs):
        raise OSError("new synthetic render failure")
    monkeypatch.setattr(analysis_pdf, "render_analysis_pdf", fail_render)
    status = backend.run_task(query, config_path=config_path, selected_modules=["deepfake"], services=services())
    root = Path(status["task_dir"])
    assert (root / "final_decision.json").is_file()
    assert (root / "stage_receipts/analysis.initial.report_ready.json").is_file()
    assert not (root / "stage_receipts/analysis.initial.report.json").exists()
    assert not (root / "研判分析报告.pdf").exists()
    assert status["status"] == "partial_success"
    assert status["reports"]["status"] != "success"
    assert status["reports"]["files"]["json"]
    before = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in root.rglob("*") if path.is_file()}
    monkeypatch.setattr(analysis_pdf, "render_analysis_pdf", original_render)
    def forbidden(*args, **kwargs):
        pytest.fail("Explicit report recovery must reuse saved decision without judge/writer calls")
    monkeypatch.setattr(analysis_pipeline, "run_judge", forbidden)
    monkeypatch.setattr(analysis_pipeline, "run_writer", forbidden)
    from reporting.analysis_recovery import recover_analysis_report
    settings = llm_client.load_llm_settings(config, environ={})
    recovered = recover_analysis_report(root, read(root / "analysis_source_report.json"), config=config, settings=settings)
    assert recovered["result"]["final_decision"] == read(root / "final_decision.json")
    assert recovered["result"]["recovery"]["model_calls"] == recovered["result"]["recovery"]["cloud_calls"] == 0
    assert analysis_pipeline.verify_analysis_artifacts(recovered["task_dir"]) == []
    for name, sha in before.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
