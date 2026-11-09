"""New synthetic reports only; all recovery calls are local and model-free."""
from copy import deepcopy
import hashlib
import json

import pytest

from reporting import analysis_pipeline as pipeline
from reporting import analysis_pdf
from reporting.analysis_recovery import recover_analysis_report
from reporting.llm_client import LLMSettings


def synthetic_report():
    return {"task": {"task_id": "synthetic-recovery", "status": "partial_success"},
            "input": {}, "detection": {}, "identity": {}, "source": {}}


def interrupted(tmp_path, monkeypatch, *, during="render"):
    report = synthetic_report()
    settings = LLMSettings()
    original_render = analysis_pdf.render_analysis_pdf
    original_writer = pipeline.run_writer
    def fail(*args, **kwargs):
        raise OSError("synthetic disk interruption")
    if during == "render":
        monkeypatch.setattr(analysis_pdf, "render_analysis_pdf", fail)
    else:
        monkeypatch.setattr(pipeline, "run_writer", fail)
    with pytest.raises(OSError, match="synthetic"):
        pipeline.generate_analysis_report(tmp_path, report, config={}, settings=settings)
    monkeypatch.setattr(analysis_pdf, "render_analysis_pdf", original_render)
    monkeypatch.setattr(pipeline, "run_writer", original_writer)
    return report, settings


@pytest.mark.parametrize("during", ["render", "writer"])
def test_explicit_render_recovery_never_reruns_judge_or_writer(tmp_path, monkeypatch, during):
    report, settings = interrupted(tmp_path, monkeypatch, during=during)
    before = {p.relative_to(tmp_path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in tmp_path.rglob("*") if p.is_file()}
    def forbidden(*args, **kwargs):
        pytest.fail("report recovery must not call judge or writer")
    monkeypatch.setattr(pipeline, "run_judge", forbidden)
    monkeypatch.setattr(pipeline, "run_writer", forbidden)
    with pytest.raises(ValueError, match="显式补报告"):
        pipeline.generate_analysis_report(tmp_path, report, config={}, settings=settings)
    recovered = recover_analysis_report(tmp_path, report, config={}, settings=settings)
    from pathlib import Path
    root = Path(recovered["task_dir"])
    result = recovered["result"]
    assert not pipeline.verify_analysis_artifacts(root)
    assert (root / result["pdf_file"]).read_bytes().startswith(b"%PDF-")
    assert result["final_decision"] == json.loads((tmp_path / "final_decision.json").read_text("utf-8"))
    assert result["recovery"]["model_calls"] == result["recovery"]["cloud_calls"] == 0
    assert result["recovery"]["source_stage"] == ("report_ready" if during == "render" else "decision")
    assert not (tmp_path / "analysis_report.json").exists()
    for name, digest in before.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("mutation", ["source", "config", "final", "receipt", "checkpoint", "missing", "stop"])
def test_recovery_refuses_changed_or_unverified_artifacts(tmp_path, monkeypatch, mutation):
    report, settings = interrupted(tmp_path, monkeypatch)
    config = {}
    if mutation == "source":
        report["task"]["status"] = "different"
    elif mutation == "config":
        config = {"identity": {"top_k": 7}}
    elif mutation == "final":
        path = tmp_path / "final_decision.json"
        data = json.loads(path.read_text("utf-8"))
        data["final_band"] = "obvious"
        path.write_text(json.dumps(data), encoding="utf-8")
    elif mutation == "receipt":
        (tmp_path / "stage_receipts/analysis.initial.report_ready.json").write_text("{}", encoding="utf-8")
    elif mutation == "checkpoint":
        path = tmp_path / "recovery_v4/report_ready.json"
        data = json.loads(path.read_text("utf-8"))
        data["payload"]["final_decision"]["final_band"] = "none"
        path.write_text(json.dumps(data), encoding="utf-8")
    elif mutation == "missing":
        (tmp_path / "analysis_input_manifest.json").unlink()
    else:
        (tmp_path / "STOP").write_text("explicit synthetic stop", encoding="utf-8")
    with pytest.raises(ValueError):
        recover_analysis_report(tmp_path, report, config=config, settings=settings)
    assert not (tmp_path / "report_recovery_attempts").exists()
    assert not (tmp_path / "rr_v4").exists()


def test_completed_original_can_be_recovered_without_overwriting_it(tmp_path, monkeypatch):
    report, settings = synthetic_report(), LLMSettings()
    original = pipeline.generate_analysis_report(tmp_path, report, config={}, settings=settings)
    original_pdf = (tmp_path / original["pdf_file"]).read_bytes()
    original_json = (tmp_path / "analysis_report.json").read_bytes()
    recovered = recover_analysis_report(tmp_path, report, config={}, settings=settings)
    assert recovered["result"]["final_decision"] == original["final_decision"]
    assert (tmp_path / original["pdf_file"]).read_bytes() == original_pdf
    assert (tmp_path / "analysis_report.json").read_bytes() == original_json
    assert not pipeline.verify_analysis_artifacts(tmp_path)


def test_report_ready_does_not_claim_pdf_was_completed(tmp_path, monkeypatch):
    interrupted(tmp_path, monkeypatch)
    assert (tmp_path / "stage_receipts/analysis.initial.report_ready.json").exists()
    assert not (tmp_path / "stage_receipts/analysis.initial.report.json").exists()
    assert not (tmp_path / "研判分析报告.pdf").exists()
