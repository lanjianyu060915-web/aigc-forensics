"""Immutable report checkpoints and explicit, local-only report recovery.

Recovery writes a new attempt directory. It never reclassifies evidence, calls
models/cloud, changes a STOP marker, or replaces an original task artifact.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import uuid

from artifact_io import atomic_write_bytes, atomic_write_json, write_stage_receipt
from contracts.task_contract_v4 import canonical_digest, validate_saved_decision
from product_info import PRODUCT_NAME
from .evidence_figures import local_file


def report_config_digest(config: Mapping, settings) -> str:
    # Credentials are neither stored nor incorporated. Endpoint/model/options
    # bind report semantics; model and preprocessing config binds inference.
    def clean(value):
        if isinstance(value, Mapping):
            return {str(k): clean(v) for k, v in value.items()
                    if not any(word in str(k).casefold() for word in ("api_key", "password", "token", "secret"))}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value
    return canonical_digest({"config": clean(config), "endpoint": settings.base_url,
                             "model": settings.model, "options": settings.options})


def save_report_checkpoint(root: Path, *, stage: str, payload: Mapping,
                           source_digest: str, config_digest: str,
                           artifact_names: list[str]) -> Path:
    """Publish only after all enumerated artifacts are present and readable."""
    if stage not in {"decision", "report_ready"}:
        raise ValueError("unsupported report checkpoint")
    records = []
    for name in dict.fromkeys(artifact_names):
        path = local_file(root, name)
        if path is None:
            raise ValueError("checkpoint artifact missing or unsafe: " + name)
        records.append({"path": path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    target = root / "recovery_v4" / (stage + ".json")
    value = {"schema_version": "4.0", "stage": stage,
             "source_report_sha256": source_digest, "config_digest": config_digest,
             "payload": deepcopy(dict(payload)), "artifacts": records}
    atomic_write_json(target, value, role="report_checkpoint", overwrite=False)
    write_stage_receipt(root / "stage_receipts", task_id="analysis", attempt_id="initial",
                        stage=stage, status="completed", input_digest=source_digest,
                        config_digest=config_digest,
                        artifacts=[{"path": target.relative_to(root).as_posix(),
                                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}])
    return target


def _load_checkpoint(root: Path, stage: str, *, source_digest: str,
                     config_digest: str) -> dict:
    path = local_file(root, f"recovery_v4/{stage}.json")
    receipt = local_file(root, f"stage_receipts/analysis.initial.{stage}.json")
    if path is None or receipt is None:
        raise ValueError("verified report checkpoint is unavailable")
    raw = path.read_bytes()
    record = json.loads(receipt.read_text("utf-8"))
    if (record.get("status") != "completed" or record.get("stage") != stage
            or record.get("input_digest") != source_digest or record.get("config_digest") != config_digest
            or record.get("artifacts") != [{"path": f"recovery_v4/{stage}.json", "sha256": hashlib.sha256(raw).hexdigest()}]):
        raise ValueError("checkpoint receipt or bindings changed")
    checkpoint = json.loads(raw)
    if (checkpoint.get("schema_version") != "4.0" or checkpoint.get("stage") != stage
            or checkpoint.get("source_report_sha256") != source_digest
            or checkpoint.get("config_digest") != config_digest):
        raise ValueError("checkpoint input or configuration mismatch")
    for artifact in checkpoint["artifacts"]:
        source = local_file(root, artifact["path"])
        if source is None or hashlib.sha256(source.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("checkpoint evidence artifact missing or changed")
    return checkpoint


def recover_analysis_report(task_dir: str | Path, report: Mapping, *, config: Mapping,
                            settings, stop_requested: bool = False) -> dict:
    """An explicit call repairs PDF rendering using a verified saved decision.

    A render-ready checkpoint reuses the saved writer. If interruption happened
    before writing completed, the decision checkpoint uses a local template and
    records that choice. An existing report is preserved in its entirety.
    """
    if stop_requested:
        raise ValueError("report recovery is stopped")
    root = Path(task_dir).resolve(strict=True)
    if any((root / name).exists() for name in ("STOP", "CANCEL_SHUTDOWN")):
        raise ValueError("task has a stop marker")
    source_report = {k: v for k, v in report.items() if k not in {"analysis", "evidence_package"}}
    source_digest = canonical_digest(source_report)
    config_digest = report_config_digest(config, settings)
    stage = "report_ready" if (root / "recovery_v4/report_ready.json").exists() else "decision"
    checkpoint = _load_checkpoint(root, stage, source_digest=source_digest, config_digest=config_digest)
    data = checkpoint["payload"]
    final = data["final_decision"]
    validate_saved_decision(final, data["decision_input"], requested_mode=settings.options["decision_mode"],
                            source_hashes=final["rule_source_hashes"])
    # Bind to recorded rule sources, not today's code: rendering is not a new
    # judgment. Its final object must exactly match the immutable decision file.
    original_final = local_file(root, "final_decision.json")
    if original_final is None or json.loads(original_final.read_text("utf-8")) != final:
        raise ValueError("saved decision differs from checkpoint")
    from .analysis_templates import build_template_content
    from .analysis_pdf import render_analysis_pdf
    from .llm_client import DEFAULT_LLM_CONFIG
    package = deepcopy(data["package"])
    writer = deepcopy(data.get("writer") or {"status": "fallback", "content": {},
                                            "reason": "显式补报告：复用已保存决策，本次使用本地模板，未重试云端"})
    labels = {**DEFAULT_LLM_CONFIG["band_labels"], **settings.options["band_labels"]}
    content = deepcopy(data.get("content") or build_template_content(package, final, band_labels=labels))
    if not data.get("content"):
        from .coverage_summary_v4 import append_coverage_notes
        content = append_coverage_notes(content, package)
    attempt_id = uuid.uuid4().hex
    # Keep nested report recovery usable under Windows' default path budget.
    # Full identity stays in the saved receipt; collision never overwrites.
    destination = root / "rr_v4" / attempt_id[:16]
    destination.mkdir(parents=True, exist_ok=False)
    # Only explicit verified files are copied; no arbitrary directory scan.
    for artifact in checkpoint["artifacts"]:
        source = local_file(root, artifact["path"])
        raw = source.read_bytes() if source else b""
        if hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
            raise ValueError("evidence changed during report recovery")
        atomic_write_bytes(destination / artifact["path"], raw, role="recovery_evidence", overwrite=False)
    for figure in package.get("figures", []):
        for key in ("original_path", "path", "preview_path"):
            if figure.get(key):
                original = local_file(root, figure[key])
                if original is not None:
                    target = destination / original.relative_to(root)
                    if target.is_file():
                        figure[key] = str(target)
    product_name = source_report.get("product", {}).get("name") or PRODUCT_NAME
    kwargs = {"band_labels": labels, "product_name": product_name}
    pdf = render_analysis_pdf(package, final, content, destination, **kwargs)
    pdf_name = "研判分析报告.pdf"
    atomic_write_bytes(destination / pdf_name, pdf, role="recovered_pdf", overwrite=False)
    records = []
    values = {"analysis_source_report.json": source_report, "final_decision.json": final,
              "evidence_package.json": package, "analysis_content.json": content}
    for name, value in values.items():
        existing = destination / name
        if existing.exists() and json.loads(existing.read_text("utf-8")) != value:
            # package paths are intentionally rebased only in this new attempt.
            if name != "evidence_package.json":
                raise ValueError("recovery artifact conflicts with original")
        atomic_write_json(existing, value, role="recovery_json")
    names = set(values) | {a["path"] for a in checkpoint["artifacts"]} | {pdf_name, "analysis_pdf_render.json"}
    for name in sorted(names):
        path = local_file(destination, name)
        if path is not None:
            records.append({"path": name, "kind": "recovery_artifact", "size_bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {"schema_version": "4.0", "status": "success", "source_report_sha256": source_digest,
              "evidence_package": package, "final_decision": final, "writer": writer, "content": content,
              "pdf_file": pdf_name, "settings": settings.public_status(), "artifact_manifest": records,
              "artifact_manifest_file": "analysis_artifact_manifest.json", "external_language_model_used": False,
              "recovery": {"attempt_id": attempt_id, "action": "render_report", "source_stage": stage,
                           "model_calls": 0, "cloud_calls": 0,
                           "historical_cloud_used": final.get("source") == "llm" or writer.get("status") in {"success", "partial"},
                           "original_task_unchanged": True}}
    atomic_write_json(destination / "analysis_report.json", result, role="recovery_result", overwrite=False)
    path = destination / "analysis_report.json"
    records.append({"path": path.name, "kind": "analysis_result", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    atomic_write_json(destination / "analysis_artifact_manifest.json", {"schema_version": "4.0", "artifacts": records},
                      role="recovery_manifest", overwrite=False)
    return {"task_dir": str(destination), "result": result}
