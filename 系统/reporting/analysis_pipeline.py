"""从已有任务生成研判报告，保留机器结果并另存模型调用及图件证据。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from .evidence_figures import collect_figures, local_file
from .evidence_package import basename, build_evidence_package, mapping, to_llm_payload
from .llm_client import LLMSettings, canonical_bytes, load_llm_settings, write_json
from .llm_service import run_judge, run_writer
from product_info import PRODUCT_NAME
from artifact_io import atomic_write_bytes, write_stage_receipt
from .analysis_recovery import report_config_digest, save_report_checkpoint


def _file_record(root: Path, path: Path, kind: str) -> dict | None:
    safe = local_file(root, path)
    if safe is None:
        return None
    return {"path": safe.relative_to(root).as_posix(), "kind": kind,
            "size_bytes": safe.stat().st_size, "sha256": hashlib.sha256(safe.read_bytes()).hexdigest()}


def verify_analysis_artifacts(task_dir: str | Path) -> list[str]:
    """独立研判清单覆盖输入、原图、调用审计与报告；旧任务也能校验。"""
    root = Path(task_dir).resolve(strict=True)
    manifest_path = local_file(root, "analysis_artifact_manifest.json")
    if manifest_path is None:
        return ["研判证据清单缺失"]
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
        records = manifest["artifacts"]
        if not isinstance(records, list) or not records:
            raise ValueError()
    except (OSError, ValueError, KeyError, TypeError):
        return ["研判证据清单不可读取"]
    problems = []
    paths = {record.get("path") for record in records if isinstance(record, Mapping) and isinstance(record.get("path"), str)}
    for name in ("analysis_report.json", "analysis_source_report.json", "evidence_package.json", "final_decision.json", "研判分析报告.pdf"):
        if name not in paths:
            problems.append(name + " 未列入研判证据清单")
    for record in records:
        if not isinstance(record, Mapping):
            problems.append("研判证据条目格式无效")
            continue
        path = local_file(root, record.get("path"))
        if path is None or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
            problems.append(str(record.get("path") or "未记录文件") + " 缺失或摘要不一致")
    return problems


def _complete_input_evidence(root: Path, report: dict, package: dict) -> Path | None:
    material = next(v["data"] for v in package["items"] if v["item_id"] == "E09")
    process = next(v["data"] for v in package["items"] if v["item_id"] == "E10")
    task_input = mapping(mapping(mapping(report.get("provenance")).get("task_status")).get("input"))
    video = mapping(mapping(report.get("input")).get("video"))
    input_path = (task_input.get("video_file") or video.get("video_file")) if material["type"] == "video" else mapping(report.get("input")).get("query_image")
    fixed_input = local_file(root, input_path)
    if fixed_input is None and input_path:
        # 移交后的历史任务可能保留原绝对路径，只读本任务内同名固定件，并重新核对摘要。
        fixed_input = local_file(root, basename(input_path))
    if fixed_input is not None:
        material["size_bytes"] = fixed_input.stat().st_size
        material["size_basis"] = "任务内已固定输入文件"
        digest = hashlib.sha256(fixed_input.read_bytes()).hexdigest()
        material["computed_input_sha256"] = digest
        material["hash_mismatch"] = digest != material.get("input_sha256") if material.get("input_sha256") else None
    selection_path = local_file(root, task_input.get("frame_selection") or video.get("frame_selection"))
    if material["type"] == "video" and selection_path is not None:
        try:
            if selection_path.stat().st_size > 20 * 1024 * 1024:
                raise ValueError("采样记录过大")
            selection = mapping(json.loads(selection_path.read_text("utf-8-sig")))
            if not material.get("input_sha256") or selection.get("video_sha256") != material["input_sha256"]:
                raise ValueError("采样记录不对应当前视频")
            process["frame_selection"] = {k: deepcopy(selection.get(k)) for k in (
                "status", "sampling", "selection", "face_detection", "selected_coverage", "input_parameters")}
            process["frame_selection"]["source_file"] = selection_path.relative_to(root).as_posix()
            process["frame_selection"]["source_sha256"] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
            process["candidate_frame_count"] = mapping(selection.get("sampling")).get("candidate_frames_decoded")
            process["selected_frame_count"] = mapping(selection.get("selection")).get("selected_frames")
        except (OSError, ValueError, TypeError):
            process["frame_selection"] = {"status": "unavailable", "note": "采样记录缺失、损坏或不对应当前视频"}
            process.pop("candidate_frame_count", None)
            process.pop("selected_frame_count", None)
    return fixed_input


def _record_figure_transmission(root: Path, figures: list[dict], final: dict, writer: dict, settings: LLMSettings) -> dict:
    keys = ("planned_figure_ids", "included_request_figure_ids", "attempted_figure_ids", "uploaded_figure_ids",
            "delivery_unknown_figure_ids", "transmission_status", "transmission_note", "omitted_for_request_size")
    stages = {stage: {key: deepcopy(result.get(key, [] if key.endswith("ids") or key == "omitted_for_request_size" else "not_sent"))
                      for key in keys} for stage, result in (("judge", final), ("writer", writer))}
    for figure in figures:
        identifier = figure["figure_id"]
        figure["transmission"] = {}
        for stage, audit in stages.items():
            confirmed = identifier in audit["uploaded_figure_ids"]
            unknown = identifier in audit["delivery_unknown_figure_ids"]
            status = ("cached_response" if audit["transmission_status"] == "cached_response" else "response_received") if confirmed else "delivery_unknown" if unknown else "not_sent"
            figure["transmission"][stage] = {"status": status, "confirmed": confirmed,
                "attempted": identifier in audit["attempted_figure_ids"],
                "omitted_for_request_size": identifier in audit["omitted_for_request_size"]}
        figure["uploaded"] = any(v["confirmed"] for v in figure["transmission"].values())
        figure["delivery_unknown"] = any(v["status"] == "delivery_unknown" for v in figure["transmission"].values())
    path = root / "report_figures" / "manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    manifest["figures"] = figures
    manifest["transmission_stages"] = stages
    write_json(path, manifest)
    return {"upload_scope": settings.options["upload_scope"], "judge_send_images": settings.options["judge_send_images"], "stages": stages}


def generate_analysis_report(task_dir: str | Path, report: Mapping[str, Any], *, config: Mapping[str, Any],
                             settings: LLMSettings | None = None, profile_path: str | Path | None = None,
                             force: bool = False, bypass_cache: bool = False) -> dict:
    """新增报告及审计文件；不重写输入report或历史final_report.json。"""
    root = Path(task_dir).resolve(strict=True)
    report = deepcopy(dict(report))
    report.pop("evidence_package", None)
    report.pop("analysis", None)
    source_hash = hashlib.sha256(canonical_bytes(report)).hexdigest()
    saved_path = root / "analysis_report.json"
    if saved_path.exists():
        if force:
            raise ValueError("已有研判原件不能覆盖；如需重新分析，请另建任务副本")
        safe_saved = local_file(root, "analysis_report.json")
        if safe_saved is None:
            raise ValueError("研判记录路径无效")
        saved = json.loads(safe_saved.read_text("utf-8-sig"))
        if saved.get("source_report_sha256") == source_hash and local_file(root, saved.get("pdf_file")):
            if verify_analysis_artifacts(root):
                raise ValueError("已有研判证据缺失或发生变化，不能复用缓存")
            return saved
        raise ValueError("任务已有不同来源或不完整的研判原件，请另建任务副本重新分析")
    if (root / "final_decision.json").exists() or (root / "recovery_v4").exists():
        raise ValueError("任务已有决策或中断记录；请显式补报告，不能自动重新判定或覆盖原件")
    if profile_path is None:
        data_root = Path(os.environ.get("FSTS_DATA_ROOT") or Path(__file__).resolve().parents[2] / "用户数据")
        profile_path = data_root / "profile" / "llm.json"
    settings = settings or load_llm_settings(config, profile_path=profile_path)
    config_digest = report_config_digest(config, settings)
    started = time.perf_counter()
    package = build_evidence_package(report, config=config)
    fixed_input = _complete_input_evidence(root, report, package)
    figures = collect_figures(report, package, root, config={**dict(config), "llm": settings.options})
    package["figures"] = figures
    write_json(root / "analysis_source_report.json", report)
    evidence_records = []
    for name in ("analysis_source_report.json", "detection_result.json", "general_aigc_result.json", "identity_result.json", "source_result.json",
                 "file_provenance_result.json", "visible_label_result.json", "pair_compare_result.json", "frame_selection.json"):
        record = _file_record(root, root / name, "module_result")
        if record:
            evidence_records.append(record)
    if fixed_input is not None:
        evidence_records.append(_file_record(root, fixed_input, "fixed_input"))
    for figure in figures:
        record = _file_record(root, Path(figure["original_path"]), "original_figure")
        if record and record["path"] not in {v["path"] for v in evidence_records}:
            evidence_records.append(record)
    input_manifest = {"source_report_sha256": source_hash, "artifacts": evidence_records,
                      "note": "报告生成前固定的输入证据，避免报告与总清单相互引用形成摘要循环"}
    write_json(root / "analysis_input_manifest.json", input_manifest)
    package["source_manifest"] = {"file": "analysis_input_manifest.json",
                                  "sha256": hashlib.sha256((root / "analysis_input_manifest.json").read_bytes()).hexdigest()}
    aliases: dict[str, str] = {}
    view = to_llm_payload(package, settings.options["upload_scope"], alias_map=aliases)
    if aliases:
        write_json(root / "llm_alias_map.json", aliases)
    final = run_judge(view, settings=settings, task_dir=root, figures=figures, bypass_cache=bypass_cache)
    checkpoint_names = [record["path"] for record in evidence_records]
    checkpoint_names += ["analysis_input_manifest.json", "final_decision.json"] + final.get("audit_files", [])
    save_report_checkpoint(root, stage="decision", source_digest=source_hash, config_digest=config_digest,
                           payload={"package": package, "decision_input": view, "final_decision": final},
                           artifact_names=checkpoint_names)
    writer = run_writer(view, final, settings=settings, task_dir=root, figures=figures, bypass_cache=bypass_cache)
    package["llm_processing"] = _record_figure_transmission(root, figures, final, writer, settings)
    from .analysis_templates import build_template_content, merge_writer_content
    from .analysis_pdf import render_analysis_pdf
    template = build_template_content(package, final, band_labels=settings.options["band_labels"])
    content = merge_writer_content(template, writer)
    from .coverage_summary_v4 import append_coverage_notes
    content = append_coverage_notes(content, package)
    save_report_checkpoint(root, stage="report_ready", source_digest=source_hash, config_digest=config_digest,
                           payload={"package": package, "decision_input": view, "final_decision": final,
                                    "writer": writer, "content": content},
                           artifact_names=checkpoint_names + writer.get("audit_files", []))
    pdf = render_analysis_pdf(package, final, content, root,
                              product_name=mapping(report.get("product")).get("name", PRODUCT_NAME),
                              band_labels=settings.options["band_labels"])
    pdf_name = "研判分析报告.pdf"
    atomic_write_bytes(root / pdf_name, pdf, role="analysis_pdf", overwrite=False)
    write_json(root / "evidence_package.json", package)
    write_json(root / "analysis_content.json", content)
    audit_files = list(dict.fromkeys(final.get("audit_files", []) + writer.get("audit_files", []) + [
        "final_decision.json", "evidence_package.json", "analysis_content.json", "analysis_input_manifest.json",
        "analysis_pdf_render.json", "report_figures/manifest.json", "llm_alias_map.json", pdf_name]))
    audit_files += ["recovery_v4/decision.json", "recovery_v4/report_ready.json",
                    "stage_receipts/analysis.initial.decision.json", "stage_receipts/analysis.initial.report_ready.json"]
    audit_records = list(evidence_records)
    for name in audit_files:
        record = _file_record(root, root / name, "analysis_audit")
        if record and record["path"] not in {v["path"] for v in audit_records}:
            audit_records.append(record)
    for folder in ("llm_upload", "report_figures"):
        for path in sorted((root / folder).iterdir()):
            record = _file_record(root, path, folder)
            if record and record["path"] not in {v["path"] for v in audit_records}:
                audit_records.append(record)
    result = {"schema_version": "1.0", "status": "success", "source_report_sha256": source_hash,
              "evidence_package": package, "final_decision": final, "writer": writer,
              "content": content, "pdf_file": pdf_name, "runtime_seconds": round(time.perf_counter() - started, 6),
              "settings": settings.public_status(), "external_language_model_used": final.get("source") == "llm" or writer.get("status") in {"success", "partial"},
              "artifact_manifest": audit_records, "artifact_manifest_file": "analysis_artifact_manifest.json"}
    write_json(saved_path, result)
    # 清单最后写，覆盖最终研判记录本身；记录不反向存清单摘要，避免循环引用。
    all_records = [*audit_records, _file_record(root, saved_path, "analysis_result")]
    write_json(root / "analysis_artifact_manifest.json", {"schema_version": "1.0", "artifacts": all_records})
    write_stage_receipt(root / "stage_receipts", task_id="analysis", attempt_id="initial", stage="report",
                        status="completed", input_digest=source_hash, config_digest=config_digest,
                        artifacts=[_file_record(root, root / "analysis_artifact_manifest.json", "analysis_manifest")])
    return result


def generate_review_analysis(task_dir: str | Path, *, reviews: list[dict], product_name: str = PRODUCT_NAME) -> bytes:
    """人工复核只追加一份新PDF，不改自动研判原件。"""
    from .analysis_pdf import render_analysis_pdf
    root = Path(task_dir).resolve(strict=True)
    saved = local_file(root, "analysis_report.json")
    if saved is None:
        raise ValueError("请先生成研判分析报告")
    data = json.loads(saved.read_text("utf-8-sig"))
    if verify_analysis_artifacts(root):
        raise ValueError("研判原件校验未通过，不能生成复核版")
    pdf = render_analysis_pdf(data["evidence_package"], data["final_decision"], data["content"], root,
                              product_name=product_name, band_labels=data.get("settings", {}).get("band_labels"), review=reviews)
    target = root / "研判分析报告（复核版）.pdf"
    atomic_write_bytes(target, pdf, role="review_pdf")
    write_json(root / "analysis_review_artifact_manifest.json", {
        "schema_version": "1.0", "review_records_sha256": hashlib.sha256(canonical_bytes(reviews)).hexdigest(),
        "artifacts": [_file_record(root, target, "analysis_review"),
                      _file_record(root, root / "analysis_pdf_review_render.json", "analysis_review_render")],
        "source_analysis_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
    })
    return pdf
