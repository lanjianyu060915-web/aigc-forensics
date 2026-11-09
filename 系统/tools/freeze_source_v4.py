"""Plan and materialize a new, explicitly enumerated source-only snapshot.

No git operation, model execution, dataset enumeration or upload is performed.
The CLI only accepts the approved development worktree as source. A clean
source manifest does not certify runnable assets or authorize remote publishing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys


VERSION = "source-freeze-v4.1"
SOURCE_ROOT = Path("D:/aigc-forensics-v4")
PRODUCT_ROOT = Path(__file__).resolve().parents[1]
FONT_HASHES = {
    "系统/assets/fonts/LXGWWenKai-Regular.ttf": "0819604ad9357afbc6e7cc097328a16ed85143ee0848ffda09fda35a30da6506",
    "系统/assets/fonts/LXGWWenKai-Bold.ttf": "7a356605eb36c92e29cc64090e2856e4675694572a2ae4da84adbadffecae907",
    "系统/assets/fonts/OFL.txt": "04b0fffb7719757b4452067d0e2de7da444988f6b9a7951cfc8c9ff95696aec5",
}
# Exact generic-source cases reviewed by the coordinator; this is not a data
# admission exception. Any additional path/hash needs a separate code review.
BACKUP_SOURCE_REVIEW_SCOPE = {
    "系统/case_libraries.py": "42f8f5fc8c90bdf85918a7d9c82a85f2e58866b90bd39974037fa06283be7f40",
    "系统/frontend/components/evidence_dashboard.py": "df7ed96f4390095dc4514f4a698e71c3ba145d5bec2894dc042e82a2b75ef67c",
    "系统/frontend/components/llm_views.py": "6190b161ab379de43cb0f760b799f0d2dba2991c3bf6be6f738e4cc24d14accf",
    "系统/frontend/components/result_views.py": "80063548d0d587270890bfbb972628921a71b089f864818174e252c90757c1fc",
    "系统/modules/face_crops/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "系统/modules/identity_search/identity_search/reference_detection.py": "14f2dab44734bc69f01599d727550065d731bb522cda541a62ae539114dc4c49",
    "系统/modules/identity_search_v2/run_identity_search_v2.py": "bc7e1ed0cc386d8a0188073c7dd5b0d057bb6590489f87f786f01a11d96d0292",
}
DENIED_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", ".run_tmp", "sample_data", "mock_results",
    "mock_task", "reports", "logs", "cache", "caches", "outputs", "results", "datasets", "data", "history",
    "experiments", "research", "training", "train", "evaluation", "eval", "评测", "文档", "用户数据",
    "python_runtime", "runtime_data", "reference_snapshots", "models"})
DENIED_NAMES = frozenset({"config.json", "agents.md", "known_issues.md", "reference_statistics.json",
    "band_evidence.json", ".env", ".git", "secrets.toml"})
SOURCE_AREAS = ("系统/contracts/", "系统/decision/", "系统/engine/aigc_engine/", "系统/frontend/components/",
                "系统/modules/", "系统/reporting/", "系统/video_pipeline/")
TOOLS = frozenset({"build_source_cache.py", "check_runtime.py", "package_checksums.py", "prepare_v4_profile.py",
    "prepare_v4_inference_profile.py", "runtime_builder.py", "self_check.py", "setup_runtime.ps1",
    "verify_source_cache.py", "freeze_source_v4.py", "run_v4_dev_batch.py", "recover_task_v4.py",
    "run_v4_acceptance_batch.py", "evaluate_v4_batch.py", "run_v4_sampling_comparison.py"})
DATA_TOOLS = frozenset({"build_cross_manifests.py", "calibrate_general_aigc.py", "eval_case3.py",
    "eval_full_image_video.py", "eval_general_aigc.py", "recalibrate_detector_v2.py", "report_eval.py",
    "reproduce_detector_metrics.py", "run_engine_tests.py"})
EXPLICIT_TEXT = frozenset({".gitignore", ".gitattributes", "THIRD_PARTY.md", "README_V4.md", "CHANGELOG_V4.md", "搭建环境.bat", "启动系统.bat", "运行自检.bat",
    "系统/.streamlit/config.toml", "系统/modules/visible_label/keywords.json", "系统/modules/pair_compare/rule.json",
    "系统/requirements/runtime-sources.json", "系统/frontend/requirements.txt"})
LICENSE_NAMES = frozenset({"license", "license.txt", "license.md", "copying", "notice", "ofl.txt"})
MAX_SOURCE_BYTES = 32 * 1024 * 1024
SECRET_PATTERNS = [
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("provider_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|LTAI[A-Za-z0-9]{16,})\b")),
    ("credential_url", re.compile(r"https?://[^\s/@:]+:[^\s/@]+@", re.I)),
    ("literal_bearer", re.compile(r"Bearer\s+[A-Za-z0-9_.-]{20,}")),
]
SENSITIVE_LITERAL = re.compile(r'''(?ix)["']?(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)["']?\s*[:=]\s*["']([^"'\r\n]+)["']''')


class FreezeError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def relative(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise FreezeError("invalid_relative_source_path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise FreezeError("invalid_relative_source_path")
    return path.as_posix()


def _signature(value):
    return value.st_size, value.st_mtime_ns, value.st_dev, value.st_ino


def _no_links(path, root):
    root = root.absolute()
    try:
        parts = path.absolute().relative_to(root).parts
    except ValueError as exc:
        raise FreezeError("path_outside_source_root") from exc
    current = root
    for part in (None, *parts):
        if part is not None:
            current = current / part
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise FreezeError("links_or_junctions_forbidden")
    if path.resolve(strict=True) != path.absolute():
        raise FreezeError("resolved_path_changed")


def stable_bytes(path, root):
    _no_links(path, root)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_SOURCE_BYTES:
            raise FreezeError("not_regular_or_size_budget_exceeded")
        raw = stream.read(MAX_SOURCE_BYTES + 1)
        after = os.fstat(stream.fileno())
    _no_links(path, root)
    if len(raw) > MAX_SOURCE_BYTES or _signature(before) != _signature(after) or _signature(path.stat()) != _signature(after):
        raise FreezeError("source_changed_during_read")
    return raw


def _path_policy(name, mode, audited_tests):
    path = PurePosixPath(name)
    parts = path.parts
    if name == "系统/tools/prepare_v4_manual_profile.py":
        return "excluded", "local_private_deployment_configuration_tool_delivered_separately"
    if path.name.casefold() in DENIED_NAMES:
        return "excluded", "private_config_boundary_or_historical_result"
    if "tests" in parts:
        return ("candidate", "explicit_audited_synthetic_test") if mode == "backup" and name in audited_tests else ("excluded", "tests_not_explicitly_admitted")
    if name == "系统/v4_test_isolation.py" or name == "系统/pytest.ini":
        return ("candidate", "synthetic_test_support") if mode == "backup" and audited_tests else ("excluded", "runtime_omits_test_support")
    if name in FONT_HASHES:
        return "candidate", "explicit_licensed_font_resource"
    if name.startswith("系统/assets/"):
        return "excluded", "asset_not_explicitly_admitted"
    if name in EXPLICIT_TEXT or path.name.casefold() in LICENSE_NAMES:
        return "candidate", "explicit_configuration_or_license"
    if name.startswith("系统/reporting/prompts/") and path.suffix == ".md":
        return "candidate", "runtime_prompt"
    if name.startswith("系统/requirements/") and path.suffix == ".txt":
        return "candidate", "dependency_lock"
    if path.name in {"requirements.txt", "requirements.lock.txt"} and name.startswith(SOURCE_AREAS):
        return "candidate", "dependency_lock"
    if name.startswith("系统/tools/"):
        if path.name in TOOLS and len(parts) == 3:
            return "candidate", "explicit_runtime_tool"
        if path.name in DATA_TOOLS:
            return "excluded", "training_evaluation_or_legacy_data_tool"
    if name == "系统/frontend/app.py" or (len(parts) == 2 and parts[0] == "系统" and path.suffix in {".py", ".ps1", ".bat"}):
        return "candidate", "runtime_entrypoint"
    if name.startswith(SOURCE_AREAS) and path.suffix == ".py":
        return "candidate", "python_source"
    if path.suffix.casefold() in {".py", ".ps1", ".bat", ".toml", ".yaml", ".yml", ".json", ".ttf", ".dll", ".exe"}:
        return "rejected", "unreviewed_source_configuration_or_binary"
    if path.suffix.casefold() in {".bin", ".dat", ".so", ".dylib", ".zip", ".7z", ".rar"}:
        return "rejected", "unknown_binary_not_admitted"
    return "excluded", "documentation_or_non_source_not_admitted"


def inventory(root, mode, audited_tests):
    """List source metadata only; prune excluded directories before traversal."""
    rows = []
    for folder, directories, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(folder)
        for name in sorted(directories[:]):
            path = base / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                directories.remove(name)
                rows.append({"path": rel, "kind": "directory", "classification": "rejected", "reason": "links_or_junctions_forbidden"})
            elif (name.casefold() in DENIED_DIRS and rel != "系统/engine/aigc_engine/models") or (name == "tests" and mode == "runtime"):
                directories.remove(name)
                rows.append({"path": rel, "kind": "directory", "classification": "excluded", "reason": "excluded_directory_not_enumerated"})
            elif rel == "系统/assets":
                # Only fonts can be enumerated; other assets are never read.
                pass
            elif rel.startswith("系统/assets/") and rel != "系统/assets/fonts":
                directories.remove(name)
                rows.append({"path": rel, "kind": "directory", "classification": "excluded", "reason": "assets_not_admitted"})
        for name in sorted(files):
            path = base / name
            rel = path.relative_to(root).as_posix()
            state, reason = _path_policy(rel, mode, audited_tests)
            if path.is_symlink():
                state, reason = "rejected", "links_or_junctions_forbidden"
            rows.append({"path": rel, "kind": "file", "classification": state, "reason": reason})
    return sorted(rows, key=lambda row: row["path"])


def scan_bytes(name, raw, protected_roots):
    findings = []
    if name in FONT_HASHES:
        if sha(raw) != FONT_HASHES[name]:
            return "rejected", [{"reason": "licensed_font_resource_hash_mismatch"}]
        if name.endswith(".ttf"):
            if raw[:4] not in {b"\x00\x01\x00\x00", b"OTTO", b"ttcf"}:
                return "rejected", [{"reason": "font_magic_mismatch"}]
            return "licensed_font_binary", []
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError:
        return "rejected", [{"reason": "unknown_binary_or_non_utf8_text"}]
    if any(ord(char) < 32 and char not in "\n\r\t" for char in text):
        return "rejected", [{"reason": "binary_control_byte_in_text"}]
    for reason, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append({"reason": reason, "line": text.count("\n", 0, match.start()) + 1})
    for match in SENSITIVE_LITERAL.finditer(text):
        literal = match.group(1)
        # Recognizable public placeholders are allowed, actual provider patterns
        # above are always rejected. Findings never echo credential bytes.
        if len(literal) >= 8 and not re.search(r"(?i)(example|placeholder|synthetic|dummy|fake|test|your_|changeme|\{|\})", literal):
            findings.append({"reason": "nonempty_credential_literal_needs_review", "line": text.count("\n", 0, match.start()) + 1})
    normalized = text.replace("\\\\", "\\").replace("\\", "/").casefold()
    for root in protected_roots:
        protected = str(root).replace("\\", "/").rstrip("/").casefold()
        match = re.search(re.escape(protected) + r'''(?=/|[\s"']|$)''', normalized) if protected else None
        if match:
            findings.append({"reason": "literal_protected_path", "line": normalized.count("\n", 0, match.start()) + 1})
    return "utf8_text", findings


def policy_binding(context):
    bound = context["bindings"]
    return {key: bound[key] for key in ("policy_sha256", "policy_version", "checker_sha256")}


def _protected(path, context):
    return context["core"].within(path, context["roots"]) or context["core"].within(path.resolve(), context["roots"])


def _validate_local_review(review, context):
    if not review:
        return {}
    expected_keys = {"schema_version", "purpose", "upload_prohibited", "policy_binding", "provenance_review_sha256", "files"}
    if (not isinstance(review, dict) or set(review) != expected_keys or review.get("schema_version") != "source_overlap_review.v1"
            or review.get("purpose") != "local_runtime_only" or review.get("upload_prohibited") is not True
            or review.get("policy_binding") != policy_binding(context)
            or not re.fullmatch(r"[a-f0-9]{64}", str(review.get("provenance_review_sha256", "")))):
        raise FreezeError("local_source_review_missing_or_not_bound_to_current_policy")
    records = review.get("files")
    if not isinstance(records, dict) or not records:
        raise FreezeError("explicit_local_source_review_files_required")
    for name, record in records.items():
        relative(name)
        if (not name.endswith(".py") or not isinstance(record, dict) or set(record) != {"sha256", "reviewed_generic_source", "upload_prohibited"}
                or record.get("reviewed_generic_source") is not True
                or record.get("upload_prohibited") is not True
                or not re.fullmatch(r"[a-f0-9]{64}", str(record.get("sha256", "")))):
            raise FreezeError("local_source_review_only_allows_exact_python_source")
    return records


def _validate_backup_review(review, context):
    if not review:
        return {}
    expected_keys = {"schema_version", "purpose", "remote_action_authorized", "policy_binding", "provenance_review_sha256", "files"}
    if (not isinstance(review, dict) or set(review) != expected_keys
            or review.get("schema_version") != "source_overlap_backup_review.v1"
            or review.get("purpose") != "source_only_github_backup" or review.get("remote_action_authorized") is not False
            or review.get("policy_binding") != policy_binding(context)
            or not re.fullmatch(r"[a-f0-9]{64}", str(review.get("provenance_review_sha256", "")))):
        raise FreezeError("source_backup_review_scope_or_policy_invalid")
    records = review.get("files")
    if not isinstance(records, dict) or set(records) != set(BACKUP_SOURCE_REVIEW_SCOPE):
        raise FreezeError("source_backup_review_must_match_exact_preapproved_paths")
    for name, file_hash in BACKUP_SOURCE_REVIEW_SCOPE.items():
        if records[name] != {"sha256": file_hash, "reviewed_generic_source": True, "training_allowed": False}:
            raise FreezeError("source_backup_review_must_match_exact_preapproved_hashes")
    return records


def build_plan(root, context, *, mode="runtime", audited_tests=None, local_runtime_source_review=None, backup_source_review=None):
    if mode not in {"runtime", "backup"}:
        raise FreezeError("unsupported_snapshot_mode")
    root = Path(root).absolute()
    _no_links(root, root)
    if _protected(root, context):
        raise FreezeError("source_root_is_protected")
    audited_tests = audited_tests or {}
    local_runtime_source_review = local_runtime_source_review or {}
    local_review = _validate_local_review(local_runtime_source_review, context)
    backup_source_review = backup_source_review or {}
    backup_review = _validate_backup_review(backup_source_review, context)
    for name, record in audited_tests.items():
        relative(name)
        if ("tests" not in PurePosixPath(name).parts or not isinstance(record, dict)
                or set(record) != {"reviewed_synthetic", "sha256"} or record.get("reviewed_synthetic") is not True
                or not re.fullmatch(r"[a-f0-9]{64}", str(record.get("sha256", "")))):
            raise FreezeError("explicit_synthetic_test_review_required")
    records, excluded, rejected, overlaps = [], [], [], []
    for row in inventory(root, mode, audited_tests):
        if row["classification"] == "excluded":
            excluded.append(row)
            continue
        if row["classification"] == "rejected":
            rejected.append(row)
            continue
        path = root / row["path"]
        try:
            if _protected(path, context):
                raise FreezeError("protected_path_or_resolved_target")
            raw = stable_bytes(path, root)
            file_hash = sha(raw)
            file_type, findings = scan_bytes(row["path"], raw, context["roots"])
            protected_collision = file_hash in context["hashes"]
            locally_reviewed = (mode == "runtime" and local_review.get(row["path"], {}).get("sha256") == file_hash)
            backup_reviewed = (mode == "backup" and backup_review.get(row["path"], {}).get("sha256") == file_hash)
            if protected_collision and not (locally_reviewed or backup_reviewed):
                findings.append({"reason": "protected_sha256_collision_no_upload"})
            if row["path"] in audited_tests and audited_tests[row["path"]].get("sha256") != file_hash:
                findings.append({"reason": "audited_test_hash_changed"})
            item = {"path": row["path"], "sha256": file_hash, "size_bytes": len(raw), "file_type": file_type,
                    "role": row["reason"], "findings": findings}
            if findings or file_type == "rejected":
                rejected.append({**item, "classification": "rejected"})
            else:
                if protected_collision:
                    overlap = {"path": row["path"], "sha256": file_hash,
                               "disposition": "reviewed_local_runtime_only" if locally_reviewed else "reviewed_source_only_backup_inclusion",
                               "upload_prohibited": bool(locally_reviewed), "training_allowed": False,
                               "remote_action_authorized": False}
                    item["protected_source_overlap"] = overlap
                    overlaps.append(overlap)
                records.append(item)
        except (OSError, FreezeError) as exc:
            rejected.append({"path": row["path"], "classification": "rejected", "reason": type(exc).__name__ + ":" + str(exc)})
    present = {row["path"] for row in records + rejected}
    for name in sorted(set(audited_tests) - present):
        rejected.append({"path": name, "classification": "rejected", "reason": "audited_test_missing_or_excluded"})
    value = {"schema_version": VERSION, "mode": mode, "source_files": records, "excluded": excluded, "rejected": rejected,
             "policy_binding": policy_binding(context), "audited_tests": audited_tests,
             "local_runtime_source_review": local_runtime_source_review, "backup_source_review": backup_source_review,
             "protected_source_overlaps": overlaps,
             "distribution_restriction": "local_runtime_only_no_upload" if any(item["upload_prohibited"] for item in overlaps) else "remote_publication_requires_separate_audit",
             "summary": {"source_file_count": len(records), "source_bytes": sum(row["size_bytes"] for row in records),
                         "excluded_entry_count": len(excluded), "rejected_entry_count": len(rejected),
                         "local_only_overlap_count": sum(item["upload_prohibited"] for item in overlaps),
                         "backup_source_overlap_count": sum(not item["upload_prohibited"] for item in overlaps),
                         "types": dict(Counter(row["file_type"] for row in records))},
             "status": "blocked" if rejected else "source_manifest_ready_for_review",
             "runtime_readiness": "not_asserted_external_interpreter_models_and_frozen_config_required",
             "upload_authorized": False, "training_allowed": False,
             "omissions": ["No active config, model files, old statistics, data, caches, historical reports or git history.",
                           "Directory exclusions are pruned; excluded_entry_count is not a count of their unvisited files.",
                           "Unknown binaries and unreviewed protected-SHA source collisions are blocking, never silently omitted.",
                           "Local-runtime review does not permit backup; backup requires its separate exact preapproved source review.",
                           "No source review changes the data policy or authorizes a remote action."]}
    value["manifest_digest"] = sha(canonical(value))
    return value


def _write_new(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def verify_snapshot(destination, plan):
    """Verify explicit bytes and reject unlisted files without reading them."""
    root = Path(destination).absolute()
    expected = {relative(item["path"]) for item in plan["source_files"]}
    found = set()
    for folder, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            _no_links(Path(folder) / name, root)
        found.update((Path(folder) / name).relative_to(root).as_posix() for name in files)
    allowed_metadata = {"source_manifest.json", "source_snapshot_complete.json"}
    if found - allowed_metadata != expected or "source_snapshot_failed.json" in found:
        raise FreezeError("snapshot_contains_missing_or_unlisted_files")
    for item in plan["source_files"]:
        raw = stable_bytes(root / item["path"], root)
        if sha(raw) != item["sha256"] or len(raw) != item["size_bytes"]:
            raise FreezeError("snapshot_byte_verification_failed:" + item["path"])
    return {"status": "verified", "file_count": len(expected)}


def materialize(root, destination, plan, authority):
    """Refuse any changed source/policy; retain partial directory on failure."""
    if not Path(root).is_absolute() or not Path(destination).is_absolute():
        raise FreezeError("source_and_destination_must_be_explicit_absolute_paths")
    root, destination = Path(root).absolute(), Path(destination).absolute()
    unsigned = {key: value for key, value in plan.items() if key != "manifest_digest"}
    if plan.get("schema_version") != VERSION or sha(canonical(unsigned)) != plan.get("manifest_digest"):
        raise FreezeError("manifest_digest_or_schema_mismatch")
    if plan.get("rejected") or plan.get("status") != "source_manifest_ready_for_review":
        raise FreezeError("blocked_manifest_cannot_be_materialized")
    if destination.exists() or destination == root or destination.is_relative_to(root) or root.is_relative_to(destination):
        raise FreezeError("destination_must_be_new_and_separate")
    # Check existing parent chain before creating any destination directory.
    ancestor = destination.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    _no_links(ancestor, Path(destination.anchor))
    context = authority.read()
    if _protected(destination, context):
        raise FreezeError("destination_is_protected")
    if policy_binding(context) != plan["policy_binding"]:
        raise FreezeError("current_policy_changed_replan_required")
    current = build_plan(root, context, mode=plan["mode"], audited_tests=plan["audited_tests"], local_runtime_source_review=plan.get("local_runtime_source_review"), backup_source_review=plan.get("backup_source_review"))
    if current != plan:
        raise FreezeError("source_inventory_or_bytes_changed_replan_required")
    destination.mkdir(parents=True, exist_ok=False)
    _no_links(destination, Path(destination.anchor))
    try:
        for item in plan["source_files"]:
            rel = relative(item["path"])
            raw = stable_bytes(root / rel, root)
            if sha(raw) != item["sha256"] or len(raw) != item["size_bytes"]:
                raise FreezeError("source_changed_during_copy:" + rel)
            target = destination / rel
            _write_new(target, raw)
            if sha(stable_bytes(target, destination)) != item["sha256"]:
                raise FreezeError("destination_byte_verification_failed:" + rel)
        latest = authority.read()
        if policy_binding(latest) != plan["policy_binding"] or build_plan(root, latest, mode=plan["mode"], audited_tests=plan["audited_tests"], local_runtime_source_review=plan.get("local_runtime_source_review"), backup_source_review=plan.get("backup_source_review")) != plan:
            raise FreezeError("source_or_current_policy_changed_during_copy")
        verify_snapshot(destination, plan)
        # Manifest and completion marker are written only after byte checks.
        _write_new(destination / "source_manifest.json", canonical(plan) + b"\n")
        complete = {"schema_version": VERSION, "status": "source_snapshot_verified", "manifest_digest": plan["manifest_digest"],
                    "source_manifest_sha256": sha((destination / "source_manifest.json").read_bytes()),
                    "file_count": len(plan["source_files"]), "upload_authorized": False,
                    "distribution_restriction": plan["distribution_restriction"],
                    "runtime_readiness": plan["runtime_readiness"]}
        _write_new(destination / "source_snapshot_complete.json", canonical(complete) + b"\n")
        return complete
    except BaseException as exc:
        _write_new(destination / "source_snapshot_failed.json", canonical({"status": "failed", "error_type": type(exc).__name__, "manifest_digest": plan["manifest_digest"]}) + b"\n")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "materialize"))
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("runtime", "backup"), default="runtime")
    parser.add_argument("--audited-tests", type=Path)
    parser.add_argument("--local-runtime-review", type=Path)
    parser.add_argument("--backup-source-review", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args(argv)
    if Path(__file__).resolve().parents[2] != SOURCE_ROOT.resolve(strict=True):
        raise FreezeError("CLI_source_is_not_the_authorized_worktree")
    sys.path.insert(0, str(PRODUCT_ROOT))
    from modules.data_admission import PolicyAuthority
    authority = PolicyAuthority(args.policy)
    if args.action == "plan":
        reviewed = json.loads(args.audited_tests.read_text("utf-8-sig")) if args.audited_tests else {}
        local_review = json.loads(args.local_runtime_review.read_text("utf-8-sig")) if args.local_runtime_review else None
        backup_review = json.loads(args.backup_source_review.read_text("utf-8-sig")) if args.backup_source_review else None
        plan = build_plan(SOURCE_ROOT, authority.read(), mode=args.mode, audited_tests=reviewed, local_runtime_source_review=local_review, backup_source_review=backup_review)
        _write_new(args.manifest, canonical(plan) + b"\n")
        print(json.dumps({"status": plan["status"], **plan["summary"], "manifest_digest": plan["manifest_digest"],
                          "rejected_paths": [row["path"] for row in plan["rejected"]]}, ensure_ascii=False))
        return 2 if plan["rejected"] else 0
    if args.destination is None:
        parser.error("materialize requires --destination")
    plan = json.loads(args.manifest.read_text("utf-8-sig"))
    print(json.dumps(materialize(SOURCE_ROOT, args.destination, plan, authority), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "rejected", "error_type": type(error).__name__, "reason": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
