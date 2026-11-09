"""New source/text fixtures only; no protected material or real keys are read."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import freeze_source_v4 as freeze


def write(root, name, raw=b"value = 1\n"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def context(tmp_path, hashes=()):
    protected = tmp_path / "new-empty-protected-root"
    def within(path, roots):
        return any(Path(path).absolute().is_relative_to(Path(root).absolute()) for root in roots)
    return {"roots": [protected], "hashes": set(hashes), "core": SimpleNamespace(within=within),
            "bindings": {"policy_sha256": "a" * 64, "policy_version": 1, "checker_sha256": "b" * 64}}


class Authority:
    def __init__(self, value):
        self.value = value
    def read(self):
        return self.value


def test_runtime_includes_explicit_recovery_entrypoint(tmp_path):
    root = tmp_path / "new-source"
    write(root, "系统/tools/recover_task_v4.py")
    plan = freeze.build_plan(root, context(tmp_path))
    assert plan["status"] == "source_manifest_ready_for_review"
    assert [row["path"] for row in plan["source_files"]] == ["系统/tools/recover_task_v4.py"]


def test_runtime_plan_contains_only_explicit_source_and_never_reads_pruned_data(tmp_path, monkeypatch):
    root = tmp_path / "new-source"
    write(root, "系统/launcher.py")
    write(root, "系统/engine/aigc_engine/models/core.py")
    write(root, "系统/reporting/prompts/writer_v1.md", b"new synthetic prompt")
    for name in ("系统/config.json", "系统/KNOWN_ISSUES.md", "系统/reporting/reference_statistics.json",
                 "系统/decision/band_evidence.json", "系统/models/weights.bin", "系统/tests/test_new.py",
                 "系统/frontend/mock_results/result.json", "系统/sample_data/query.png", "AGENTS.md"):
        write(root, name, b"MUST NOT READ")
    original = freeze.stable_bytes
    read_names = []
    def audit_read(path, base):
        read_names.append(path.relative_to(base).as_posix())
        return original(path, base)
    monkeypatch.setattr(freeze, "stable_bytes", audit_read)
    plan = freeze.build_plan(root, context(tmp_path))
    assert plan["status"] == "source_manifest_ready_for_review"
    assert len(plan["source_files"]) == 3
    assert all("MUST NOT READ" not in (root / name).read_text() for name in read_names)
    assert any(row["path"] == "系统/models" and row["kind"] == "directory" for row in plan["excluded"])


def test_backup_tests_require_explicit_review_and_unchanged_hash(tmp_path):
    root = tmp_path / "new-source"
    path = write(root, "系统/tests/test_new.py", b"def test_independent():\n    assert True\n")
    write(root, "系统/tests/test_unreviewed.py")
    reviewed = {"系统/tests/test_new.py": {"reviewed_synthetic": True, "sha256": freeze.sha(path.read_bytes())}}
    plan = freeze.build_plan(root, context(tmp_path), mode="backup", audited_tests=reviewed)
    assert [row["path"] for row in plan["source_files"]] == ["系统/tests/test_new.py"]
    path.write_text("changed = True\n", encoding="utf-8")
    assert freeze.build_plan(root, context(tmp_path), mode="backup", audited_tests=reviewed)["status"] == "blocked"


def test_protected_source_sha_is_separately_blocking_not_silently_removed(tmp_path):
    root = tmp_path / "new-source"
    path = write(root, "系统/launcher.py")
    plan = freeze.build_plan(root, context(tmp_path, [freeze.sha(path.read_bytes())]))
    assert plan["status"] == "blocked"
    assert plan["rejected"][0]["path"] == "系统/launcher.py"
    assert {item["reason"] for item in plan["rejected"][0]["findings"]} == {"protected_sha256_collision_no_upload"}


def reviewed_source(name, file_sha, ctx):
    return {"schema_version": "source_overlap_review.v1", "purpose": "local_runtime_only", "upload_prohibited": True,
            "policy_binding": freeze.policy_binding(ctx), "provenance_review_sha256": "c" * 64,
            "files": {name: {"sha256": file_sha, "reviewed_generic_source": True, "upload_prohibited": True}}}


def test_exact_review_allows_only_local_runtime_and_preserves_policy_collision(tmp_path):
    root = tmp_path / "new-source"
    name = "系统/launcher.py"
    path = write(root, name)
    file_sha = freeze.sha(path.read_bytes())
    ctx = context(tmp_path, [file_sha])
    review = reviewed_source(name, file_sha, ctx)
    plan = freeze.build_plan(root, ctx, local_runtime_source_review=review)
    assert plan["status"] == "source_manifest_ready_for_review"
    assert plan["distribution_restriction"] == "local_runtime_only_no_upload"
    assert plan["source_files"][0]["protected_source_overlap"]["upload_prohibited"] is True
    assert file_sha in ctx["hashes"]
    assert freeze.build_plan(root, ctx, mode="backup", local_runtime_source_review=review)["status"] == "blocked"
    destination = tmp_path / "new-local-snapshot"
    assert freeze.materialize(root, destination, plan, Authority(ctx))["distribution_restriction"] == "local_runtime_only_no_upload"


@pytest.mark.parametrize("mutation", ["path", "hash", "policy", "generic_secret"])
def test_local_review_is_not_a_wildcard_or_secret_bypass(tmp_path, mutation):
    root = tmp_path / "new-source"
    name = "系统/launcher.py"
    path = write(root, name)
    if mutation == "generic_secret":
        path.write_text("api_key = 'sk-" + "A1" * 20 + "'\n", encoding="utf-8")
    file_sha = freeze.sha(path.read_bytes())
    ctx = context(tmp_path, [file_sha])
    review = reviewed_source(name, file_sha, ctx)
    if mutation == "path":
        review["files"]["系统/other.py"] = review["files"].pop(name)
    elif mutation == "hash":
        review["files"][name]["sha256"] = "d" * 64
    elif mutation == "policy":
        ctx["bindings"]["policy_version"] = 2
        with pytest.raises(freeze.FreezeError, match="current_policy"):
            freeze.build_plan(root, ctx, local_runtime_source_review=review)
        return
    assert freeze.build_plan(root, ctx, local_runtime_source_review=review)["status"] == "blocked"


def test_backup_review_is_separate_exact_scope_and_not_a_data_exception(tmp_path, monkeypatch):
    root = tmp_path / "new-source"
    name = "系统/launcher.py"
    path = write(root, name)
    file_hash = freeze.sha(path.read_bytes())
    ctx = context(tmp_path, [file_hash])
    # Independently constructed stand-in for the seven coordinator-reviewed
    # source hashes; no real code bytes are loaded by this unit test.
    monkeypatch.setattr(freeze, "BACKUP_SOURCE_REVIEW_SCOPE", {name: file_hash})
    review = {"schema_version": "source_overlap_backup_review.v1", "purpose": "source_only_github_backup",
              "remote_action_authorized": False, "policy_binding": freeze.policy_binding(ctx),
              "provenance_review_sha256": "c" * 64,
              "files": {name: {"sha256": file_hash, "reviewed_generic_source": True, "training_allowed": False}}}
    plan = freeze.build_plan(root, ctx, mode="backup", backup_source_review=review)
    assert plan["status"] == "source_manifest_ready_for_review"
    assert plan["protected_source_overlaps"][0]["disposition"] == "reviewed_source_only_backup_inclusion"
    assert plan["upload_authorized"] is False and file_hash in ctx["hashes"]
    assert freeze.build_plan(root, ctx, mode="runtime", backup_source_review=review)["status"] == "blocked"
    review["files"]["系统/other.py"] = deepcopy(review["files"][name])
    with pytest.raises(freeze.FreezeError, match="exact_preapproved_paths"):
        freeze.build_plan(root, ctx, mode="backup", backup_source_review=review)


def test_review_records_cannot_smuggle_extra_content_into_manifest(tmp_path):
    root = tmp_path / "new-source"
    name = "系统/launcher.py"
    path = write(root, name)
    file_hash = freeze.sha(path.read_bytes())
    ctx = context(tmp_path, [file_hash])
    review = reviewed_source(name, file_hash, ctx)
    review["unexpected_private_text"] = "must not be included"
    with pytest.raises(freeze.FreezeError):
        freeze.build_plan(root, ctx, local_runtime_source_review=review)


def test_new_destination_parents_are_checked_and_created(tmp_path):
    root = tmp_path / "new-source"
    write(root, "系统/launcher.py")
    ctx = context(tmp_path)
    plan = freeze.build_plan(root, ctx)
    destination = tmp_path / "new-parent" / "new-release"
    assert freeze.materialize(root, destination, plan, Authority(ctx))["status"] == "source_snapshot_verified"


@pytest.mark.parametrize("name,raw,reason", [
    ("系统/launcher.py", b"value = b'\x00'", "binary_control_byte_in_text"),
    ("系统/launcher.py", b"\x89PNG\xff", "unknown_binary_or_non_utf8_text"),
    ("系统/blob.bin", b"opaque", "unknown_binary_not_admitted"),
    ("系统/assets/fonts/LXGWWenKai-Regular.ttf", b"\x00\x01\x00\x00fake-font", "licensed_font_resource_hash_mismatch")])
def test_unknown_binary_and_replaced_font_are_rejected(tmp_path, name, raw, reason):
    root = tmp_path / "new-source"
    write(root, name, raw)
    plan = freeze.build_plan(root, context(tmp_path))
    assert plan["status"] == "blocked"
    item = plan["rejected"][0]
    assert reason in [item.get("reason"), *[finding["reason"] for finding in item.get("findings", [])]]


def test_secret_and_private_paths_are_reported_without_echoing_values(tmp_path):
    root = tmp_path / "new-source"
    ctx = context(tmp_path)
    fake_token = "sk-" + "A1" * 20
    write(root, "系统/launcher.py", ("api_key = '" + fake_token + "'\nprivate_path = '" + str(ctx["roots"][0]).replace("\\", "/") + "/file'\n").encode())
    plan = freeze.build_plan(root, ctx)
    assert plan["status"] == "blocked"
    findings = plan["rejected"][0]["findings"]
    assert {value["reason"] for value in findings} >= {"provider_token", "literal_protected_path"}
    assert fake_token not in str(plan)
    assert all("line" in value for value in findings)


def test_safe_placeholder_and_sibling_path_do_not_match_protected_root(tmp_path):
    ctx = context(tmp_path)
    sibling = str(ctx["roots"][0]) + "-source"
    file_type, findings = freeze.scan_bytes("系统/launcher.py", ("api_key = 'YOUR_API_KEY'\npath = '" + sibling.replace("\\", "/") + "'\n").encode(), ctx["roots"])
    assert file_type == "utf8_text" and not findings


def test_materialize_new_directory_verifies_all_bytes_and_refuses_overwrite(tmp_path):
    root = tmp_path / "new-source"
    write(root, "系统/launcher.py")
    ctx = context(tmp_path)
    plan = freeze.build_plan(root, ctx)
    destination = tmp_path / "new-snapshot"
    result = freeze.materialize(root, destination, plan, Authority(ctx))
    assert result["status"] == "source_snapshot_verified"
    assert freeze.verify_snapshot(destination, plan)["file_count"] == 1
    assert (destination / "source_manifest.json").is_file()
    with pytest.raises(freeze.FreezeError, match="new_and_separate"):
        freeze.materialize(root, destination, plan, Authority(ctx))
    write(destination, "unlisted.bin", b"unknown")
    with pytest.raises(freeze.FreezeError, match="unlisted"):
        freeze.verify_snapshot(destination, plan)


@pytest.mark.parametrize("mutation", ["bytes", "new_source", "policy", "manifest"])
def test_changed_source_inventory_policy_or_manifest_refuses_copy(tmp_path, mutation):
    root = tmp_path / "new-source"
    path = write(root, "系统/launcher.py")
    ctx = context(tmp_path)
    plan = freeze.build_plan(root, ctx)
    if mutation == "bytes":
        path.write_text("changed = True\n")
    elif mutation == "new_source":
        write(root, "系统/artifact_io.py")
    elif mutation == "policy":
        ctx["bindings"]["policy_version"] = 2
    else:
        plan["source_files"][0]["size_bytes"] = 999
    destination = tmp_path / "new-snapshot"
    with pytest.raises(freeze.FreezeError):
        freeze.materialize(root, destination, plan, Authority(ctx))
    assert not destination.exists()


def test_mutation_during_copy_never_publishes_completion_marker(tmp_path, monkeypatch):
    root = tmp_path / "new-source"
    source = write(root, "系统/launcher.py")
    ctx = context(tmp_path)
    plan = freeze.build_plan(root, ctx)
    destination = tmp_path / "new-snapshot"
    original = freeze._write_new
    def mutate_after_copy(path, raw):
        original(path, raw)
        if path.name == "launcher.py":
            source.write_text("changed_during_copy = True\n")
    monkeypatch.setattr(freeze, "_write_new", mutate_after_copy)
    with pytest.raises(freeze.FreezeError, match="changed_during_copy"):
        freeze.materialize(root, destination, plan, Authority(ctx))
    assert (destination / "source_snapshot_failed.json").is_file()
    assert not (destination / "source_snapshot_complete.json").exists()


def test_blocked_plan_cannot_copy_and_paths_cannot_escape(tmp_path):
    root = tmp_path / "new-source"
    write(root, "系统/blob.bin")
    ctx = context(tmp_path)
    plan = freeze.build_plan(root, ctx)
    with pytest.raises(freeze.FreezeError, match="blocked"):
        freeze.materialize(root, tmp_path / "new-snapshot", plan, Authority(ctx))
    for bad in ("../outside", "C:/absolute", "/absolute", "a\\b", "a/../b"):
        with pytest.raises(freeze.FreezeError):
            freeze.relative(bad)
