"""Synthetic policy/marker-file tests; explicitly supplied checker source only.

No dataset media, cached features, historical report, labels or holdout is read.
Native and acceptance interfaces run the same exclusion cases.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.data_admission import AdmissionError, AdmittedDataset, PolicyAuthority, create_receipt


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return digest(path.read_bytes())


@pytest.fixture
def fixture(tmp_path):
    configured = os.environ.get("V4_TEST_EXCLUSION_CHECKER")
    expected = os.environ.get("V4_TEST_CHECKER_SHA256")
    if not configured or not expected:
        pytest.skip("Explicit verified exclusion checker source is required")
    source = Path(configured).read_bytes()
    assert digest(source) == expected, "test source must match policy checker binding"
    core_path = tmp_path / "exclusion_core.py"
    core_path.write_bytes(source)
    core = types.ModuleType("synthetic_fixture_core")
    core.__file__ = str(core_path)
    exec(compile(source, str(core_path), "exec"), core.__dict__)
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    protected = blocked / "marker.txt"
    protected.write_bytes(b"independently constructed protected marker")
    data = tmp_path / "independent_marker.txt"
    data.write_bytes(b"independently constructed allowed marker")
    policy_path = tmp_path / "current_policy.json"
    policy = {"policy_version": 1, "protected_roots": [str(blocked)],
              "protected_sha256": [digest(protected.read_bytes())],
              "protected_dataset_ids": [core.REQUIRED_DATASET],
              "checker": {"path": str(core_path), "sha256": expected}}
    write_json(policy_path, policy)
    return types.SimpleNamespace(root=tmp_path, data=data, protected=protected,
                                 policy=policy, policy_path=policy_path, core=core,
                                 core_path=core_path, authority=PolicyAuthority(policy_path))


def manifest(f, purpose="model_selection", *, mutate=None):
    record_path = f.root / "provenance.json"
    record = {"schema_version": "data_admission.provenance.v1",
              "source_dataset_id": "synthetic_independent", "file_sha256": digest(f.data.read_bytes()),
              "ancestor_sha256": [], "source_reference": "constructed-in-test",
              "original_content_id": "marker-original", "license_reference": "synthetic-fixture-only",
              "reviewer": "synthetic-fixture-author", "reviewed_at": "2026-01-01T00:00:00+00:00",
              "usage_authorized": True, "allowed_purposes": [purpose],
              "evidence": [{"kind": "construction", "reference": "this-test-function"}],
              "source_kind": "original"}
    record_sha = write_json(record_path, record)
    item = {"path": str(f.data), "sha256": digest(f.data.read_bytes()),
            "split": "acceptance" if purpose == "acceptance" else "dev",
            "purpose": purpose, "source_dataset_id": "synthetic_independent",
            "ancestor_sha256": [], "provenance_known": True,
            "provenance_record": {"path": str(record_path), "sha256": record_sha}}
    obj = {"schema_version": "data_admission.inputs.v1", "purpose": purpose, "items": [item]}
    if purpose == "acceptance":
        obj.update(operation="read_only_inference", frozen_release={"version": "synthetic-v1", "sha256": "a" * 64})
    if mutate:
        mutate(obj)
    path = f.root / "inputs.json"
    write_json(path, obj)
    return path, obj, record_path, record


def admitted(f, purpose="model_selection"):
    path, obj, _, _ = manifest(f, purpose)
    receipt_path = f.root / "receipt.json"
    result = create_receipt(f.authority, path, receipt_path)
    assert result["status"] == "passed"
    ds = AdmittedDataset(f.authority, path, receipt_path, expected_purpose=purpose,
                         expected_frozen_release=obj.get("frozen_release"))
    return ds, path, obj, receipt_path, result


@pytest.mark.parametrize("purpose", ["model_selection", "acceptance"])
def test_explicit_verified_snapshot_and_formal_acceptance(fixture, purpose, monkeypatch):
    f = fixture
    monkeypatch.setattr(Path, "rglob", lambda *a, **k: pytest.fail("directory scan forbidden"))
    monkeypatch.setattr(Path, "glob", lambda *a, **k: pytest.fail("directory scan forbidden"))
    ds, path, obj, receipt_path, result = admitted(f, purpose)
    snapshot = ds.read_snapshot(0)
    assert snapshot.data == b"independently constructed allowed marker"
    assert isinstance(snapshot.data, bytes)
    assert snapshot.purpose == purpose
    assert json.loads(path.read_text())["items"][0]["split"] == obj["items"][0]["split"]
    assert result["bindings"]["inputs_sha256"] == digest(path.read_bytes())
    assert result["bindings"]["checker_sha256"] == digest(f.core_path.read_bytes())
    assert len(result["bindings"]["adapter_sha256"]) == 64
    if purpose == "acceptance":
        assert result["bindings"]["interface_version"] == "data_admission.acceptance.v1"
        assert result["active_splits"] == ["acceptance"]
        assert "acceptance" not in result["native_core_purposes"]
        assert "acceptance" not in f.core.SPLITS  # original checker untouched


@pytest.mark.parametrize("purpose", ["model_selection", "acceptance"])
@pytest.mark.parametrize("case", ["protected_path", "protected_copy", "ancestor", "dataset", "unknown",
                                  "missing_source_record", "source_record_hash", "cache_copy"])
def test_same_exclusion_core_cases_for_native_and_acceptance(fixture, purpose, case):
    f = fixture
    path, obj, record_path, record = manifest(f, purpose)
    item = obj["items"][0]
    if case == "protected_path":
        item["path"], item["sha256"] = str(f.protected), digest(f.protected.read_bytes())
    elif case in {"protected_copy", "cache_copy"}:
        f.data.write_bytes(f.protected.read_bytes())
        item["sha256"] = digest(f.data.read_bytes())
    elif case == "ancestor":
        item["ancestor_sha256"] = [digest(f.protected.read_bytes())]
    elif case == "dataset":
        item["source_dataset_id"] = f.core.REQUIRED_DATASET
    elif case == "unknown":
        item["provenance_known"] = False
    elif case == "missing_source_record":
        item.pop("provenance_record")
    elif case == "source_record_hash":
        item["provenance_record"]["sha256"] = "f" * 64
    write_json(path, obj)
    receipt_path = f.root / "rejected.json"
    assert create_receipt(f.authority, path, receipt_path)["status"] == "rejected"
    with pytest.raises(AdmissionError):
        AdmittedDataset(f.authority, path, receipt_path, expected_purpose=purpose,
                        expected_frozen_release=obj.get("frozen_release"))


@pytest.mark.parametrize("field,value", [("split", "dev"), ("purpose", "model_selection")])
def test_acceptance_cannot_disguise_item_role(fixture, field, value):
    f = fixture
    path, obj, _, _ = manifest(f, "acceptance")
    obj["items"][0][field] = value
    write_json(path, obj)
    result = create_receipt(f.authority, path, f.root / "receipt.json")
    assert result["status"] == "rejected"
    assert obj["purpose"] == "acceptance"


@pytest.mark.parametrize("change", ["missing", "rejected", "purpose", "policy", "checker", "file", "manifest", "provenance", "forged_item"])
def test_loading_refuses_stale_missing_or_modified_bindings(fixture, change):
    f = fixture
    ds, path, obj, receipt_path, result = admitted(f)
    if change == "missing":
        receipt_path.unlink()
    elif change == "rejected":
        result["status"] = "rejected"
        write_json(receipt_path, result)
    elif change == "purpose":
        ds.expected_purpose = "calibration"
    elif change == "policy":
        f.policy["policy_version"] = 2
        write_json(f.policy_path, f.policy)
    elif change == "checker":
        with f.core_path.open("ab") as stream:
            stream.write(b"\n# changed source\n")
    elif change == "file":
        f.data.write_bytes(b"changed independent bytes")
    elif change == "manifest":
        obj["items"][0]["split"] = "train"
        write_json(path, obj)
    elif change == "provenance":
        Path(obj["items"][0]["provenance_record"]["path"]).write_text("{}")
    elif change == "forged_item":
        result["items"][0]["actual_sha256"] = "f" * 64
        write_json(receipt_path, result)
    with pytest.raises((AdmissionError, OSError)):
        ds.read_snapshot(0)


def test_acceptance_requires_independently_supplied_frozen_release(fixture):
    f = fixture
    _, path, obj, receipt_path, _ = admitted(f, "acceptance")
    for frozen in (None, {"version": "other-version", "sha256": "b" * 64}):
        with pytest.raises(AdmissionError):
            AdmittedDataset(f.authority, path, receipt_path, expected_purpose="acceptance", expected_frozen_release=frozen)


@pytest.mark.parametrize("target", ["manifest", "provenance", "receipt", "policy"])
def test_duplicate_json_keys_are_rejected(fixture, target):
    f = fixture
    ds, path, obj, receipt_path, _ = admitted(f)
    chosen = {"manifest": path, "receipt": receipt_path, "policy": f.policy_path,
              "provenance": Path(obj["items"][0]["provenance_record"]["path"])}[target]
    chosen.write_text('{"purpose":"acceptance","purpose":"model_selection"}', encoding="utf-8")
    with pytest.raises((AdmissionError, ValueError)):
        ds.read_snapshot(0)


def test_snapshot_budget_index_and_receipt_non_overwrite(fixture):
    f = fixture
    ds, path, _, receipt_path, _ = admitted(f)
    with pytest.raises(AdmissionError, match="budget"):
        ds.read_snapshot(0, maximum_bytes=2)
    for index in (-1, 1, True, "0"):
        with pytest.raises(AdmissionError):
            ds.read_snapshot(index)
    with pytest.raises(AdmissionError, match="new"):
        create_receipt(f.authority, path, receipt_path)
    with pytest.raises(AdmissionError, match="protected"):
        create_receipt(f.authority, path, f.protected.parent / "receipt.json")


def test_symlink_target_change_even_with_identical_bytes_is_rejected(fixture):
    f = fixture
    link = f.root / "link.txt"
    other = f.root / "same_bytes_other.txt"
    other.write_bytes(f.data.read_bytes())
    try:
        link.symlink_to(f.data)
    except OSError as exc:
        pytest.skip("OS disallows this synthetic symlink fixture: " + str(exc))
    path, obj, _, _ = manifest(f)
    obj["items"][0]["path"] = str(link)
    write_json(path, obj)
    receipt_path = f.root / "receipt.json"
    assert create_receipt(f.authority, path, receipt_path)["status"] == "passed"
    ds = AdmittedDataset(f.authority, path, receipt_path, expected_purpose="model_selection")
    link.unlink()
    link.symlink_to(other)
    with pytest.raises(AdmissionError):
        ds.read_snapshot(0)


@pytest.mark.parametrize("protected_target", [False, True])
def test_resolved_target_recheck_without_os_symlink_privilege(fixture, monkeypatch, protected_target):
    """Independently simulate path resolution; not a claim of real OS link QA."""
    f = fixture
    link = f.root / "simulated-link.txt"
    other = f.root / "same-bytes-new-target.txt"
    other.write_bytes(f.data.read_bytes())
    target = [f.data]
    original = Path.resolve
    def resolve(path, *args, **kwargs):
        return original(target[0], *args, **kwargs) if path == link else original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    path, obj, _, _ = manifest(f)
    obj["items"][0]["path"] = str(link)
    write_json(path, obj)
    receipt_path = f.root / "receipt.json"
    assert create_receipt(f.authority, path, receipt_path)["status"] == "passed"
    ds = AdmittedDataset(f.authority, path, receipt_path, expected_purpose="model_selection")
    target[0] = f.protected if protected_target else other
    with pytest.raises(AdmissionError):
        ds.read_snapshot(0)


def test_duplicate_provenance_json_rejected_even_when_claimed_hash_matches(fixture):
    f = fixture
    path, obj, record_path, _ = manifest(f)
    record_path.write_text('{"schema_version":"data_admission.provenance.v1","schema_version":"data_admission.provenance.v1"}')
    obj["items"][0]["provenance_record"]["sha256"] = digest(record_path.read_bytes())
    write_json(path, obj)
    result = create_receipt(f.authority, path, f.root / "receipt.json")
    assert result["status"] == "rejected"
    assert "duplicate_json_key" in str(result["items"][0]["reasons"])
