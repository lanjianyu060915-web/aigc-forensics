"""Versioned receipt adapter and immutable input snapshots.

The caller must configure the authoritative *current* policy location. A receipt
cannot choose that authority. This module does not prove attestations are true,
grant data permission, or intercept programs which bypass this API.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import types
from typing import Any
import uuid


ADAPTER_VERSION = "data_admission.v1"
ACCEPTANCE_INTERFACE = "data_admission.acceptance.v1"
MANIFEST_SCHEMA = "data_admission.inputs.v1"
RECEIPT_SCHEMA = "data_admission.receipt.v1"
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


class AdmissionError(ValueError):
    """Closed admission: callers must not catch this and load the input anyway."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


_IMPORTED_ADAPTER_SHA256 = _sha(Path(__file__).read_bytes())


def _digest(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise AdmissionError("invalid_sha256")
    return value.lower()


def _absolute(value: Any) -> Path:
    if not isinstance(value, (str, Path)) or not Path(value).is_absolute():
        raise AdmissionError("absolute_explicit_file_path_required")
    return Path(value).absolute()


def _signature(stat):
    return stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino


def _read_stable(path: Path, *, maximum: int = DEFAULT_MAX_BYTES) -> tuple[bytes, Path]:
    """Read once from one descriptor and verify both descriptor and path binding."""
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise AdmissionError("positive_snapshot_limit_required")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise AdmissionError("explicit_regular_file_required")
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > maximum:
            raise AdmissionError("snapshot_size_budget_exceeded")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > maximum or _signature(before) != _signature(after):
        raise AdmissionError("file_changed_during_snapshot")
    if path.resolve(strict=True) != resolved or _signature(resolved.stat()) != _signature(after):
        raise AdmissionError("file_or_link_target_changed")
    return raw, resolved


def _json(raw: bytes) -> Any:
    def unique(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise AdmissionError("duplicate_json_key:" + key)
            output[key] = value
        return output
    try:
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(AdmissionError("nonfinite_json_number")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AdmissionError("invalid_json") from exc


@dataclass(frozen=True)
class PolicyAuthority:
    latest_policy_path: Path
    minimum_policy_version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "latest_policy_path", _absolute(self.latest_policy_path))

    def read(self) -> dict[str, Any]:
        raw, resolved = _read_stable(self.latest_policy_path)
        policy = _json(raw)
        if not isinstance(policy, dict):
            raise AdmissionError("policy_not_object")
        version = policy.get("policy_version")
        if isinstance(version, bool) or not isinstance(version, int) or version < self.minimum_policy_version:
            raise AdmissionError("policy_version_missing_or_below_authority_floor")
        checker = policy.get("checker")
        if not isinstance(checker, dict):
            raise AdmissionError("policy_checker_binding_missing")
        checker_path = _absolute(checker.get("path"))
        source, checker_resolved = _read_stable(checker_path)
        expected = _digest(checker.get("sha256"))
        if _sha(source) != expected:
            raise AdmissionError("policy_checker_digest_mismatch")
        # Execute exactly the verified bytes, not a second path-based import.
        core = types.ModuleType("_verified_exclusion_core")
        core.__file__ = str(checker_resolved)
        exec(compile(source, str(checker_resolved), "exec"), core.__dict__)
        for symbol in ("policy_values", "check_item", "within", "PURPOSES", "SPLITS"):
            if not hasattr(core, symbol):
                raise AdmissionError("unsupported_exclusion_core_interface")
        roots, hashes, datasets = core.policy_values(policy)
        adapter_raw, _ = _read_stable(Path(__file__).resolve())
        if _sha(adapter_raw) != _IMPORTED_ADAPTER_SHA256:
            raise AdmissionError("adapter_changed_since_import")
        return {"policy": policy, "core": core, "roots": roots, "hashes": hashes,
                "datasets": datasets, "bindings": {
                    "policy_path": str(self.latest_policy_path), "policy_resolved_path": str(resolved),
                    "policy_sha256": _sha(raw), "policy_version": version,
                    "checker_path": str(checker_path), "checker_resolved_path": str(checker_resolved),
                    "checker_sha256": expected, "adapter_version": ADAPTER_VERSION,
                    "adapter_sha256": _sha(adapter_raw)}}


def _check_path(path: Path, context):
    resolved = path.resolve(strict=True)
    if context["core"].within(path, context["roots"]) or context["core"].within(resolved, context["roots"]):
        raise AdmissionError("protected_path_or_resolved_target")
    return resolved


def _checked_snapshot(path, expected, context, *, maximum=DEFAULT_MAX_BYTES):
    _check_path(path, context)
    raw, resolved = _read_stable(path, maximum=maximum)
    actual = _sha(raw)
    if actual in context["hashes"]:
        raise AdmissionError("protected_actual_sha256")
    if actual != _digest(expected):
        raise AdmissionError("snapshot_digest_mismatch")
    _check_path(path, context)
    return raw, resolved


def _known_text(value: Any, core, field: str) -> str:
    try:
        core.source_id(value)
    except (ValueError, TypeError) as exc:
        raise AdmissionError("provenance_missing_or_unknown:" + field) from exc
    return value


def _provenance(item, purpose, context):
    reference = item.get("provenance_record")
    if not isinstance(reference, dict):
        raise AdmissionError("explicit_provenance_record_required")
    path = _absolute(reference.get("path"))
    raw, resolved = _checked_snapshot(path, reference.get("sha256"), context)
    record = _json(raw)
    core = context["core"]
    if not isinstance(record, dict) or record.get("schema_version") != "data_admission.provenance.v1":
        raise AdmissionError("provenance_schema_mismatch")
    if core.source_id(record.get("source_dataset_id")) != core.source_id(item.get("source_dataset_id")):
        raise AdmissionError("provenance_dataset_mismatch")
    if _digest(record.get("file_sha256")) != _digest(item.get("sha256")):
        raise AdmissionError("provenance_file_digest_mismatch")
    if not isinstance(record.get("ancestor_sha256"), list) or sorted(map(_digest, record["ancestor_sha256"])) != sorted(map(_digest, item["ancestor_sha256"])):
        raise AdmissionError("provenance_ancestor_mismatch")
    for field in ("source_reference", "original_content_id", "license_reference", "reviewer", "reviewed_at"):
        _known_text(record.get(field), core, field)
    try:
        reviewed = datetime.fromisoformat(record["reviewed_at"].replace("Z", "+00:00"))
        if reviewed.tzinfo is None:
            raise ValueError("timezone required")
    except ValueError as exc:
        raise AdmissionError("invalid_provenance_review_time") from exc
    allowed = record.get("allowed_purposes")
    if record.get("usage_authorized") is not True or not isinstance(allowed, list) or purpose not in allowed:
        raise AdmissionError("provenance_purpose_not_authorized")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise AdmissionError("source_evidence_references_required")
    for entry in evidence:
        if not isinstance(entry, dict):
            raise AdmissionError("invalid_source_evidence_reference")
        _known_text(entry.get("kind"), core, "evidence.kind")
        _known_text(entry.get("reference"), core, "evidence.reference")
    if record.get("source_kind") not in {"original", "derived"}:
        raise AdmissionError("source_kind_required")
    if record["source_kind"] == "derived":
        if not item["ancestor_sha256"] or not isinstance(record.get("derivation_parameters"), dict):
            raise AdmissionError("derivation_chain_required")
        _known_text(record.get("derivation_tool"), core, "derivation_tool")
    return {"path": str(path), "resolved_path": str(resolved), "sha256": _sha(raw)}


def _validate(authority: PolicyAuthority, manifest_path: Path) -> dict[str, Any]:
    context = authority.read()
    _check_path(manifest_path, context)
    manifest_raw, manifest_resolved = _read_stable(manifest_path)
    if _sha(manifest_raw) in context["hashes"]:
        raise AdmissionError("protected_manifest_digest")
    manifest = _json(manifest_raw)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise AdmissionError("manifest_schema_mismatch")
    core = context["core"]
    purpose = manifest.get("purpose")
    interface = "data_admission.native.v1"
    check_item = core.check_item
    frozen = None
    if purpose == "acceptance":
        interface = ACCEPTANCE_INTERFACE
        if manifest.get("operation") != "read_only_inference":
            raise AdmissionError("acceptance_requires_read_only_inference")
        frozen = manifest.get("frozen_release")
        if not isinstance(frozen, dict):
            raise AdmissionError("acceptance_frozen_release_required")
        _known_text(frozen.get("version"), core, "frozen_release.version")
        _digest(frozen.get("sha256"))
        # A versioned interface extension: identical exclusion function bytecode,
        # new *explicit* split vocabulary. Original manifest bytes stay untouched.
        globals_copy = dict(core.check_item.__globals__)
        globals_copy["SPLITS"] = frozenset({"acceptance"})
        check_item = types.FunctionType(core.check_item.__code__, globals_copy,
                                        "acceptance_check_item_v1", core.check_item.__defaults__)
    elif purpose not in core.PURPOSES:
        raise AdmissionError("unsupported_purpose")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise AdmissionError("nonempty_explicit_manifest_required")
    results, provenance = [], []
    for index, item in enumerate(items):
        result = check_item(item, index, purpose, context["roots"], context["hashes"], context["datasets"])
        if purpose == "acceptance":
            result["reasons"] = ["split_not_acceptance" if reason == "split_not_train_or_dev" else reason
                                 for reason in result["reasons"]]
        results.append(result)
        if result["status"] == "passed":
            try:
                provenance.append(_provenance(item, purpose, context))
            except (AdmissionError, OSError, ValueError, TypeError) as exc:
                result["status"] = "rejected"
                result["reasons"].append("provenance_rejected:" + str(exc))
    bindings = {**context["bindings"], "inputs_path": str(manifest_path),
                "inputs_resolved_path": str(manifest_resolved), "inputs_sha256": _sha(manifest_raw),
                "interface_version": interface, "provenance_records": provenance}
    # Authority and explicit manifest must still be current after byte hashing.
    if authority.read()["bindings"] != context["bindings"]:
        raise AdmissionError("policy_or_checker_changed_during_check")
    final_raw, final_path = _read_stable(manifest_path)
    if _sha(final_raw) != bindings["inputs_sha256"] or final_path != manifest_resolved:
        raise AdmissionError("inputs_changed_during_check")
    return {"schema_version": RECEIPT_SCHEMA, "status": "passed" if all(r["status"] == "passed" for r in results) else "rejected",
            "purpose": purpose, "operation": manifest.get("operation"), "frozen_release": frozen,
            "bindings": bindings, "native_core_purposes": sorted(core.PURPOSES),
            "native_core_splits": sorted(core.SPLITS),
            "active_splits": ["acceptance"] if purpose == "acceptance" else sorted(core.SPLITS),
            "items": results, "counts": {"items": len(results), "passed": sum(r["status"] == "passed" for r in results),
                                          "rejected": sum(r["status"] != "passed" for r in results)},
            "directory_scan_executed": False, "media_decoded": False, "training_executed": False,
            "limit": "Attestations require real human/source review. This wrapper cannot prove their truth or block bypassing external programs."}


def create_receipt(authority: PolicyAuthority, manifest_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    manifest_path, output_path = _absolute(manifest_path), _absolute(output_path)
    context = authority.read()
    output_resolved = output_path.resolve(strict=False)
    if context["core"].within(output_path, context["roots"]) or context["core"].within(output_resolved, context["roots"]):
        raise AdmissionError("receipt_cannot_be_in_protected_roots")
    if output_path.exists() or output_path.is_symlink():
        raise AdmissionError("receipt_must_be_new")
    try:
        receipt = _validate(authority, manifest_path)
    except (AdmissionError, OSError, ValueError, TypeError) as exc:
        receipt = {"schema_version": RECEIPT_SCHEMA, "status": "rejected", "global_reasons": [type(exc).__name__ + ":" + str(exc)],
                   "bindings": context["bindings"], "directory_scan_executed": False,
                   "media_decoded": False, "training_executed": False}
    receipt["created_at"] = datetime.now(timezone.utc).isoformat()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve(strict=False) != output_resolved:
        raise AdmissionError("receipt_parent_changed_before_write")
    temporary = output_path.with_name("." + output_path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output_path)  # exclusive atomic publication, never replace a receipt
    finally:
        temporary.unlink(missing_ok=True)
    return receipt


@dataclass(frozen=True)
class DataSnapshot:
    data: bytes
    sha256: str
    source_dataset_id: str
    purpose: str
    split: str
    frozen_release: dict[str, Any] | None


class AdmittedDataset:
    """Only explicit indexed byte snapshots; no glob, directory or mutable path API."""
    def __init__(self, authority: PolicyAuthority, manifest_path: str | Path, receipt_path: str | Path,
                 *, expected_purpose: str, expected_frozen_release: dict[str, Any] | None = None):
        self.authority = authority
        self.manifest_path = _absolute(manifest_path)
        self.receipt_path = _absolute(receipt_path)
        self.expected_purpose = expected_purpose
        self.expected_frozen_release = expected_frozen_release
        self._verify()

    def _verify(self):
        raw, resolved = _read_stable(self.receipt_path)
        receipt = _json(raw)
        if not isinstance(receipt, dict) or receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("status") != "passed":
            raise AdmissionError("missing_valid_passed_receipt")
        if receipt.get("purpose") != self.expected_purpose:
            raise AdmissionError("purpose_mismatch")
        if self.expected_purpose == "acceptance":
            if not self.expected_frozen_release or receipt.get("frozen_release") != self.expected_frozen_release:
                raise AdmissionError("frozen_release_mismatch")
        current = _validate(self.authority, self.manifest_path)
        if current["status"] != "passed":
            raise AdmissionError("live_exclusion_recheck_rejected")
        for key in ("bindings", "purpose", "operation", "frozen_release", "items", "counts", "active_splits"):
            if receipt.get(key) != current.get(key):
                raise AdmissionError("receipt_binding_mismatch:" + key)
        again, again_resolved = _read_stable(self.receipt_path)
        if _sha(raw) != _sha(again) or resolved != again_resolved:
            raise AdmissionError("receipt_changed_during_check")
        return current

    def read_snapshot(self, index: int, *, maximum_bytes: int = DEFAULT_MAX_BYTES) -> DataSnapshot:
        receipt = self._verify()
        manifest_raw, _ = _read_stable(self.manifest_path)
        if _sha(manifest_raw) != receipt["bindings"]["inputs_sha256"]:
            raise AdmissionError("inputs_changed_before_load")
        manifest = _json(manifest_raw)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(manifest["items"]):
            raise AdmissionError("explicit_manifest_index_required")
        item = manifest["items"][index]
        context = self.authority.read()
        for key, value in context["bindings"].items():
            if receipt["bindings"].get(key) != value:
                raise AdmissionError("authority_changed_before_load")
        raw, resolved = _checked_snapshot(_absolute(item["path"]), item["sha256"], context, maximum=maximum_bytes)
        if str(resolved) != receipt["items"][index]["resolved_path"]:
            raise AdmissionError("input_link_target_changed_before_load")
        # Refresh all bindings after the final read; return immutable bytes only.
        self._verify()
        return DataSnapshot(raw, _sha(raw), item["source_dataset_id"], self.expected_purpose,
                            item["split"], receipt.get("frozen_release"))
