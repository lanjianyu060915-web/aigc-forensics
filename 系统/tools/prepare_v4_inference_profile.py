"""Prepare the frozen baseline for explicitly admitted independent dev videos.

Only metadata and hashes are read. No decoder, model, fitting or inference is
invoked. This tool does not freeze/approve a code release or authorize acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PRODUCT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PRODUCT_ROOT))

from artifact_io import atomic_write_json
from contracts.execution_profile_v4 import (audit_execution_profile, build_independent_profile,
    collect_asset_bindings, collect_runtime_code_bindings, execution_binding)
from modules.data_admission import AdmittedDataset, PolicyAuthority


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "asset-root", "independent-root", "interpreter", "query-manifest", "query-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    authority = PolicyAuthority(args.policy)
    context = authority.read()
    base_path = PRODUCT_ROOT / "config.json"
    if context["core"].within(base_path, context["roots"]) or context["core"].within(base_path.resolve(), context["roots"]):
        parser.error("product baseline resolves inside protected data root")
    # This fixed code baseline is expressly authorized for configuration audit.
    # Its protected-byte status is retained; it is never a dataset input.
    base_bytes = base_path.read_bytes()
    base = json.loads(base_bytes.decode("utf-8-sig"))
    config = build_independent_profile(base, product_root=PRODUCT_ROOT, asset_root=args.asset_root,
        independent_root=args.independent_root, interpreter=args.interpreter)
    for module in ("video", "deepfake", "general_aigc"):
        config[module]["enabled"] = True
    config["runtime"]["persistent_workers_enabled"] = base.get("runtime", {}).get("persistent_workers_enabled", True)
    config["execution_profile"].update(mode="independent_inference", asset_verification="pending_hash_observation",
        query_admission={"manifest_path": str(args.query_manifest.absolute()), "receipt_path": str(args.query_receipt.absolute()),
                         "purpose": "model_selection"}, code_release_verification="pending_coordinator_freeze",
        evaluation_scope="development_only_not_formal_acceptance")
    audit_execution_profile(config, authority=authority, config_path=args.output)
    # Reject missing/stale provenance or gate receipt before even hashing assets.
    AdmittedDataset(authority, args.query_manifest, args.query_receipt, expected_purpose="model_selection")
    bindings = collect_asset_bindings(config, authority=authority)
    config["execution_profile"].update(model_bindings=bindings, runtime_code_bindings=collect_runtime_code_bindings(config, authority=authority),
        asset_verification="verified_required_asset_closure")
    audit = audit_execution_profile(config, authority=authority, config_path=args.output, for_inference=True)
    receipt = args.output.with_suffix(".audit.json")
    if args.output.exists() or receipt.exists():
        parser.error("output and audit must both be new; refusal to overwrite")
    code_files = [Path(__file__), PRODUCT_ROOT / "contracts/execution_profile_v4.py"]
    observed_code = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "verification": "observed_source_bytes_not_frozen_release"} for path in code_files]
    atomic_write_json(args.output, config, role="independent_dev_profile", overwrite=False)
    atomic_write_json(receipt, {**audit, "execution_binding": execution_binding(config, input_digest=None,
                       code_bindings=observed_code, model_bindings=bindings),
        "profile_file_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "base_configuration_registered_protected": hashlib.sha256(base_bytes).hexdigest() in context["hashes"],
        "base_configuration_use": "explicit_authorized_product_configuration_only_not_dataset_input",
        "base_configuration_training_allowed": False, "code_release_verification": "pending_coordinator_freeze",
        "query_manifest_sha256": hashlib.sha256(args.query_manifest.read_bytes()).hexdigest(),
        "query_receipt_sha256": hashlib.sha256(args.query_receipt.read_bytes()).hexdigest(),
        "inference_performed": False, "model_files_copied": False, "credentials_copied": False,
        "not_formal_acceptance": True}, role="independent_dev_profile_audit", overwrite=False)
    print(json.dumps({"status": "prepared_development_profile", "profile": str(args.output), "audit": str(receipt),
        "config_digest": audit["config_digest"], "model_asset_count": len(bindings), "inference_performed": False,
        "code_release_verification": "pending_coordinator_freeze"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print("Independent profile rejected: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
