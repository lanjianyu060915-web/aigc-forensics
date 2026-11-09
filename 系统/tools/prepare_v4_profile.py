"""Create/audit an explicit empty-library engineering profile; no model loads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PRODUCT_ROOT = Path(__file__).resolve().parents[1]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))

from artifact_io import atomic_write_json
from contracts.execution_profile_v4 import audit_execution_profile, build_independent_profile, execution_binding
from modules.data_admission import PolicyAuthority


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="Maintained authority path supplied by the operator")
    parser.add_argument("--minimum-policy-version", type=int, default=1)
    parser.add_argument("--audit-only", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--independent-root", type=Path)
    parser.add_argument("--interpreter", type=Path)
    parser.add_argument("--output", type=Path, help="New profile path within the independent root; never overwritten")
    args = parser.parse_args(argv)
    authority = PolicyAuthority(args.policy, args.minimum_policy_version)
    if args.audit_only:
        config = json.loads(args.audit_only.read_text(encoding="utf-8-sig"))
        result = audit_execution_profile(config, authority=authority, config_path=args.audit_only)
        print(json.dumps({"status": result["status"], "config_digest": result["config_digest"],
                          "model_verification": result["model_verification"], "models_loaded": False}))
        return 0
    if any(value is None for value in (args.base_config, args.asset_root, args.independent_root, args.interpreter)):
        parser.error("creation requires --base-config --asset-root --independent-root --interpreter")
    # Only an explicitly named normal product configuration is read. Never point
    # this tool at the team's profile; authority-path checks reject that input.
    context = authority.read()
    base_path = args.base_config.absolute()
    if base_path.resolve() != (PRODUCT_ROOT / "config.json").resolve():
        parser.error("only this worktree's explicitly authorized product config is accepted as a code baseline")
    if context["core"].within(base_path, context["roots"]) or context["core"].within(base_path.resolve(), context["roots"]):
        parser.error("base config is in a protected root")
    raw = base_path.read_bytes()
    base_registered_protected = hashlib.sha256(raw).hexdigest() in context["hashes"]
    base = json.loads(raw.decode("utf-8-sig"))
    config = build_independent_profile(base, product_root=PRODUCT_ROOT, asset_root=args.asset_root,
                                        independent_root=args.independent_root, interpreter=args.interpreter)
    output = args.output or args.independent_root / "profiles" / "config.v4.independent.json"
    audit = audit_execution_profile(config, authority=authority, config_path=output)
    receipt = output.with_suffix(".audit.json")
    if output.exists() or receipt.exists():
        parser.error("new profile and audit paths required; refusing to overwrite")
    directories = [args.independent_root / name for name in ("profiles", "manifests", "provenance", "receipts", "inbox", "outputs/tasks", "cache/identity", "cache/source", "cache/pair", "libraries/identity", "libraries/source")]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, config, role="independent_profile", overwrite=False)
    code_files = [Path(__file__), PRODUCT_ROOT / "contracts" / "execution_profile_v4.py"]
    binding = execution_binding(config, input_digest=None,
                                 code_bindings=[{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "verification": "observed_source_bytes"} for path in code_files])
    atomic_write_json(receipt, {**audit, "execution_binding": binding, "profile_file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                                "base_configuration_registered_protected": base_registered_protected,
                                "base_configuration_use": "explicit_authorized_product_configuration_only_not_dataset_input",
                                "base_configuration_training_allowed": False,
                                "model_files_copied": False, "secrets_copied": False, "inference_authorized_by_profile": False},
                      role="independent_profile_audit", overwrite=False)
    print(json.dumps({"status": "created_engineering_only", "profile": str(output), "audit": str(receipt),
                      "config_digest": audit["config_digest"], "models_loaded": False, "model_verification": "not_performed"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print("Profile preparation rejected: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
