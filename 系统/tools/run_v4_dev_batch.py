"""Run an explicitly admitted development manifest, serially, from frozen code.

The default development entry point never opens acceptance data. The separate
G2-bound acceptance CLI uses the same fixed-universe executor. Neither resumes old product batches, retries
an interrupted inference, selects samples by result, calls cloud, or shuts down.
The item limit pauses a fixed manifest prefix for a resource pilot; it does not
change the denominator. Continue the same batch to process the remaining items.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import uuid


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def verify_release(root, manifest_sha256):
    """Bootstrap using stdlib only before importing any frozen product code."""
    root = Path(root).resolve(strict=True)
    manifest_path = root / "source_manifest.json"
    if digest(manifest_path) != manifest_sha256:
        raise ValueError("frozen source manifest changed")
    manifest = read(manifest_path)
    if manifest.get("schema_version") != "source-freeze-v4.1" or manifest.get("rejected"):
        raise ValueError("invalid or rejected source release")
    complete = read(root / "source_snapshot_complete.json")
    if (complete.get("status") != "source_snapshot_verified"
            or complete.get("source_manifest_sha256") != manifest_sha256
            or complete.get("manifest_digest") != manifest.get("manifest_digest")):
        raise ValueError("source snapshot has no valid completion binding")
    paths = set()
    for row in manifest["source_files"]:
        relative = Path(row["path"])
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts or path.is_symlink() or path.resolve() != path.absolute():
            raise ValueError("unsafe frozen source path")
        if row["path"] in paths:
            raise ValueError("duplicate source file")
        paths.add(row["path"])
        if path.stat().st_size != row["size_bytes"] or digest(path) != row["sha256"]:
            raise ValueError("source bytes changed: " + row["path"])
    permitted = paths | {"source_manifest.json", "source_snapshot_complete.json"}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("link in frozen source snapshot")
        if path.is_file() and path.relative_to(root).as_posix() not in permitted:
            raise ValueError("unlisted file in frozen source snapshot")
    return manifest


@contextmanager
def exclusive_batch_lock(path):
    """Never auto-steal stale locks: interruption needs an explicit new attempt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "started_unix": time.time()}))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        yield
    finally:
        path.unlink()


def select_pending(items, batch, maximum_items):
    """Keep all failed/partial outcomes; never repeat an existing attempt."""
    pending = []
    for index, item in enumerate(items):
        started = batch / "items" / f"{index:04d}.started.json"
        finished = batch / "items" / f"{index:04d}.result.json"
        if finished.exists():
            record = read(finished)
            if record.get("index") != index or record.get("input_sha256") != item["sha256"]:
                raise ValueError("saved item binding changed")
            continue
        if started.exists():
            raise ValueError(f"item {index} was interrupted; automatic inference retry is forbidden")
        pending.append(index)
    return pending[:maximum_items] if maximum_items is not None else pending


def semantic_config_digest(config):
    from copy import deepcopy
    value = deepcopy(config)
    profile = value.get('execution_profile', {})
    for name in ('query_admission', 'frozen_release'):
        profile.pop(name, None)
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def record_execution_budget(config, batch, label):
    """Publish every resource check, including rejection; never alter input."""
    from artifact_io import atomic_write_json
    from resource_monitor import check_execution_budget, ExecutionBudgetRejected
    if not isinstance(label, str) or not label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in label):
        raise ValueError('invalid resource check label')
    destination = Path(batch) / 'resource_checks' / (label + '-' + uuid.uuid4().hex[:16] + '.json')
    budget = config.get('runtime', {}).get('execution_budget')
    try:
        receipt = check_execution_budget(budget)
    except ExecutionBudgetRejected as exc:
        atomic_write_json(destination, exc.receipt, role='execution_budget_rejected', overwrite=False)
        raise
    except ValueError as exc:
        atomic_write_json(destination, {'status': 'rejected', 'reason': 'invalid_explicit_execution_budget',
            'error_type': type(exc).__name__, 'error_message': str(exc), 'observations': None},
            role='execution_budget_configuration_error', overwrite=False)
        raise
    atomic_write_json(destination, receipt, role='execution_budget_passed', overwrite=False)
    return {'path': destination.relative_to(batch).as_posix(), 'sha256': digest(destination), 'status': receipt['status']}


def verify_role_binding(config, *, acceptance=False, g2_path=None, source_manifest_sha256=None):
    profile = config['execution_profile']
    binding = profile['query_admission']
    release = profile.get('frozen_release')
    if not acceptance:
        if binding.get('purpose') not in {'model_selection', 'calibration'}:
            raise ValueError('this runner accepts development inference only')
        return None
    if binding.get('purpose') != 'acceptance' or not isinstance(release, dict):
        raise ValueError('formal acceptance purpose and frozen release required')
    g2 = read(g2_path)
    if (digest(g2_path) != release.get('sha256') or g2.get('version') != release.get('version')
            or g2.get('status') != 'candidate_frozen' or g2.get('source_manifest_sha256') != source_manifest_sha256
            or g2.get('config_semantic_digest') != semantic_config_digest(config)):
        raise ValueError('G2 release or fixed inference parameters changed')
    pending = read(g2['pending_manifest_path'])
    if digest(g2['pending_manifest_path']) != g2['pending_manifest_sha256']:
        raise ValueError('sealed acceptance manifest changed')
    admitted = read(binding['manifest_path'])
    if (admitted.get('items') != pending.get('items') or not admitted.get('items')
            or admitted.get('protocol_sha256') != g2['protocol_sha256']):
        raise ValueError('acceptance universe or protocol differs from G2')
    return g2


def run(args, *, acceptance=False):
    product = Path(__file__).resolve().parents[1]
    snapshot = product.parent
    verify_release(snapshot, args.source_manifest_sha256)
    config = read(args.config)
    profile = config["execution_profile"]
    if Path(profile["product_root"]).resolve() != product:
        raise ValueError("config must point at this frozen product snapshot")
    binding = profile["query_admission"]
    frozen_release = profile.get('frozen_release')
    verify_role_binding(config, acceptance=acceptance, g2_path=getattr(args, 'g2_release', None),
                        source_manifest_sha256=args.source_manifest_sha256)
    independent = Path(profile["independent_root"]).resolve()
    batch = Path(args.batch).resolve()
    if not batch.is_relative_to(independent) or batch == independent:
        raise ValueError("batch output must be under the independent root")
    if args.max_items is not None and args.max_items < 1:
        raise ValueError("max-items must be positive")
    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 0:
        raise ValueError("disk reserve must be finite and non-negative")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(product))
    from contracts.execution_profile_v4 import audit_execution_profile, isolated_environment, open_admitted_manifest
    from modules.data_admission import PolicyAuthority
    from artifact_io import atomic_write_json
    from contracts.task_contract_v4 import canonical_digest
    authority = PolicyAuthority(args.policy)
    audited = audit_execution_profile(config, authority=authority, config_path=args.config, for_inference=True)
    dataset, manifest = open_admitted_manifest(binding, authority=authority, expected_frozen_release=frozen_release)
    items = manifest["items"]
    split = 'acceptance' if acceptance else 'dev'
    if not items or any(item.get("split") != split for item in items):
        raise ValueError("nonempty fixed-role explicit manifest required")
    if any(item.get("truth_type") not in {"real", "fully_generated", "face_manipulated"} for item in items):
        raise ValueError("explicit truth type is required")
    # Set process-level isolation before product imports and worker creation.
    environment = isolated_environment(config)
    environment["FSTS_EXCLUSION_POLICY_PATH"] = str(Path(args.policy).resolve())
    environment["FSTS_CONFIG_PATH"] = str(Path(args.config).resolve())
    os.environ.clear()
    os.environ.update(environment)
    from orchestrator import run_task
    from runtime_services import RuntimeServiceManager
    from resource_monitor import ResourceMonitor
    from subprocess_resources_v4 import check_task_continuation
    plan = {"schema_version": "acceptance.batch.v1" if acceptance else "development.batch.v1", "config_sha256": digest(args.config),
            "config_digest": audited["config_digest"], "source_manifest_sha256": args.source_manifest_sha256,
            "input_manifest_sha256": digest(binding["manifest_path"]), "receipt_sha256": digest(binding["receipt_path"]),
            "purpose": binding["purpose"], "denominator": len(items), "policy_binding": audited["policy_binding"],
            "protocol_sha256": manifest.get("protocol_sha256"), "training_allowed": False,
            "cloud_allowed": False, "selected_modules": ["deepfake"], "automatic_retry": False,
            **({'frozen_release': frozen_release, 'g2_release_sha256': digest(args.g2_release)} if acceptance else {}),
            "items": [{"index": i, "input_sha256": x["sha256"], "content_group_id": x["content_group_id"],
                       "source_dataset_id": x["source_dataset_id"], "truth_type": x["truth_type"], "media_kind": x["media_kind"]}
                      for i, x in enumerate(items)]}
    batch.mkdir(parents=True, exist_ok=True)
    if (batch / "plan.json").exists():
        if canonical_digest(read(batch / "plan.json")) != canonical_digest(plan):
            raise ValueError("batch freeze differs; existing batch cannot be repurposed")
    else:
        atomic_write_json(batch / "plan.json", plan, role="development_batch_plan", overwrite=False)
    if not args.execute:
        print(json.dumps({"status": "prepared_no_models_loaded", "denominator": len(items)}), flush=True)
        return 0
    write_lock = threading.Lock()
    def event(value):
        with write_lock:
            with (batch / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"time_unix": time.time(), "event": value}, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    manager = None
    with exclusive_batch_lock(independent / "locks" / "heavy-development.lock"):
        pending = select_pending(items, batch, args.max_items)
        if not pending:
            print(json.dumps({"status": "no_pending_items", "denominator": len(items)}), flush=True)
            return 0
        if any((place / name).exists() for place in (independent, batch) for name in ("STOP", "CANCEL_SHUTDOWN")):
            raise ValueError("development execution is stopped")
        if shutil.disk_usage(independent).free < args.minimum_free_gib * 1024 ** 3:
            raise ValueError("insufficient reserved disk space; no data removed")
        startup = time.perf_counter()
        try:
            startup_budget = record_execution_budget(config, batch, 'startup')
            event({'type': 'startup_budget_check', 'receipt': startup_budget})
            if config.get("runtime", {}).get("persistent_workers_enabled"):
                manager = RuntimeServiceManager(args.config, progress_callback=event)
                initial_status = manager.start()
                event({"type": "batch_startup", "seconds": time.perf_counter() - startup, "workers": initial_status})
            for index in pending:
                if any((place / name).exists() for place in (independent, batch) for name in ("STOP", "CANCEL_SHUTDOWN")):
                    event({"type": "batch_stopped", "next_index": index})
                    break
                if shutil.disk_usage(independent).free < args.minimum_free_gib * 1024 ** 3:
                    raise ValueError("disk reserve exhausted; no outputs removed")
                verify_release(snapshot, args.source_manifest_sha256)
                if digest(args.config) != plan["config_sha256"]:
                    raise ValueError("configuration changed")
                # Verifies the entire admitted manifest and bytes, without decode.
                dataset.read_snapshot(index)
                budget_receipt = record_execution_budget(config, batch, f'item-{index:04d}')
                item = items[index]
                base = {"index": index, "input_sha256": item["sha256"], "content_group_id": item["content_group_id"],
                        "started_unix": time.time(), "plan_digest": canonical_digest(plan), 'budget_receipt': budget_receipt}
                atomic_write_json(batch / "items" / f"{index:04d}.started.json", base, role="development_attempt_start", overwrite=False)
                print(json.dumps({"status": "running", "index": index, "denominator": len(items)}), flush=True)
                started = time.perf_counter()
                monitor = ResourceMonitor().start()
                try:
                    status = run_task(item["path"], config_path=args.config, selected_modules=["deepfake"],
                                      services=manager, progress_callback=event)
                    outcome = {"status": status["status"], "task_id": status["task_id"], "task_dir": status["task_dir"],
                               "task_status_sha256": digest(Path(status["task_dir"]) / "task_status.json")}
                except Exception as exc:
                    outcome = {"status": "error", "error_type": type(exc).__name__, "error_message": str(exc)}
                finally:
                    resources = monitor.stop()
                # An integrity error must stop the batch, not become an efficacy score.
                verify_release(snapshot, args.source_manifest_sha256)
                record = {**base, **outcome, "elapsed_seconds": time.perf_counter() - started,
                          "coordinator_resources": resources, "workers": manager.get_status() if manager else {},
                          "cloud_requests": 0, "automatic_inference_retries": 0}
                atomic_write_json(batch / "items" / f"{index:04d}.result.json", record, role="development_result", overwrite=False)
                continuation = check_task_continuation(record.get('task_dir'), worker_status=record['workers'], task_status=record['status'])
                atomic_write_json(batch / 'items' / f'{index:04d}.continuation.json', continuation,
                                  role='post_task_continuation_check', overwrite=False)
                if continuation['status'] != 'passed':
                    atomic_write_json(batch / 'STOP', {'reason': 'unverified_subprocess_cleanup_or_worker_health',
                        'index': index, 'result_preserved_in_denominator': True, 'automatic_retry': False},
                        role='unsafe_continuation_stop', overwrite=False)
                    raise ValueError('Saved outcome retained; subprocess cleanup or worker health requires review before further work')
                print(json.dumps({"index": index, "status": record["status"], "seconds": round(record["elapsed_seconds"], 3)}), flush=True)
        finally:
            if manager is not None:
                manager.close()
                event({"type": "workers_closed", "workers": manager.get_status()})
    completed = sum((batch / "items" / f"{i:04d}.result.json").exists() for i in range(len(items)))
    summary = {"status": "all_items_attempted" if completed == len(items) else "paused_with_remaining_items",
               "completed_attempts": completed, "denominator": len(items), "plan_digest": canonical_digest(plan),
               "effectiveness_passed": False, "interpretation": "fixed outcomes; separate frozen-protocol evaluation required" if acceptance else "development outcomes only; no formal acceptance claim"}
    atomic_write_json(batch / "progress.json", summary, role="development_progress")
    print(json.dumps(summary), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--minimum-free-gib", type=float, default=20)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "stopped", "error_type": type(exc).__name__, "error_message": str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
