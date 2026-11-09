"""Prepare or execute a fixed paired development sampling comparison.

Only the two explicit sampling switches may differ. One manager initialized
from A serves the serial AB/BA queue; saved results and interrupted attempts are
never retried. This tool neither selects a strategy nor opens acceptance data.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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

# The bootstrap helper has stdlib-only top-level imports and verifies the frozen
# tree before any orchestrator, worker or model service is imported.
sys.dont_write_bytecode = True
try:
    from .run_v4_dev_batch import digest, exclusive_batch_lock, read, select_pending, verify_release
except ImportError:  # direct CLI invocation from the frozen tools directory
    from run_v4_dev_batch import digest, exclusive_batch_lock, read, select_pending, verify_release


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def validate_config_pair(config_a, config_b):
    """Deep comparison; no path, model, budget or admission exception."""
    values = [deepcopy(config_a), deepcopy(config_b)]
    canonical_digest(values)  # reject nonfinite or non-JSON configuration
    for arm, value in zip(("A", "B"), values):
        profile = value.get("execution_profile", {})
        binding = profile.get("query_admission", {})
        if (profile.get("schema_version") != "execution_profile.independent.v1"
                or profile.get("mode") != "independent_inference"
                or binding.get("purpose") != "model_selection"):
            raise ValueError("sampling comparison requires independent model_selection; acceptance is forbidden")
        section = value.get("video")
        if not isinstance(section, dict):
            raise ValueError("explicit video configuration required")
        expected = ("legacy_quality_v1", False) if arm == "A" else ("segmented_time_v1", True)
        strategy = section.pop("selection_strategy", "legacy_quality_v1")
        enabled = section.pop("allow_experimental_selection", False)
        if not isinstance(enabled, bool) or (strategy, enabled) != expected:
            raise ValueError("A must be legacy/false and B segmented/true with explicit boolean switches")
    if values[0] != values[1]:
        raise ValueError("A/B configurations differ outside the two video sampling switches")
    return {"allowed_difference_paths": ["video.selection_strategy", "video.allow_experimental_selection"],
            "common_config_digest": canonical_digest(values[0]), "full_deepcopy_comparison": "passed"}


def build_schedule(items):
    """Manifest order stays fixed; SHA-256(content_group_id) parity sets AB/BA."""
    if not isinstance(items, list) or not items:
        raise ValueError("nonempty explicit development manifest required")
    schedule = []
    for index, item in enumerate(items):
        if (item.get("split") != "dev" or item.get("media_kind") != "video"
                or item.get("truth_type") not in {"real", "fully_generated", "face_manipulated"}):
            raise ValueError("comparison accepts explicit dev video truth metadata only")
        group = item.get("content_group_id")
        if not isinstance(group, str) or not group:
            raise ValueError("original content group is required")
        group_hash = hashlib.sha256(group.encode("utf-8")).hexdigest()
        order = "AB" if int(group_hash, 16) % 2 == 0 else "BA"
        for arm in order:
            schedule.append({"sequence": len(schedule), "index": index, "arm": arm,
                             "content_group_id": group, "order": order, "order_hash": group_hash})
    return schedule


def build_arm_plan(config, config_path, audited, manifest, source_sha):
    """Exactly the development batch schema consumed by evaluate_v4_batch."""
    binding, items = config["execution_profile"]["query_admission"], manifest["items"]
    return {"schema_version": "development.batch.v1", "config_sha256": digest(config_path),
            "config_digest": audited["config_digest"], "source_manifest_sha256": source_sha,
            "input_manifest_sha256": digest(binding["manifest_path"]), "receipt_sha256": digest(binding["receipt_path"]),
            "purpose": "model_selection", "denominator": len(items), "policy_binding": audited["policy_binding"],
            "protocol_sha256": manifest.get("protocol_sha256"), "training_allowed": False, "cloud_allowed": False,
            "selected_modules": ["deepfake"], "automatic_retry": False,
            "items": [{"index": i, "input_sha256": x["sha256"], "content_group_id": x["content_group_id"],
                       "source_dataset_id": x["source_dataset_id"], "truth_type": x["truth_type"], "media_kind": x["media_kind"]}
                      for i, x in enumerate(items)]}


def publish_fixed(path, value, writer):
    if path.exists():
        if canonical_digest(read(path)) != canonical_digest(value):
            raise ValueError("existing comparison or arm plan differs; do not repurpose a batch")
    else:
        writer(path, value, role="sampling_comparison_plan", overwrite=False)


def pending_schedule(items, schedule, batches, plans, comparison_digest):
    pending = {arm: set(select_pending(items, batches[arm], None)) for arm in ("A", "B")}
    remaining, gap = [], False
    for row in schedule:
        arm, index = row["arm"], row["index"]
        if index in pending[arm]:
            gap = True
            remaining.append(row)
            continue
        if gap:
            raise ValueError("saved outcomes are not a prefix of the frozen AB/BA order")
        result = read(batches[arm] / "items" / f"{index:04d}.result.json")
        started_path = batches[arm] / "items" / f"{index:04d}.started.json"
        if not started_path.is_file():
            raise ValueError("completed item lacks its immutable start record")
        started = read(started_path)
        bindings = {"index": index, "input_sha256": items[index]["sha256"],
                    "content_group_id": items[index]["content_group_id"], "plan_digest": canonical_digest(plans[arm]),
                    "comparison_digest": comparison_digest, "arm": arm, "schedule_sequence": row["sequence"]}
        if any(result.get(key) != value or started.get(key) != value for key, value in bindings.items()):
            raise ValueError("saved comparison item binding changed")
    return remaining


def check_stop_and_disk(roots, independent, minimum_free_gib):
    if any((root / name).exists() for root in roots for name in ("STOP", "CANCEL_SHUTDOWN")):
        raise ValueError("sampling comparison stopped by an existing STOP/CANCEL_SHUTDOWN marker")
    if shutil.disk_usage(independent).free < minimum_free_gib * 1024 ** 3:
        raise ValueError("sampling comparison disk reserve insufficient; no data removed")


def execute_queue(*, items, schedule, batches, plans, comparison_digest, comparison_root,
                  config_paths, configs, verify_inputs, guard, manager_factory,
                  task_runner, monitor_factory, budget_checker, writer, event,
                  continuation_checker=None):
    """Serial executor; injectable services support model-free contract tests."""
    remaining = pending_schedule(items, schedule, batches, plans, comparison_digest)
    if not remaining:
        return {"status": "no_pending_items", "denominator_per_arm": len(items), "manager_initializations": 0}
    session = uuid.uuid4().hex
    session_root = comparison_root / "sessions" / session
    if continuation_checker is None:
        from subprocess_resources_v4 import check_task_continuation
        continuation_checker = check_task_continuation
    manager = None
    completed = 0
    def budget(label):
        try:
            receipt = budget_checker(configs["A"]["runtime"].get("execution_budget"))
        except Exception as exc:
            writer(session_root / (label + ".budget.json"),
                   {"status": "stopped", "error_type": type(exc).__name__, "error_message": str(exc),
                    "receipt": getattr(exc, "receipt", None)}, role="sampling_budget_failure", overwrite=False)
            raise
        writer(session_root / (label + ".budget.json"), receipt, role="sampling_budget", overwrite=False)
        return receipt
    try:
        guard()
        # Startup must also recheck source/config/current policy before imports
        # inside the manager. The initial call has no media index.
        verify_inputs(None)
        startup_budget = budget("startup")
        startup = time.perf_counter()
        startup_monitor = monitor_factory().start()
        startup_error = None
        try:
            manager = manager_factory(config_paths["A"], progress_callback=event)
            manager.start()
        except Exception as exc:
            startup_error = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            resources = startup_monitor.stop()
            writer(session_root / "startup.json", {"status": "error" if startup_error else "initialized",
                "error": startup_error, "seconds": time.perf_counter() - startup,
                "manager_initialization_config": "A", "config_sha256": plans["A"]["config_sha256"],
                "manager_initializations": 1, "coordinator_resources": resources,
                "workers": manager.get_status() if manager else {}, "budget": startup_budget,
                "no_result_cache_reuse": True}, role="sampling_cold_start", overwrite=False)
        for row in remaining:
            arm, index = row["arm"], row["index"]
            guard()
            verify_inputs(index)
            budget_receipt = budget(f"case-{row['sequence']:04d}-{arm}-{index:04d}")
            item = items[index]
            base = {"index": index, "input_sha256": item["sha256"], "content_group_id": item["content_group_id"],
                    "started_unix": time.time(), "plan_digest": canonical_digest(plans[arm]),
                    "comparison_digest": comparison_digest, "arm": arm, "schedule_sequence": row["sequence"],
                    "order": row["order"], "session_id": session, "manager_initialization_config": "A",
                    "no_result_cache_reuse": True, "budget_receipt": budget_receipt}
            writer(batches[arm] / "items" / f"{index:04d}.started.json", base,
                   role="development_attempt_start", overwrite=False)
            started = time.perf_counter()
            monitor = monitor_factory().start()
            try:
                status = task_runner(item["path"], config_path=config_paths[arm], selected_modules=["deepfake"],
                                     services=manager, progress_callback=event)
                outcome = {"status": status["status"], "task_id": status["task_id"], "task_dir": status["task_dir"],
                           "task_status_sha256": digest(Path(status["task_dir"]) / "task_status.json")}
            except Exception as exc:
                outcome = {"status": "error", "error_type": type(exc).__name__, "error_message": str(exc)}
            finally:
                resources = monitor.stop()
            # Preserve failures before a post-run integrity stop; never erase a
            # completed inference because the next source check detects change.
            record = {**base, **outcome, "elapsed_seconds": time.perf_counter() - started,
                      "coordinator_resources": resources, "workers": manager.get_status(),
                      "cloud_requests": 0, "automatic_inference_retries": 0}
            try:
                verify_inputs(index)
            except Exception as exc:
                writer(session_root / f"case-{row['sequence']:04d}.integrity_failure.json",
                       {"status": "integrity_failure", "error_type": type(exc).__name__, "error_message": str(exc),
                        "unaccepted_outcome": record}, role="sampling_integrity_failure", overwrite=False)
                raise  # started remains interrupted; no ordinary result/automatic retry
            writer(batches[arm] / "items" / f"{index:04d}.result.json", record,
                   role="development_result", overwrite=False)
            completed += 1
            continuation = continuation_checker(record.get("task_dir"), record["workers"], record["status"])
            writer(session_root / f"case-{row['sequence']:04d}.continuation.json", continuation,
                   role="sampling_continuation_check", overwrite=False)
            if continuation.get("status") != "passed":
                writer(comparison_root / "STOP", {"reason": "unsafe_to_continue_after_saved_outcome",
                    "session_id": session, "arm": arm, "index": index,
                    "continuation": continuation}, role="sampling_comparison_stop", overwrite=False)
                raise ValueError("comparison stopped after preserving outcome: subprocess cleanup or worker health unverified")
            event({"type": "sampling_case_complete", "arm": arm, "index": index,
                   "sequence": row["sequence"], "status": record["status"]})
        return {"status": "all_items_attempted", "denominator_per_arm": len(items),
                "completed_this_session": completed, "manager_initializations": 1, "session_id": session}
    finally:
        if manager is not None:
            manager.close()
            event({"type": "shared_workers_closed", "session_id": session, "workers": manager.get_status()})


def run(args):
    product = Path(__file__).resolve().parents[1]
    snapshot = product.parent
    verify_release(snapshot, args.source_manifest_sha256)
    config_paths = {"A": Path(args.config_a).resolve(), "B": Path(args.config_b).resolve()}
    configs = {arm: read(path) for arm, path in config_paths.items()}
    differences = validate_config_pair(configs["A"], configs["B"])
    profile = configs["A"]["execution_profile"]
    if Path(profile["product_root"]).resolve() != product:
        raise ValueError("configs must bind this frozen product snapshot")
    independent = Path(profile["independent_root"]).resolve()
    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 0:
        raise ValueError("disk reserve must be finite and non-negative")
    def output_path(value):
        path = Path(value).absolute()
        if path.resolve() != path or not path.is_relative_to(independent) or path == independent:
            raise ValueError("comparison outputs must be direct paths under the independent root")
        return path
    batches = {"A": output_path(args.batch_a), "B": output_path(args.batch_b)}
    comparison_root = output_path(args.comparison_root)
    outputs = [*batches.values(), comparison_root]
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(outputs) for b in outputs[i + 1:]):
        raise ValueError("arm batches and comparison root must be separate nonnested directories")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(product))
    from contracts.execution_profile_v4 import audit_execution_profile, isolated_environment, open_admitted_manifest
    from modules.data_admission import PolicyAuthority
    from artifact_io import atomic_write_json
    authority = PolicyAuthority(args.policy)
    audited = {arm: audit_execution_profile(configs[arm], authority=authority, config_path=config_paths[arm], for_inference=True)
               for arm in ("A", "B")}
    dataset, manifest = open_admitted_manifest(profile["query_admission"], authority=authority)
    items = manifest["items"]
    schedule = build_schedule(items)
    plans = {arm: build_arm_plan(configs[arm], config_paths[arm], audited[arm], manifest, args.source_manifest_sha256)
             for arm in ("A", "B")}
    comparison = {"schema_version": "sampling.comparison.v1", "denominator_per_arm": len(items),
                  "source_manifest_sha256": args.source_manifest_sha256,
                  "arm_plan_digests": {arm: canonical_digest(plan) for arm, plan in plans.items()},
                  "arm_batches": {arm: str(path) for arm, path in batches.items()},
                  "config_paths": {arm: str(path) for arm, path in config_paths.items()},
                  "config_pair": differences, "schedule": schedule,
                  "order_rule": "SHA256(UTF8(content_group_id)) integer parity: even AB, odd BA; manifest index order",
                  "manager_initialization_config": "A", "no_result_cache_reuse": True,
                  "automatic_retry": False, "automatic_strategy_selection": False,
                  "acceptance_allowed": False, "training_allowed": False, "cloud_allowed": False}
    comparison_digest = canonical_digest(comparison)
    for arm in ("A", "B"):
        publish_fixed(batches[arm] / "plan.json", plans[arm], atomic_write_json)
    publish_fixed(comparison_root / "plan.json", comparison, atomic_write_json)
    if not args.execute:
        print(json.dumps({"status": "prepared_no_models_loaded", "denominator_per_arm": len(items),
                          "comparison_digest": comparison_digest}), flush=True)
        return 0
    environment = isolated_environment(configs["A"])
    environment.update(FSTS_EXCLUSION_POLICY_PATH=str(Path(args.policy).resolve()), FSTS_CONFIG_PATH=str(config_paths["A"]))
    os.environ.clear()
    os.environ.update(environment)
    from orchestrator import run_task
    from runtime_services import RuntimeServiceManager
    from resource_monitor import ResourceMonitor, check_execution_budget
    write_lock = threading.Lock()
    def event(value):
        with write_lock:
            with (comparison_root / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"time_unix": time.time(), "event": value}, ensure_ascii=False,
                                       allow_nan=False, default=str) + "\n")
    def guard():
        check_stop_and_disk([independent, *outputs], independent, args.minimum_free_gib)
        for path in outputs:
            output_path(path)
    def verify_inputs(index):
        verify_release(snapshot, args.source_manifest_sha256)
        for arm in ("A", "B"):
            if digest(config_paths[arm]) != plans[arm]["config_sha256"]:
                raise ValueError("configuration changed during comparison")
            if canonical_digest(read(batches[arm] / "plan.json")) != canonical_digest(plans[arm]):
                raise ValueError("arm plan changed during comparison")
        if canonical_digest(read(comparison_root / "plan.json")) != comparison_digest:
            raise ValueError("comparison schedule changed")
        if index is None:
            open_admitted_manifest(profile["query_admission"], authority=authority)
        else:
            dataset.read_snapshot(index)  # current policy/checker/receipt and entire manifest reverified
    with exclusive_batch_lock(independent / "locks" / "heavy-development.lock"):
        try:
            summary = execute_queue(items=items, schedule=schedule, batches=batches, plans=plans,
                comparison_digest=comparison_digest, comparison_root=comparison_root, config_paths=config_paths,
                configs=configs, verify_inputs=verify_inputs, guard=guard, manager_factory=RuntimeServiceManager,
                task_runner=run_task, monitor_factory=ResourceMonitor, budget_checker=check_execution_budget,
                writer=atomic_write_json, event=event)
        except Exception as exc:
            event({"type": "sampling_comparison_stopped", "error_type": type(exc).__name__, "error_message": str(exc)})
            raise
        finally:
            for arm in ("A", "B"):
                completed = sum((batches[arm] / "items" / f"{i:04d}.result.json").exists() for i in range(len(items)))
                atomic_write_json(batches[arm] / "progress.json", {"status": "all_items_attempted" if completed == len(items) else "paused_with_remaining_items",
                    "completed_attempts": completed, "denominator": len(items), "plan_digest": canonical_digest(plans[arm]),
                    "effectiveness_passed": False, "interpretation": "sampling development only; no strategy enablement"}, role="development_progress")
        atomic_write_json(comparison_root / "progress.json", {**summary, "comparison_digest": comparison_digest,
            "effectiveness_passed": False, "automatic_strategy_selection": False}, role="sampling_comparison_progress")
    print(json.dumps(summary), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config-a", "config-b", "policy", "batch-a", "batch-b", "comparison-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=20)
    parser.add_argument("--execute", action="store_true")
    try:
        return run(parser.parse_args())
    except Exception as exc:
        print(json.dumps({"status": "stopped", "error_type": type(exc).__name__, "error_message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
