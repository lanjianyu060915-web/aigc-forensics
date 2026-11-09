"""Independent synthetic metadata, fake manager/task/budget/monitor only.

No real dev/acceptance paths, media decode, model imports or protected results.
Each fixture writes newly constructed marker JSON to its pytest directory.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_v4_sampling_comparison as runner
from tools.freeze_source_v4 import TOOLS


def save(path, value, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if kwargs.get("overwrite") is False else "w"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)


def configs():
    a = {"execution_profile": {"schema_version": "execution_profile.independent.v1", "mode": "independent_inference",
         "query_admission": {"purpose": "model_selection", "manifest_path": "synthetic-manifest", "receipt_path": "synthetic-receipt"}},
         "runtime": {"tasks_root": "synthetic-output", "execution_budget": {"synthetic": True}},
         "video": {"candidate_frames": 24, "max_key_frames": 5},
         "assets": {"model": {"sha256": "a" * 64}}}
    b = deepcopy(a)
    b["video"].update(selection_strategy="segmented_time_v1", allow_experimental_selection=True)
    return {"A": a, "B": b}


def items(count=3):
    return [{"path": f"new-marker-{i}.mp4", "sha256": hashlib.sha256(f"new-marker-{i}".encode()).hexdigest(),
             "split": "dev", "truth_type": "fully_generated", "media_kind": "video",
             "source_dataset_id": "synthetic-publisher", "content_group_id": f"new-group-{i}"} for i in range(count)]


def test_configuration_deepcopy_allows_only_two_sampling_switches():
    values = configs()
    before = deepcopy(values)
    result = runner.validate_config_pair(values["A"], values["B"])
    assert values == before
    assert result["full_deepcopy_comparison"] == "passed"
    assert result["allowed_difference_paths"] == ["video.selection_strategy", "video.allow_experimental_selection"]


@pytest.mark.parametrize("section,key,value", [
    ("video", "candidate_frames", 25), ("runtime", "tasks_root", "different-output"),
    ("assets", "model", {"sha256": "b" * 64}), ("runtime", "execution_budget", {"different": True}),
    ("execution_profile", "query_admission", {"purpose": "model_selection", "manifest_path": "other", "receipt_path": "other"}),
])
def test_configuration_other_differences_are_rejected(section, key, value):
    values = configs()
    values["B"][section][key] = value
    with pytest.raises(ValueError, match="outside"):
        runner.validate_config_pair(values["A"], values["B"])


@pytest.mark.parametrize("change", ["acceptance", "calibration", "nonboolean", "wrong_mode", "both_baseline"])
def test_configuration_rejects_wrong_role_or_candidate(change):
    values = configs()
    if change in {"acceptance", "calibration"}:
        for value in values.values():
            value["execution_profile"]["query_admission"]["purpose"] = change
    elif change == "nonboolean":
        values["B"]["video"]["allow_experimental_selection"] = 1
    elif change == "wrong_mode":
        values["B"]["execution_profile"]["mode"] = "engineering_only"
    else:
        values["B"]["video"] = deepcopy(values["A"]["video"])
    with pytest.raises(ValueError):
        runner.validate_config_pair(values["A"], values["B"])


def test_schedule_keeps_every_manifest_index_and_hash_parity_before_results():
    inputs = items(20)
    inputs[19]["content_group_id"] = inputs[0]["content_group_id"]
    schedule = runner.build_schedule(inputs)
    assert len(schedule) == 40 and schedule == runner.build_schedule(deepcopy(inputs))
    for index in range(20):
        pair = schedule[2 * index:2 * index + 2]
        assert [row["index"] for row in pair] == [index, index]
        expected = "AB" if int(hashlib.sha256(inputs[index]["content_group_id"].encode()).hexdigest(), 16) % 2 == 0 else "BA"
        assert "".join(row["arm"] for row in pair) == expected
    assert schedule[0]["order"] == schedule[-1]["order"]
    assert {row["order"] for row in schedule} == {"AB", "BA"}


@pytest.mark.parametrize("field,value", [("split", "acceptance"), ("media_kind", "image"), ("truth_type", "guess"), ("content_group_id", "")])
def test_schedule_rejects_non_dev_video_or_unknown_metadata(field, value):
    inputs = items(1)
    inputs[0][field] = value
    with pytest.raises(ValueError):
        runner.build_schedule(inputs)


class Monitor:
    def start(self):
        return self
    def stop(self):
        return {"status": "synthetic_sample", "scope": "single_pid_excludes_children"}


@pytest.fixture
def harness(tmp_path):
    inputs, values = items(3), configs()
    plans = {arm: {"config_sha256": ("a" if arm == "A" else "b") * 64,
                   "denominator": len(inputs), "fixture_origin": "new_synthetic_metadata"} for arm in ("A", "B")}
    paths = {arm: tmp_path / f"config-{arm}.json" for arm in ("A", "B")}
    batches = {arm: tmp_path / f"batch-{arm}" for arm in ("A", "B")}
    state = {"managers": [], "calls": [], "verify": [], "events": [], "budgets": 0}
    class Manager:
        def __init__(self, config, **kwargs):
            self.config, self.closed = config, False
            state["managers"].append(self)
        def start(self):
            return {"synthetic-worker": "ready"}
        def get_status(self):
            return {"synthetic-worker": "closed" if self.closed else "ready"}
        def close(self):
            self.closed = True
    def task(path, *, config_path, services, **kwargs):
        arm = next(arm for arm, value in paths.items() if value == config_path)
        state["calls"].append((arm, path, services))
        task_root = tmp_path / "new-tasks" / str(len(state["calls"]))
        status = {"status": "success", "task_id": task_root.name, "task_dir": str(task_root)}
        save(task_root / "task_status.json", status)
        return status
    def budget(value):
        state["budgets"] += 1
        return {"status": "passed", "synthetic": True, "budget": value}
    kwargs = dict(items=inputs, schedule=runner.build_schedule(inputs), batches=batches, plans=plans,
                  comparison_digest="c" * 64, comparison_root=tmp_path / "comparison", config_paths=paths,
                  configs=values, verify_inputs=lambda index: state["verify"].append(index), guard=lambda: None,
                  manager_factory=Manager, task_runner=task, monitor_factory=Monitor, budget_checker=budget,
                  writer=save, event=state["events"].append)
    return kwargs, state, tmp_path


def test_serial_shared_manager_one_start_all_results_and_no_repeat(harness):
    kwargs, state, root = harness
    result = runner.execute_queue(**kwargs)
    expected = [(row["arm"], kwargs["items"][row["index"]]["path"]) for row in kwargs["schedule"]]
    assert [(arm, path) for arm, path, _ in state["calls"]] == expected
    assert len(state["managers"]) == 1 and state["managers"][0].config == kwargs["config_paths"]["A"]
    assert all(manager is state["managers"][0] for _, _, manager in state["calls"])
    assert state["managers"][0].closed and state["budgets"] == 7
    assert state["verify"] == [None, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert result["denominator_per_arm"] == 3 and result["manager_initializations"] == 1
    for arm in ("A", "B"):
        for index in range(3):
            record = runner.read(kwargs["batches"][arm] / "items" / f"{index:04d}.result.json")
            assert record["no_result_cache_reuse"] is True and record["manager_initialization_config"] == "A"
            assert record["automatic_inference_retries"] == record["cloud_requests"] == 0
    again = runner.execute_queue(**kwargs)
    assert again["status"] == "no_pending_items" and len(state["managers"]) == 1
    assert len(list((root / "comparison/sessions").glob("*/startup.json"))) == 1


def test_task_failure_preserved_and_fixed_denominator_unchanged(harness):
    kwargs, state, _ = harness
    real_fake_task = kwargs["task_runner"]
    count = 0
    def failing(*args, **extra):
        nonlocal count
        count += 1
        if count == 1:
            raise RuntimeError("new synthetic model failure")
        return real_fake_task(*args, **extra)
    kwargs["task_runner"] = failing
    result = runner.execute_queue(**kwargs)
    first = kwargs["schedule"][0]
    record = runner.read(kwargs["batches"][first["arm"]] / "items/0000.result.json")
    assert record["status"] == "error" and result["denominator_per_arm"] == 3
    runner.execute_queue(**kwargs)
    assert count == 6  # failed first attempt also remains completed


def test_interruption_is_never_automatically_retried(harness):
    kwargs, state, _ = harness
    def interrupt(*args, **extra):
        raise KeyboardInterrupt("synthetic interruption")
    kwargs["task_runner"] = interrupt
    with pytest.raises(KeyboardInterrupt):
        runner.execute_queue(**kwargs)
    assert state["managers"][0].closed
    with pytest.raises(ValueError, match="interrupted"):
        runner.execute_queue(**kwargs)
    assert len(state["managers"]) == 1


@pytest.mark.parametrize("reject_at", [1, 2])
def test_budget_rejection_saved_before_startup_or_task_without_attempt(harness, reject_at):
    kwargs, state, root = harness
    count = 0
    class Rejected(RuntimeError):
        receipt = {"status": "rejected", "reasons": ["synthetic missing observation"]}
    def budget(value):
        nonlocal count
        count += 1
        if count == reject_at:
            raise Rejected("synthetic budget rejection")
        return {"status": "passed", "synthetic": True}
    kwargs["budget_checker"] = budget
    with pytest.raises(Rejected):
        runner.execute_queue(**kwargs)
    assert not state["calls"]
    assert not any(root.glob("batch-*/items/*.started.json"))
    receipts = [runner.read(path) for path in (root / "comparison/sessions").glob("*/*.budget.json")]
    rejected = [value for value in receipts if value.get("status") == "stopped"]
    assert rejected[0]["receipt"]["status"] == "rejected"
    assert len(state["managers"]) == reject_at - 1


def test_post_task_integrity_failure_preserves_unaccepted_outcome_and_blocks_resume(harness):
    kwargs, state, root = harness
    calls = 0
    def verify(index):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("synthetic changed source")
    kwargs["verify_inputs"] = verify
    with pytest.raises(ValueError, match="changed source"):
        runner.execute_queue(**kwargs)
    assert not any(root.glob("batch-*/items/*.result.json"))
    records = list((root / "comparison/sessions").glob("*/*.integrity_failure.json"))
    assert runner.read(records[0])["unaccepted_outcome"]["status"] == "success"
    with pytest.raises(ValueError, match="interrupted"):
        runner.execute_queue(**kwargs)


def test_post_task_cleanup_rejection_keeps_result_and_stops_before_next_case(harness):
    kwargs, state, root = harness
    kwargs["continuation_checker"] = lambda *args: {"status": "rejected", "reasons": ["synthetic cleanup unknown"]}
    with pytest.raises(ValueError, match="after preserving outcome"):
        runner.execute_queue(**kwargs)
    assert len(state["calls"]) == 1 and state["managers"][0].closed
    first = kwargs["schedule"][0]
    result = runner.read(kwargs["batches"][first["arm"]] / "items/0000.result.json")
    assert result["status"] == "success" and result["input_sha256"] == kwargs["items"][0]["sha256"]
    assert (root / "comparison/STOP").is_file()
    assert len(list((root / "comparison/sessions").glob("*/*.continuation.json"))) == 1
    def stop_guard():
        if (root / "comparison/STOP").exists():
            raise ValueError("synthetic stop marker preserved")
    kwargs["guard"] = stop_guard
    with pytest.raises(ValueError, match="stop marker"):
        runner.execute_queue(**kwargs)
    assert len(state["managers"]) == 1


def test_resume_rejects_changed_result_binding_and_out_of_order_completion(harness):
    kwargs, _, _ = harness
    runner.execute_queue(**kwargs)
    first, second = kwargs["schedule"][:2]
    first_path = kwargs["batches"][first["arm"]] / "items/0000.result.json"
    value = runner.read(first_path)
    value["comparison_digest"] = "d" * 64
    save(first_path, value)
    with pytest.raises(ValueError, match="binding changed"):
        runner.execute_queue(**kwargs)
    first_path.unlink()
    (first_path.parent / "0000.started.json").unlink()
    with pytest.raises(ValueError, match="prefix"):
        runner.execute_queue(**kwargs)


def test_stop_and_disk_guard_do_not_delete_markers_or_lower_budget(tmp_path, monkeypatch):
    stop = tmp_path / "STOP"
    stop.write_text("synthetic STOP", "utf-8")
    with pytest.raises(ValueError, match="stopped"):
        runner.check_stop_and_disk([tmp_path], tmp_path, 20)
    assert stop.exists()
    other = tmp_path / "fresh"
    other.mkdir()
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="disk reserve"):
        runner.check_stop_and_disk([other], other, 20)


def test_arm_plan_uses_evaluator_schema_and_publish_never_repurposes(tmp_path):
    values = configs()
    manifest_path, receipt_path, config_path = (tmp_path / name for name in ("inputs.json", "receipt.json", "config.json"))
    manifest = {"items": items(2), "protocol_sha256": "e" * 64}
    save(manifest_path, manifest)
    save(receipt_path, {"status": "passed", "synthetic": True})
    values["A"]["execution_profile"]["query_admission"].update(manifest_path=str(manifest_path), receipt_path=str(receipt_path))
    save(config_path, values["A"])
    plan = runner.build_arm_plan(values["A"], config_path, {"config_digest": "f" * 64, "policy_binding": {"synthetic": True}}, manifest, "a" * 64)
    assert plan["schema_version"] == "development.batch.v1" and plan["denominator"] == 2
    assert plan["receipt_sha256"] == runner.digest(receipt_path)
    assert [row["index"] for row in plan["items"]] == [0, 1]
    target = tmp_path / "plan.json"
    runner.publish_fixed(target, plan, save)
    runner.publish_fixed(target, plan, save)
    with pytest.raises(ValueError, match="repurpose"):
        runner.publish_fixed(target, {**plan, "denominator": 1}, save)
    assert "run_v4_sampling_comparison.py" in TOOLS
