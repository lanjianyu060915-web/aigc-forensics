"""No models or media: freeze tamper, interrupted attempts and exclusive queue."""
import json
from pathlib import Path

import pytest

from tools.run_v4_dev_batch import digest, exclusive_batch_lock, select_pending, verify_release
from tools.run_v4_dev_batch import semantic_config_digest, verify_role_binding
from tools.run_v4_dev_batch import record_execution_budget


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def freeze(tmp_path):
    code = tmp_path / "系统" / "new_synthetic_code.py"
    code.parent.mkdir()
    code.write_bytes(b"NEW_SYNTHETIC_VALUE = 1\n")
    manifest = {"schema_version": "source-freeze-v4.1", "manifest_digest": "synthetic-bound-id", "rejected": [],
                "source_files": [{"path": "系统/new_synthetic_code.py", "sha256": digest(code), "size_bytes": code.stat().st_size}]}
    save(tmp_path / "source_manifest.json", manifest)
    sha = digest(tmp_path / "source_manifest.json")
    save(tmp_path / "source_snapshot_complete.json", {"status": "source_snapshot_verified", "source_manifest_sha256": sha,
         "manifest_digest": manifest["manifest_digest"]})
    return code, sha


def test_frozen_code_tampering_is_rejected_before_import(tmp_path):
    code, sha = freeze(tmp_path)
    assert verify_release(tmp_path, sha)["source_files"]
    code.write_bytes(b"NEW_SYNTHETIC_VALUE = 2\n")
    with pytest.raises(ValueError, match="source bytes changed"):
        verify_release(tmp_path, sha)


def test_extra_code_is_not_silently_importable(tmp_path):
    _, sha = freeze(tmp_path)
    (tmp_path / "extra.py").write_bytes(b"# synthetic\n")
    with pytest.raises(ValueError, match="unlisted file"):
        verify_release(tmp_path, sha)


def test_manifest_or_completion_changes_reject(tmp_path):
    _, sha = freeze(tmp_path)
    with pytest.raises(ValueError, match="manifest changed"):
        verify_release(tmp_path, "0" * 64)
    save(tmp_path / "source_snapshot_complete.json", {"status": "incomplete"})
    with pytest.raises(ValueError, match="completion binding"):
        verify_release(tmp_path, sha)


def test_failed_item_stays_in_denominator_and_is_not_rerun(tmp_path):
    items = [{"sha256": "a" * 64}, {"sha256": "b" * 64}, {"sha256": "c" * 64}]
    save(tmp_path / "items/0000.result.json", {"index": 0, "input_sha256": items[0]["sha256"], "status": "error"})
    assert select_pending(items, tmp_path, 1) == [1]
    assert select_pending(items, tmp_path, None) == [1, 2]
    assert len(items) == 3
    save(tmp_path / "items/0001.started.json", {"status": "running"})
    with pytest.raises(ValueError, match="interrupted"):
        select_pending(items, tmp_path, None)


def test_saved_result_requires_same_input_binding(tmp_path):
    save(tmp_path / "items/0000.result.json", {"index": 0, "input_sha256": "b" * 64})
    with pytest.raises(ValueError, match="binding changed"):
        select_pending([{"sha256": "a" * 64}], tmp_path, None)


def test_queue_lock_prevents_second_owner_and_preserves_stop(tmp_path):
    lock = tmp_path / "queue.lock"
    stop = tmp_path / "STOP"
    stop.write_text("stop", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with exclusive_batch_lock(lock):
            with pytest.raises(FileExistsError):
                with exclusive_batch_lock(lock):
                    pytest.fail("second owner acquired queue")
            assert lock.exists()
            raise RuntimeError("synthetic interruption")
    assert not lock.exists() and stop.read_text("utf-8") == "stop"


def test_acceptance_requires_frozen_universe_and_keeps_parameter_binding(tmp_path):
    pending = tmp_path / 'pending.json'
    admitted = tmp_path / 'admitted.json'
    g2_path = tmp_path / 'g2.json'
    rows = [{'sha256': 'a' * 64, 'split': 'acceptance'}]
    save(pending, {'items': rows})
    save(admitted, {'items': rows, 'protocol_sha256': 'p' * 64})
    config = {'threshold': 0.5, 'execution_profile': {'query_admission': {'purpose': 'acceptance', 'manifest_path': str(admitted)}}}
    before = semantic_config_digest(config)
    save(g2_path, {'version': 'synthetic-g2', 'status': 'candidate_frozen', 'source_manifest_sha256': 's' * 64,
        'config_semantic_digest': before, 'pending_manifest_path': str(pending), 'pending_manifest_sha256': digest(pending),
        'protocol_sha256': 'p' * 64})
    config['execution_profile']['frozen_release'] = {'version': 'synthetic-g2', 'sha256': digest(g2_path)}
    assert semantic_config_digest(config) == before
    with pytest.raises(ValueError, match='development inference only'):
        verify_role_binding(config)
    assert verify_role_binding(config, acceptance=True, g2_path=g2_path, source_manifest_sha256='s' * 64)
    config['threshold'] = 0.6
    with pytest.raises(ValueError, match='parameters changed'):
        verify_role_binding(config, acceptance=True, g2_path=g2_path, source_manifest_sha256='s' * 64)
    config['threshold'] = 0.5
    save(admitted, {'items': [{'sha256': 'b' * 64}], 'protocol_sha256': 'p' * 64})
    with pytest.raises(ValueError, match='universe'):
        verify_role_binding(config, acceptance=True, g2_path=g2_path, source_manifest_sha256='s' * 64)
    save(pending, {'items': []})
    with pytest.raises(ValueError, match='sealed acceptance manifest changed'):
        verify_role_binding(config, acceptance=True, g2_path=g2_path, source_manifest_sha256='s' * 64)


def test_budget_rejection_persists_receipt_without_an_inference_attempt(tmp_path, monkeypatch):
    import resource_monitor
    def rejected(budget):
        raise resource_monitor.ExecutionBudgetRejected({'status': 'rejected', 'reasons': ['synthetic-low-headroom'], 'budget': budget})
    monkeypatch.setattr(resource_monitor, 'check_execution_budget', rejected)
    with pytest.raises(resource_monitor.ExecutionBudgetRejected):
        record_execution_budget({'runtime': {'execution_budget': {'new_synthetic_marker': 1}}}, tmp_path, 'startup')
    paths = list((tmp_path / 'resource_checks').glob('*.json'))
    assert len(paths) == 1 and json.loads(paths[0].read_text('utf-8'))['status'] == 'rejected'
    assert not (tmp_path / 'items').exists()


def test_budget_pass_and_configuration_errors_are_distinct(tmp_path, monkeypatch):
    import resource_monitor
    monkeypatch.setattr(resource_monitor, 'check_execution_budget', lambda budget: {'status': 'passed', 'budget': budget})
    result = record_execution_budget({'runtime': {'execution_budget': {}}}, tmp_path, 'startup')
    assert result['status'] == 'passed' and digest(tmp_path / result['path']) == result['sha256']
    def invalid(budget): raise ValueError('synthetic invalid explicit configuration')
    monkeypatch.setattr(resource_monitor, 'check_execution_budget', invalid)
    with pytest.raises(ValueError): record_execution_budget({}, tmp_path, 'startup')
    saved = [json.loads(path.read_text('utf-8')) for path in (tmp_path / 'resource_checks').glob('*.json')]
    assert len(saved) == 2 and any(row.get('reason') == 'invalid_explicit_execution_budget' for row in saved)
