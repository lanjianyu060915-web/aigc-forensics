"""Read-only evaluation of a predeclared, admitted, fixed-universe batch.

Outcome definitions come from the frozen protocol. No media decoding, inference,
training, threshold search, task modification or automatic deployment occurs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def mapping(value):
    return value if isinstance(value, dict) else {}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def face_check(general):
    """Observe actual face checks separately from legacy band eligibility."""
    face = mapping(general.get('face_branch'))
    audit = mapping(face.get('face_audit') or general.get('face_preprocessing'))
    coverage = mapping(audit.get('coverage'))
    frames = audit.get('frame_records')
    completed = (isinstance(frames, list) and bool(frames)
                 and all(isinstance(row, dict) and row.get('status') in {'success', 'no_face'} for row in frames))
    failures = {'detection': coverage.get('frame_failed_count'), 'crop': coverage.get('failed_count'),
                'model': coverage.get('model_failed_count')}
    rows = face.get('per_face') if isinstance(face.get('per_face'), list) else []
    scores = [row.get('score') for row in rows if isinstance(row, dict)]
    total = coverage.get('detected_total')
    status = 'unrecorded'
    if completed and coverage.get('count_status') == 'recorded' and type(total) is int:
        if any(type(value) is int and value > 0 for value in failures.values()):
            status = 'error'
        elif total == 0 and not rows and face.get('status') == 'no_face':
            status = 'no_face'
        elif total > 0 and scores and all(finite(value) for value in scores):
            # A legacy no_face aggregation can coexist with scored small faces
            # excluded by the unchanged 40-pixel main-face band rule.
            status = 'success'
        else:
            status = 'invalid'
    if face.get('status') in {'error', 'failed', 'timeout'}:
        status = 'error'
    return {'status': status, 'detected_total': total, 'detection_completed': bool(completed),
            'coverage_status': coverage.get('count_status', 'unrecorded'), 'scores': scores,
            'failure_counts': failures, 'legacy_aggregation_status': face.get('status'),
            'used_for_band_count': coverage.get('used_for_band_count'),
            'small_faces': sum(bool(row.get('too_small')) for row in rows if isinstance(row, dict))}


def timing_summary(values):
    values = sorted(float(value) for value in values if finite(value) and value >= 0)
    if not values:
        return {'count': 0, 'median_seconds': None, 'p95_seconds': None}
    return {'count': len(values), 'median_seconds': statistics.median(values),
            'p95_seconds': values[max(0, math.ceil(.95 * len(values)) - 1)],
            'p95_method': 'nearest_rank', 'minimum_seconds': values[0], 'maximum_seconds': values[-1]}


def verified_file(root, name, records):
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError('unsafe task artifact path')
    if name not in records or sha(path) != records[name]['sha256']:
        raise ValueError('missing or changed structured report artifact: ' + name)
    return read(path)


def evaluate(envelope_path, output, *, include_candidates=False):
    envelope_path, output = Path(envelope_path), Path(output)
    envelope = read(envelope_path)
    if envelope.get('schema_version') != 'evaluation.envelope.v1':
        raise ValueError('explicit evaluation envelope required')
    experiment = envelope['experiment']
    if experiment['stage'] == 'acceptance' and include_candidates:
        raise ValueError('acceptance never selects or compares new candidates')
    for path_key, hash_key in [('config_path', 'config_sha256'), ('input_manifest_path', 'input_manifest_sha256'),
                               ('receipt_path', 'receipt_sha256'), ('protocol_path', 'protocol_sha256')]:
        if sha(envelope[path_key]) != envelope[hash_key]:
            raise ValueError('evaluation input binding changed: ' + path_key)
    config = read(envelope['config_path'])
    release = Path(envelope['source_root']).resolve()
    product = release / '系统'
    if Path(config['execution_profile']['product_root']).resolve() != product:
        raise ValueError('configuration does not use frozen source')
    sys.path.insert(0, str(product))
    from tools.run_v4_dev_batch import verify_release
    from contracts.execution_profile_v4 import trusted_policy_authority
    from modules.data_admission import AdmittedDataset
    from contracts.task_contract_v4 import canonical_digest, rule_source_hashes, validate_saved_decision
    from decision.rule_candidate_v4 import verify_seal, apply_rule_candidate
    from decision.candidate_evaluation_v4 import evaluate_candidate, paired_differences
    from reporting.analysis_recovery import _load_checkpoint, report_config_digest
    from reporting.llm_client import load_llm_settings
    from artifact_io import atomic_write_json
    verify_release(release, envelope['source_manifest_sha256'])
    verify_seal(experiment)
    binding = config['execution_profile']['query_admission']
    AdmittedDataset(trusted_policy_authority(), envelope['input_manifest_path'], envelope['receipt_path'],
                    expected_purpose=binding['purpose'], expected_frozen_release=config['execution_profile'].get('frozen_release'))
    if experiment['stage'] == 'acceptance' and (binding['purpose'] != 'acceptance' or not config['execution_profile'].get('frozen_release')):
        raise ValueError('acceptance release and receipt required')
    batch = Path(envelope['batch_path']).resolve()
    plan = read(batch / 'plan.json')
    if plan['config_sha256'] != envelope['config_sha256'] or plan['input_manifest_sha256'] != envelope['input_manifest_sha256']:
        raise ValueError('batch differs from predeclared evaluation')
    if plan['source_manifest_sha256'] != envelope['source_manifest_sha256'] or plan['denominator'] != len(experiment['manifest']):
        raise ValueError('batch source or denominator mismatch')
    tasks_root = Path(config['runtime']['tasks_root']).resolve()
    settings = load_llm_settings(config, environ={})
    candidates = [candidate for candidate in experiment['candidates'] if include_candidates or candidate['policy'] == 'baseline']
    mapped = {candidate['digest']: [] for candidate in candidates}
    candidate_artifacts = []
    traces, elapsed, report_times, model_times = [], [], [], []
    task_status_counts = {}
    for row in experiment['manifest']:
        index = int(row['sample_id'])
        record_path = batch / 'items' / f'{index:04d}.result.json'
        if not record_path.exists():
            traces.append({'sample_id': row['sample_id'], 'status': 'missing_result_retained'})
            continue
        record = read(record_path)
        if record.get('index') != index or record.get('input_sha256') != row['input_sha256']:
            raise ValueError('item binding differs from fixed manifest')
        if record.get('plan_digest') != canonical_digest(plan):
            raise ValueError('item belongs to a different batch plan')
        elapsed.append(record.get('elapsed_seconds'))
        trace = {'sample_id': row['sample_id'], 'result_sha256': sha(record_path), 'mapping_errors': []}
        modules, final, evidence = {}, None, None
        task_status_counts[record.get('status', 'unrecorded')] = task_status_counts.get(record.get('status', 'unrecorded'), 0) + 1
        if record.get('task_dir'):
            root = Path(record['task_dir']).resolve()
            if not root.is_relative_to(tasks_root):
                raise ValueError('task outside frozen output root')
            if sha(root / 'task_status.json') != record['task_status_sha256']:
                raise ValueError('saved task status changed')
            try:
                artifact_manifest = read(root / 'analysis_artifact_manifest.json')
                records = {item['path']: item for item in artifact_manifest['artifacts']}
                report = verified_file(root, 'analysis_source_report.json', records)
                package = verified_file(root, 'evidence_package.json', records)
                final = verified_file(root, 'final_decision.json', records)
                analysis = verified_file(root, 'analysis_report.json', records)
                checkpoint = _load_checkpoint(root, 'decision', source_digest=canonical_digest(report),
                    config_digest=report_config_digest(config, settings))
                evidence = checkpoint['payload']['decision_input']
                validate_saved_decision(final, evidence, requested_mode='rule', source_hashes=rule_source_hashes())
                if checkpoint['payload']['final_decision'] != final or analysis['final_decision'] != final:
                    raise ValueError('saved decisions disagree')
                general = read(root / 'general_aigc_result.json')
                task = read(root / 'task_status.json')
                if task.get('config_digest') != plan['config_digest']:
                    raise ValueError('task configuration differs from frozen batch')
                inference = read(root / 'stage_receipts' / f"{task['task_id']}.{task['attempt_id']}.inference.json")
                rows = [item for item in inference['artifacts'] if item.get('path') == 'general_aigc_result.json']
                if len(rows) != 1 or rows[0]['sha256'] != sha(root / 'general_aigc_result.json') or inference['input_digest'] != row['input_sha256']:
                    raise ValueError('general result or input receipt mismatch')
                modules['general_aigc'] = {'status': general.get('status', 'unrecorded'), 'score': general.get('logit', general.get('score'))}
                modules['face_branch_check'] = face_check(general)
                modules['whole_image_auxiliary'] = {'status': task.get('inference_completion', {}).get('deepfake', {}).get('status', 'unrecorded')}
                modules['canonical_decision'] = {'status': 'success', 'evidence_digest': final['evidence_digest']}
                modules['evidence_report'] = {'status': 'success', 'persisted': True, 'verified': True, 'evidence_digest': final['evidence_digest']}
                model_times.append(general.get('runtime_seconds'))
                report_times.append(task.get('reports', {}).get('runtime_seconds'))
                trace.update(status='verified_structured_evidence', task_id=task['task_id'], final_decision_sha256=sha(root / 'final_decision.json'),
                    inference_receipt_sha256=sha(root / 'stage_receipts' / f"{task['task_id']}.{task['attempt_id']}.inference.json"),
                    pdf_status='available' if (root / '研判分析报告.pdf').is_file() else 'missing')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                trace['mapping_errors'].append(type(exc).__name__ + ': ' + str(exc))
                trace['status'] = 'structured_evidence_incomplete'
                # Retain a bound saved band only if final itself was verified.
                modules['evidence_report'] = {'status': 'error', 'persisted': False, 'verified': False}
        else:
            trace.update(status='execution_failed_without_task', error_type=record.get('error_type'), error_message=record.get('error_message'))
        for candidate in candidates:
            candidate_final = deepcopy(final)
            candidate_modules = deepcopy(modules)
            if candidate_final is not None and candidate['policy'] != 'baseline':
                try:
                    baseline = {**candidate_final, 'release_sha256': experiment['release_sha256']}
                    candidate_final = apply_rule_candidate(baseline, evidence, candidate,
                        enabled=True, experimental=True)
                    destination = output.parent / (output.stem + '.candidate_decisions') / candidate['policy'] / (row['sample_id'] + '.json')
                    atomic_write_json(destination, {'final_decision': candidate_final, 'decision_input': evidence,
                        'original_final_sha256': trace.get('final_decision_sha256'), 'candidate_digest': candidate['digest'],
                        'experiment_digest': experiment['digest'], 'scope': 'offline_counterfactual_not_original_task_report'},
                        role='offline_candidate_decision', overwrite=False)
                    candidate_artifacts.append({'path': str(destination), 'sha256': sha(destination), 'sample_id': row['sample_id'], 'policy': candidate['policy']})
                except (ValueError, TypeError, KeyError) as exc:
                    candidate_modules['canonical_decision'] = {'status': 'error'}
                    candidate_modules['evidence_report'] = {'status': 'error', 'persisted': False, 'verified': False}
                    trace['mapping_errors'].append(candidate['policy'] + ': ' + str(exc))
                    candidate_final = None
            band = candidate_final.get('final_band') if candidate_final else 'undetermined'
            for name in ('canonical_decision', 'evidence_report'):
                candidate_modules.setdefault(name, {'status': 'missing'})
                candidate_modules[name].update(candidate_digest=candidate['digest'])
            candidate_modules['canonical_decision'].update(final_band=band, release_sha256=experiment['release_sha256'])
            mapped[candidate['digest']].append({'sample_id': row['sample_id'], 'input_sha256': row['input_sha256'],
                'candidate_digest': candidate['digest'], 'release_sha256': experiment['release_sha256'],
                'experiment_digest': experiment['digest'], 'final_band': band, 'modules': candidate_modules})
        traces.append(trace)
    reports = {candidate['policy']: evaluate_candidate(experiment, candidate['digest'], mapped[candidate['digest']]) for candidate in candidates}
    comparisons = {key: paired_differences(reports['baseline'], value) for key, value in reports.items() if key != 'baseline'}
    result = {'schema_version': 'batch.evaluation.v1', 'envelope_sha256': sha(envelope_path),
        'adapter_sha256': sha(__file__), 'batch_plan_sha256': sha(batch / 'plan.json'), 'reports': reports,
        'paired_differences': comparisons, 'mapping_audit': traces, 'mapped_results': mapped,
        'counterfactual_artifacts': candidate_artifacts,
        'task_status_distribution': task_status_counts, 'performance': {'task_elapsed': timing_summary(elapsed),
        'general_including_face_preprocessing': timing_summary(model_times), 'report_rendering': timing_summary(report_times)},
        'selected_policy': 'baseline', 'selection_status': 'experimental_paths_remain_disabled',
        'selection_reason': 'Required image/video truth classes and independently labeled face subgroups are not all available; no deployment selection is justified.',
        'human_review_time': 'not_measured', 'training_allowed': False, 'original_task_files_modified': False,
        'candidate_evidence_note': 'Counterfactual candidate decisions are saved in this evaluation artifact; original task reports remain baseline.'}
    atomic_write_json(output, result, role='independent_batch_evaluation', overwrite=False)
    print(json.dumps({'output': str(output), 'status': reports['baseline']['status'], 'groups': len(reports['baseline']['groups']),
        'mapping_errors': sum(bool(row.get('mapping_errors')) for row in traces), 'selected_policy': 'baseline'}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--include-candidates', action='store_true')
    args = parser.parse_args()
    evaluate(args.freeze, args.output, include_candidates=args.include_candidates)


if __name__ == '__main__':
    main()
