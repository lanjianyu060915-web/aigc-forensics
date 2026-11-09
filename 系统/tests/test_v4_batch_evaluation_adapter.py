"""Independent structured fixtures only; no media, task scans or inference."""
from copy import deepcopy
import pytest
from tools.evaluate_v4_batch import face_check, timing_summary


def general(total=0, scores=(), status='no_face'):
    return {'face_branch': {'status': status, 'per_face': [{'score': value} for value in scores],
        'face_audit': {'coverage': {'count_status': 'recorded', 'detected_total': total, 'frame_failed_count': 0,
                                  'failed_count': 0, 'model_failed_count': 0, 'used_for_band_count': 0},
                       'frame_records': [{'status': 'no_face' if total == 0 else 'success'}]}}}


def test_no_face_requires_explicit_zero_and_completed_detection():
    assert face_check(general())['status'] == 'no_face'
    value = general(2)
    assert face_check(value)['status'] == 'invalid'
    del value['face_branch']['face_audit']
    assert face_check(value)['status'] == 'unrecorded'


def test_informational_face_scores_remain_checks_when_not_used_for_band():
    result = face_check(general(2, [0.1, 0.9]))
    assert result['status'] == 'success'
    assert result['legacy_aggregation_status'] == 'no_face'
    assert result['used_for_band_count'] == 0 and result['scores'] == [0.1, 0.9]


@pytest.mark.parametrize('part', ['frame_failed_count', 'failed_count', 'model_failed_count'])
def test_partial_face_failures_are_not_hidden_by_success(part):
    value = general(2, [0.1], 'success')
    value['face_branch']['face_audit']['coverage'][part] = 1
    assert face_check(value)['status'] == 'error'


def test_failed_detector_and_nonfinite_scores_are_invalid():
    value = general()
    value['face_branch']['face_audit']['frame_records'] = [{'status': 'detection_failed'}]
    assert not face_check(value)['detection_completed']
    assert face_check(general(1, [float('nan')], 'success'))['status'] == 'invalid'
    assert face_check(general(1, [False], 'success'))['status'] == 'invalid'


def test_missing_timings_are_not_zero_and_p95_is_declared():
    assert timing_summary([None, float('nan')])['median_seconds'] is None
    value = timing_summary([1, 2, 3, 4, None])
    assert value['count'] == 4 and value['median_seconds'] == 2.5 and value['p95_seconds'] == 4
