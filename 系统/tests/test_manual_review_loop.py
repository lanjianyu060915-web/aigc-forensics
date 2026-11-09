"""人工判断与机器结论分离，候选依据保存为当时快照。"""

import hashlib
import json
import io
import zipfile
from pathlib import Path
from PIL import Image
import pytest

from reporting.review_records import append_review_record, load_review_records, MANUAL_JUDGMENTS, ReviewRecordError
from reporting.report_builder import generate_task_reports
from reporting import pdf_report


def prepare(tmp_path):
    output = generate_task_reports(tmp_path, task_id='人工判断')
    report = output['report']
    report['forgery_type'] = None
    report['pair_compare'] = {'conclusion_enabled': False, 'candidates': [
        {'rank': 1, 'candidate_name': '参考图.jpg', 'candidate_sha256': 'a' * 64,
         'comparison': {'alignment_status': 'completed', 'faces': [{'query_face_id': 0, 'match_status': 'matched', 'cosine_similarity': .6}]},
         'artifacts': {key: key+'.png' for key in ('reference','aligned','difference')}}]}
    for name in ('reference','aligned','difference'):
        Image.new('RGB',(80,50),'blue').save(tmp_path/(name+'.png'))
    (tmp_path/'final_report.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8')
    return report


def test_manual_fields_are_append_only_snapshots(tmp_path):
    report = prepare(tmp_path)
    from orchestrator import _task_artifact_manifest
    from reporting.report_builder import report_file_names
    from frontend.components.task_history import verify_task_artifacts, build_evidence_bundle
    manifest = _task_artifact_manifest(tmp_path, '人工判断', {}, report_files=report_file_names('人工判断'))
    (tmp_path/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8')
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir() if p.is_file()}
    record = append_review_record(tmp_path, '复核员', '无法判断', '<b>&需核查',
                                  manual_judgment=MANUAL_JUDGMENTS[1], basis_candidate_rank=1)
    assert record['basis_candidate_name'] == '参考图.jpg'
    assert record['basis_candidate_sha256'] == 'a'*64
    assert load_review_records(tmp_path) == [record]
    raw = pdf_report.render_pdf_report(report,tmp_path,[record])
    assert raw.startswith(b'%PDF-')
    assert before == {name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in before}
    stored = json.loads((tmp_path/'final_report.json').read_text(encoding='utf-8'))
    assert stored['forgery_type'] is None and stored['pair_compare']['conclusion_enabled'] is False
    assert not verify_task_artifacts(tmp_path)['problems']
    with zipfile.ZipFile(io.BytesIO(build_evidence_bundle(tmp_path))) as archive:
        assert f'{tmp_path.name}/review_records.jsonl' in archive.namelist()


@pytest.mark.parametrize('kwargs', [{'manual_judgment':'确认'}, {'basis_candidate_rank':4}, {'basis_candidate_rank':True}, {'basis_candidate_rank':2}, {'target_face_index':7}])
def test_invalid_manual_inputs_do_not_append(tmp_path,kwargs):
    prepare(tmp_path)
    with pytest.raises(ReviewRecordError):
        append_review_record(tmp_path,'复核员','无法判断','意见',**kwargs)
    assert not (tmp_path/'review_records.jsonl').exists()


def test_old_record_and_old_report_remain_supported(tmp_path):
    report = prepare(tmp_path)
    record = {'reviewer':'旧复核员','conclusion':'无法判断','opinion':'旧意见','reviewed_at':'旧时间','report_sha256':hashlib.sha256((tmp_path/'final_report.json').read_bytes()).hexdigest()}
    (tmp_path/'review_records.jsonl').write_text(json.dumps(record,ensure_ascii=False)+'\n',encoding='utf-8')
    assert load_review_records(tmp_path) == [record]
    assert pdf_report.render_pdf_report(report,tmp_path,[record]).startswith(b'%PDF-')
    assert pdf_report._pair_section({},tmp_path,pdf_report._styles()) == []


def test_pdf_pair_images_and_failed_alignment(tmp_path):
    report = prepare(tmp_path)
    blocks = pdf_report._pair_section(report,tmp_path,pdf_report._styles())
    text = '\n'.join(x.getPlainText() for x in blocks if hasattr(x,'getPlainText'))
    assert '差异热图只提示像素变化位置' in text and '几何对应不等于身份绑定' in text
    report['pair_compare']['candidates'][0]['comparison']['alignment_status']='failed'
    failed = pdf_report._pair_section(report,tmp_path,pdf_report._styles())
    assert not any(isinstance(x,pdf_report.Table) for x in failed)
    assert any('对齐失败，暂不能判断是否同源' in x.getPlainText() for x in failed if hasattr(x,'getPlainText'))
