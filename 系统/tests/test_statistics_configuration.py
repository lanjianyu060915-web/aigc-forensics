"""总体统计只供匹配的人脸配置展示，原机器结果不变。"""
from copy import deepcopy
import json

import pytest

from decision import aigc_evidence as evidence
from reporting import pdf_report
from reporting.report_builder import build_final_report


def thresholds():
    return deepcopy(json.loads(evidence._FACE_CALIBRATION_PATH.read_text(encoding="utf-8"))["face_crop"])


def general(**changes):
    face = {"backend": "cf384", "band_aggregation": "largest", "status": "success", "band_thresholds": thresholds()}
    face.update(changes)
    return {"status": "success", "band": "low", "logit": -3.0, "face_branch": face}


def test_matching_default_uses_original_statistics_without_mutation():
    raw={"overall":{"band":"low"},"general_aigc":general(),"score":0.2}
    before=deepcopy(raw)
    enriched=evidence.attach_evidence(raw,{})
    assert enriched['overall']['band_evidence']==evidence.get_band_evidence('low')
    assert raw==before and enriched['score']==raw['score']


@pytest.mark.parametrize('backend,aggregation', [('gend','largest'),('cf384','max_all'),('unknown','largest'),('', 'largest'),(None,'largest'),('cf384',None),('cf384','')])
@pytest.mark.parametrize('no_face',[False,True])
def test_explicit_nondefault_and_unknown_never_reuse_statistics(backend,aggregation,no_face):
    g=general(backend=backend,band_aggregation=aggregation)
    if no_face:
        g['face_branch']['status']='no_face';g['face_branch'].pop('band_thresholds')
    assert not evidence.legacy_overall_statistics_match(g)
    out=evidence.attach_evidence({'general_aigc':g,'overall':{'band':'low'}},{})
    assert out['overall']['band_evidence']['status']=='unavailable'
    assert out['overall']['band_evidence']['rows']==[]
    assert evidence.UNAVAILABLE_NOTE in evidence.evidence_lines(out['overall'])
    assert not any('三分之二' in s for s in evidence.evidence_lines(out['overall']))


@pytest.mark.parametrize('key',['backend','band_aggregation'])
def test_partial_configuration_is_not_legacy(key):
    g=general();g['face_branch'].pop(key)
    assert not evidence.legacy_overall_statistics_match(g)


@pytest.mark.parametrize('value',[None,{},[],False,'bad'])
def test_explicit_invalid_face_branch_is_not_absent_legacy(value):
    assert not evidence.legacy_overall_statistics_match({'face_branch':value})


def test_fully_absent_face_key_is_explicit_old_structure_compatibility():
    assert evidence.legacy_overall_statistics_match({})
    assert evidence.legacy_overall_statistics_match({'status':'success'})
    assert not evidence.legacy_overall_statistics_match(None)
    assert evidence.attach_evidence({'overall':{'band':'low'},'general_aigc':None},{})['overall']['band_evidence']['status']=='unavailable'


@pytest.mark.parametrize('value',[None,{}, {'low_max':-4.949833}, {'low_max':True,'medium_max':-4.210364}, {'low_max':float('nan'),'medium_max':-4.210364}, {'low_max':-4.949833,'medium_max':float('inf')}, {'low_max':-4.949833+1e-12,'medium_max':-4.210364}])
def test_thresholds_are_complete_finite_and_exact(value):
    assert not evidence.legacy_overall_statistics_match(general(band_thresholds=value))


def test_default_no_face_missing_threshold_exception_is_narrow():
    g=general(status='no_face');g['face_branch'].pop('band_thresholds')
    assert evidence.legacy_overall_statistics_match(g)
    g['face_branch']['status']='success'
    assert not evidence.legacy_overall_statistics_match(g)
    g['face_branch']['status']='error'
    assert not evidence.legacy_overall_statistics_match(g)
    g['face_branch']['status']='no_face';g['face_branch']['band_thresholds']={}
    assert not evidence.legacy_overall_statistics_match(g)


def test_actual_old_payload_shapes_and_new_markers_are_distinct():
    old={'status':'success','logit':1.0,'band':'high','band_label':'高风险提示','band_reason':'分数依据','band_thresholds':thresholds(),'crops':[],'records':[]}
    assert evidence.legacy_overall_statistics_match({'face_branch':old})
    no_face={'status':'no_face','logit':None,'band':None,'band_label':'未检出人脸','records':[]}
    assert evidence.legacy_overall_statistics_match({'face_branch':no_face})
    for marker in ('per_face','score_kind','model','highest_score_face'):
        modified=deepcopy(no_face);modified[marker]=None
        assert not evidence.legacy_overall_statistics_match({'face_branch':modified})
    old['band_thresholds']['low_max']+=0.01
    assert not evidence.legacy_overall_statistics_match({'face_branch':old})


@pytest.mark.parametrize('kind',['bad_json','missing','empty','wrong_type'])
def test_missing_or_invalid_frozen_reference_does_not_match(tmp_path,monkeypatch,kind):
    g=general()
    ref=tmp_path/'frozen.json'
    if kind!='missing':
        ref.write_text({'bad_json':'bad','empty':'{}','wrong_type':'{"face_crop": []}'}[kind],encoding='utf-8')
    monkeypatch.setattr(evidence,'_FACE_CALIBRATION_PATH',ref)
    assert not evidence.legacy_overall_statistics_match(g)


@pytest.mark.parametrize('backend,aggregation,expect_table',[('cf384','largest',True),('gend','largest',False),('cf384','max_all',False)])
def test_pdf_saved_wrong_table_is_guarded_without_changing_input(tmp_path,monkeypatch,backend,aggregation,expect_table):
    g=general(backend=backend,band_aggregation=aggregation)
    raw={'status':'success','score':0.2,'details':{'band':'low'},'general_aigc':g,
         'overall':{'band':'low','band_label':'证据不足','reason':'固定规则','band_evidence':evidence.get_band_evidence('low'),'hints':[],'branches':[]}}
    report=build_final_report('统计测试',task_status={'status':'success'},deepfake_result=raw,identity_result={'status':'skipped'},source_result={'status':'skipped'})
    before=deepcopy(report)
    paragraphs=[];tables=[]
    original_p,original_table=pdf_report._p,pdf_report._table
    def p(value,*args,**kwargs):
        paragraphs.append(str(value));return original_p(value,*args,**kwargs)
    def table(headers,rows,*args,**kwargs):
        tables.append(deepcopy(rows));return original_table(headers,rows,*args,**kwargs)
    monkeypatch.setattr(pdf_report,'_p',p);monkeypatch.setattr(pdf_report,'_table',table)
    output=pdf_report.render_pdf_report(report,tmp_path)
    assert output.startswith(b'%PDF-') and report==before
    assert ('当前档位的证据强度' in paragraphs) is expect_table
    assert any('三分之二' in p for p in paragraphs) is expect_table
    if not expect_table:
        assert evidence.UNAVAILABLE_NOTE in paragraphs
        assert not any('552/599' in str(t) for t in tables)
        assert '证据不足不能排除伪造，请结合其他线索人工复核。' in paragraphs
