"""展示最近未发送与此前送达未知，保留原审计数据和其它正常状态。"""
from copy import deepcopy

import pytest

from reporting import analysis_pdf as pdf


def audit(tmp_path,stage,block,transmission=None):
    package={'llm_processing':{'upload_scope':'full','judge_send_images':True,'stages':{}}}
    if transmission is not None:package['llm_processing']['stages'][stage]=transmission
    decision=block if stage=='judge' else {}
    writer=block if stage=='writer' else {}
    before=deepcopy((decision,writer,package))
    out=dict(pdf._audit_rows(tmp_path,decision,writer,package))
    assert (decision,writer,package)==before
    return out


@pytest.mark.parametrize('stage,title',[('judge','裁决'),('writer','撰写')])
@pytest.mark.parametrize('saved_note',[None,'未发送请求','预算闸门阻断最近一次调用'])
def test_latest_not_sent_preserves_prior_unknown(tmp_path,stage,title,saved_note):
    transmission={'transmission_status':'not_sent','delivery_unknown_figure_ids':['F01','F03'],
                  'attempted_figure_ids':['F01','F03'],'uploaded_figure_ids':[]}
    if saved_note is not None:transmission['transmission_note']=saved_note
    out=audit(tmp_path,stage,{'status':'fallback','reason':'调度已停止，新物理请求未发送'},transmission)
    expected='此前图件已尝试发送、送达未知；最近一次调用未发送'
    if saved_note is not None:expected+='；原记录说明：'+saved_note
    assert out[title+'图件传输状态']==expected
    assert out[title+'送达未知图件数量/编号']=='2 / F01、F03'
    assert out[title+'已尝试发送图件数量/编号']=='2 / F01、F03'
    assert out[title+'已有返回或缓存支持的图件数量/编号']=='0 / 无'
    assert out[title+'校验/退回原因']['reason']=='调度已停止，新物理请求未发送'


def test_legacy_block_fallback_without_stage_record(tmp_path):
    block={'transmission_status':'not_sent','delivery_unknown_figure_ids':['F01']}
    out=audit(tmp_path,'writer',block)
    assert out['撰写图件传输状态']=='此前图件已尝试发送、送达未知；最近一次调用未发送'


@pytest.mark.parametrize('unknown',[None,[],False,'F01'])
def test_no_unknown_list_does_not_invent_prior_sending(tmp_path,unknown):
    out=audit(tmp_path,'writer',{}, {'transmission_status':'not_sent','delivery_unknown_figure_ids':unknown})
    assert out['撰写图件传输状态']=='未发送请求'


@pytest.mark.parametrize('status,expected',[
    ('delivery_unknown','已尝试发送，送达未知'),('response_received','接口已有返回'),
    ('cached_response','复用已保存响应，本次未新发送'),(None,'未记录'),('other','未记录')])
@pytest.mark.parametrize('unknown',[[],['F01']])
def test_other_statuses_keep_existing_display(tmp_path,status,expected,unknown):
    out=audit(tmp_path,'writer',{}, {'transmission_status':status,'delivery_unknown_figure_ids':unknown})
    assert out['撰写图件传输状态']==expected


@pytest.mark.parametrize('status',['not_sent','delivery_unknown','response_received','cached_response',None])
def test_existing_note_precedence_unchanged_without_prior_unknown(tmp_path,status):
    out=audit(tmp_path,'writer',{}, {'transmission_status':status,'transmission_note':'已有说明','delivery_unknown_figure_ids':[]})
    assert out['撰写图件传输状态']=='已有说明'


def test_stage_record_precedence_remains(tmp_path):
    block={'transmission_status':'not_sent','delivery_unknown_figure_ids':['F99']}
    out=audit(tmp_path,'writer',block,{'transmission_status':'response_received','delivery_unknown_figure_ids':[]})
    assert out['撰写图件传输状态']=='接口已有返回'
