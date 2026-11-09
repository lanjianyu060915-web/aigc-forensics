"""并行调度的启动屏障、失败隔离和状态发布。"""
from __future__ import annotations
import json
from pathlib import Path
import threading
import time

import pytest
import orchestrator as backend
from test_orchestrator_contract import _write_config


def _prepare(tmp_path, monkeypatch, *, parallel=True):
    config_path=_write_config(tmp_path)
    config=json.loads(config_path.read_text('utf-8'))
    config['runtime']['parallel_modules']=parallel
    config_path.write_text(json.dumps(config),encoding='utf-8')
    query=tmp_path/'查询.jpg'
    query.write_bytes(b'fixture')
    monkeypatch.setattr(backend,'analyze_visible_labels',lambda inputs: backend.visible_label_unavailable('测试中不加载文字识别模型'))
    return config_path,query


def _success(name,kwargs):
    payload=(backend._standard_deepfake_result(kwargs['query_path'],0,None,status='success')
             if name=='deepfake' else backend._standard_search_result(kwargs['query_path'],0,None,status='success'))
    return backend._finish_module(name=name,payload=payload,output_path=kwargs['output_path'],
        logs_dir=kwargs['logs_dir'],started_at=backend._now_iso(),started=time.perf_counter(),
        returncode=0,timed_out=False,execution_mode='测试替身')


def test_parallel_starts_all_modules_and_waits_before_pairing(tmp_path,monkeypatch):
    config_path,query=_prepare(tmp_path,monkeypatch)
    config=json.loads(config_path.read_text('utf-8'));config['pair_compare']={'enabled':True}
    config_path.write_text(json.dumps(config),encoding='utf-8')
    barrier=threading.Barrier(3,timeout=10)
    finished=[]
    guard=threading.Lock()
    def runner(name,kwargs):
        barrier.wait()
        result=_success(name,kwargs)
        with guard: finished.append(name)
        return result
    monkeypatch.setattr(backend,'_run_deepfake_module',lambda **kwargs:runner('deepfake',kwargs))
    monkeypatch.setattr(backend,'_run_search_module',lambda name,**kwargs:runner(name,kwargs))
    def pair(**kwargs):
        assert sorted(finished)==sorted(backend.MODULE_ORDER)
        return {'status':'skipped','candidates':[],'conclusion_enabled':False,'forgery_type':None}
    monkeypatch.setattr(backend,'run_source_pair',pair)
    status=backend.run_task(query,config_path=config_path)
    assert status['status']=='success'
    assert status['reports']['status']=='success'
    assert all(status['modules'][name]['status']=='success' for name in backend.MODULE_ORDER)
    events=[json.loads(row) for row in (Path(status['task_dir'])/'logs/progress_events.jsonl').read_text('utf-8').splitlines()]
    completed=[row['progress_current'] for row in events if row['stage']=='modules_completed']
    assert completed==[1,2,3]
    legacy=json.loads((Path(status['task_dir'])/'logs/progress.json').read_text('utf-8'))
    assert len(legacy)==len(events)


def test_disabled_parallel_keeps_original_order(tmp_path,monkeypatch):
    config_path,query=_prepare(tmp_path,monkeypatch,parallel=False)
    order=[]
    def runner(name,kwargs):
        order.append(name)
        return _success(name,kwargs)
    monkeypatch.setattr(backend,'_run_deepfake_module',lambda **kwargs:runner('deepfake',kwargs))
    monkeypatch.setattr(backend,'_run_search_module',lambda name,**kwargs:runner(name,kwargs))
    assert backend.run_task(query,config_path=config_path)['status']=='success'
    assert order==list(backend.MODULE_ORDER)


@pytest.mark.parametrize('bad',[None,0,'true'])
def test_parallel_setting_requires_boolean(tmp_path,monkeypatch,bad):
    config_path,query=_prepare(tmp_path,monkeypatch,parallel=bad)
    with pytest.raises(backend.OrchestrationError,match='parallel_modules'):
        backend.run_task(query,config_path=config_path)


def test_identity_timeout_and_source_error_keep_detection_and_report(tmp_path,monkeypatch):
    config_path,query=_prepare(tmp_path,monkeypatch)
    config=json.loads(config_path.read_text('utf-8'))
    config['identity']['timeout_seconds']=0.12
    config_path.write_text(json.dumps(config),encoding='utf-8')
    original=backend._run_search_module
    def search(name,**kwargs):
        if name=='source': raise RuntimeError('模拟素材模块错误')
        return original(name=name,**kwargs)
    def slow_identity(**kwargs):
        time.sleep(0.4)
        return backend._standard_search_result(kwargs['query_path'],0,None,status='success')
    monkeypatch.setattr(backend,'_run_search_module',search)
    monkeypatch.setattr(backend,'_run_deepfake_module',lambda **kwargs:_success('deepfake',kwargs))
    status=backend.run_task(query,config_path=config_path,services={'run_identity':slow_identity})
    assert status['identity']['timed_out'] is True
    assert status['identity']['execution_mode']=='deadline_exhausted'
    assert status['source']['status']=='error'
    assert status['deepfake']['status']=='success'
    assert status['reports']['status']=='success'
    assert (Path(status['task_dir'])/'final_report.json').is_file()


def test_parallel_only_runs_selected_modules(tmp_path,monkeypatch):
    config_path,query=_prepare(tmp_path,monkeypatch)
    called=[]
    def runner(name,**kwargs):
        called.append(name)
        return _success(name,kwargs)
    monkeypatch.setattr(backend,'_run_search_module',runner)
    status=backend.run_task(query,config_path=config_path,selected_modules=['source'])
    assert called==['source']
    assert status['identity']['status']==status['deepfake']['status']=='skipped'
    assert status['source']['status']=='success'
