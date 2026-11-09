"""研判三版式的事实、图件、排版和复核契约；全部使用合成测试材料。"""

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw
import pytest

from reporting import analysis_pdf as pdf
from reporting.analysis_templates import build_template_content, merge_writer_content


def mock_case(task_dir: Path, kind="ordinary"):
    task_dir.mkdir(parents=True, exist_ok=True)
    image=Image.new("RGB",(1600,1000),(225,236,244));draw=ImageDraw.Draw(image)
    draw.rectangle((100,180,1400,850),fill=(90,130,100));draw.rectangle((650,220,950,600),fill=(186,132,100))
    image.save(task_dir/"original.png")
    image.resize((160,100)).save(task_dir/"upload.jpg")
    figure={"figure_id":"F01","type":"original","module":"材料固定","required":True,"section":"fixation",
            "original_path":"original.png","upload_path":"upload.jpg","original_sha256":sha256((task_dir/"original.png").read_bytes()).hexdigest(),
            "sha256":sha256((task_dir/"upload.jpg").read_bytes()).hexdigest(),"description":"合成几何画面，仅用于报告排版验收。"}
    face=kind in ("face","video","source_r1")
    figures=[figure]
    if face:
        image.crop((650,220,950,600)).save(task_dir/"face.png")
        figures.append({**figure,"figure_id":"F02","type":"face_crop","module":"人脸检测","section":"detection","original_path":"face.png","original_sha256":sha256((task_dir/"face.png").read_bytes()).hexdigest()})
        figures[0].update(bbox=[650,220,950,600],face_index=1)
    if kind=="source_r1":
        for i,typ in enumerate(("source_candidate","aligned_pair","difference_heatmap"),3):
            changed=image.copy();ImageDraw.Draw(changed).rectangle((650,220,950,600),fill=(40*i,35*i,25*i));changed.save(task_dir/f"source_{i}.png")
            figures.append({**figure,"figure_id":f"F{i:02d}","type":typ,"module":"素材配对","section":"source","original_path":f"source_{i}.png","original_sha256":sha256((task_dir/f"source_{i}.png").read_bytes()).hexdigest()})
    material={"status":"success","task_id":"模拟验收_"+kind,"file_name":"合成测试画面.png","type":"video" if kind=="video" else "image","input_sha256":"a"*64,"fixed":True,"width":1600,"height":1000,"size_bytes":12345,"duration_seconds":4 if kind=="video" else None}
    branch={"status":"success","band":"low","score":-2.123456,"interval":"低于95%分界线","thresholds":{"low_max":-2,"medium_max":-1},"model":{"name":"模拟检测模型"},"statistics":{"status":"unavailable"},"known_reliability":"模拟数据不代表检测性能。"}
    face_data={**branch,"status":"success" if face else "no_face","score":-1.8 if face else None,"backend":"模拟后端","per_face":[{"face_index":1,"score":-1.8,"band":"medium","bbox":[650,220,950,600]}] if face else []}
    if kind=="video":face_data["frame_statistics"]={"status":"recorded","n":3,"min":-3,"median":-2,"max":-1.8,"above_low_max_ratio":1/3,"above_medium_max_ratio":0}
    source={"status":"success","open_set_state":"review","candidates":[{"rank":1,"candidate_name":"模拟底图.png","global_score":.81,"local_score":.7,"final_score":.75,"open_set_state":"review"}]} if kind=="source_r1" else {"status":"skipped"}
    pair={"status":"success","candidates":[{"rank":1,"candidate_name":"模拟底图.png","alignment_status":"success","metrics":{"background_ssim":.98,"face_diff":.55},"faces":{"relation":"不同"},"reason":"模拟背景相同、人脸区域不同"}]} if kind=="source_r1" else {"status":"skipped"}
    process={"status":"success","analysis_time":"2026-11-06T20:00:00+08:00（模拟）","selected_modules":["通用检测","人脸检测"],"key_frames":[{"frame_index":i,"timestamp_seconds":i,"reason":"模拟关键帧"} for i in (0,2,4)] if kind=="video" else []}
    data={"E01":{"status":"success","declared_generated":False},"E02":{"status":"success","visible_label_found":False},"E03":branch,"E04":face_data,"E05":{**branch,"status":"skipped","score":None},"E06":source,"E07":{"status":"skipped","open_set_state":"unavailable"},"E08":pair,"E09":material,"E10":process}
    package={"items":[{"item_id":key,"title":key,"data":value} for key,value in data.items()],"figures":figures,"rule_band":"review","system_version":"模拟版本","configuration":{"face_backend":"模拟后端","band_aggregation":"max"},"manifest_sha256":"b"*64,"clues":[{"rule_id":"R1","matched":True,"description":"背景一致而人脸不同。"}] if kind=="source_r1" else []}
    decision={"final_band":"review","rule_band":"review","score":None,"source":"rule_fallback","reason":"模拟断网降级，未调用 API"}
    return package,decision


def render(tmp_path, package=None, decision=None, content=None, **kwargs):
    if package is None:package,decision=mock_case(tmp_path)
    if content is None:content=build_template_content(package,decision)
    data=pdf.render_analysis_pdf(package,decision,content,tmp_path,product_name="模拟数据验收（非真实检测）",**kwargs)
    log=json.loads((tmp_path/("analysis_pdf_review_render.json" if kwargs.get("review") is not None else "analysis_pdf_render.json")).read_text("utf-8"))
    return data,log


def test_template_does_not_invent_score_or_long_analysis(tmp_path):
    package,decision=mock_case(tmp_path,"source_r1")
    content=build_template_content(package,decision)
    assert "未提供评分" in content["summary"]
    assert len(content["synthesis"])<800
    assert "同一地点拍摄的不同人" in content["process"]["clues"]
    assert len(content["figures"])==5


def test_partial_writer_merge_preserves_template_and_decision(tmp_path):
    package,decision=mock_case(tmp_path);before=deepcopy(decision)
    template=build_template_content(package,decision)
    result=merge_writer_content(template,{"status":"partial","invalid_sections":["synthesis"],"content":{"summary":"有效摘要。","synthesis":"失效分析。","process":{"identity":"有效身份说明。"},"final_band":"obvious","score":100}})
    assert result["summary"]=="有效摘要。"
    assert result["synthesis"]==template["synthesis"]
    assert result["process"]["identity"]=="有效身份说明。"
    assert decision==before and "score" not in result and "final_band" not in result


@pytest.mark.parametrize("kind",["ordinary","face","no_face","video","source_r1"])
def test_pdf_sections_original_resolution_and_readonly_inputs(tmp_path,kind,monkeypatch):
    package,decision=mock_case(tmp_path,kind);content=build_template_content(package,decision)
    before=deepcopy((package,decision,content));texts=[];original_p=pdf._p
    def collect(value,*a,**kw):texts.append(str(value));return original_p(value,*a,**kw)
    monkeypatch.setattr(pdf,"_p",collect)
    data,log=render(tmp_path,package,decision,content)
    assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")
    assert log["page_count"]>=4 and not log["missing_figures"]
    assert log["image_regions"][0]["original_width"]==1600
    assert b"/Width 1600" in data and b"/Height 1000" in data
    assert log["figure_numbers"]=={f["figure_id"]:i for i,f in enumerate(package["figures"],1)}
    assert (package,decision,content)==before
    for heading in ("一、基本情况","二、研判结论","三、分析过程","四、综合研判","五、评分依据","六、不确定性与使用边界","七、建议核查事项","附录一：技术信息","附录二：研判记录"):
        assert heading in texts
    assert "八、人工复核意见" not in texts


def test_bad_figures_fallback_references_and_auto_insert(tmp_path):
    package,decision=mock_case(tmp_path);template=build_template_content(package,decision)
    writer={"status":"success","uploaded_figure_ids":["F01"],"content":{"summary":"不能保留的引用[F99]。","process":{"fixation":"已核对原图[F01]。"},"figures":[{"figure_id":"F99","section":"fixation"},{"figure_id":"F01","section":"fixation","after_paragraph":999,"caption":"原图","reading_guide":"查看画面。"},{"figure_id":"F01","section":"source"}]}}
    content=merge_writer_content(template,writer)
    cleaned,placements,numbers,warnings,missing=pdf._prepare(package,decision,content,tmp_path.resolve(),None)
    assert cleaned["summary"]==template["summary"] and numbers=={"F01":1}
    assert placements[0]["after_paragraph"]==1 and not missing
    assert {w["code"] for w in warnings}>={"unknown_figure","duplicate_figure","invalid_reference_section_fallback"}
    assert pdf._replace_refs("见[F01]。",numbers)=="见图1。"
    content["figures"]=[{"figure_id":"F01","section":"fixation","caption":"原图","reading_guide":"说明"}];content["_writer"]["uploaded_figure_ids"]=[]
    _,placements,_,warnings,_=pdf._prepare(package,decision,content,tmp_path.resolve(),None)
    assert placements[0]["automatic"] and any(w["code"]=="not_uploaded" for w in warnings)


@pytest.mark.parametrize("failure",["hash","outside","missing"])
def test_invalid_original_never_uses_valid_upload(tmp_path,failure):
    package,decision=mock_case(tmp_path)
    if failure=="hash":package["figures"][0]["original_sha256"]="0"*64
    elif failure=="outside":
        outside=tmp_path.parent/(tmp_path.name+"_outside.png");outside.write_bytes((tmp_path/"original.png").read_bytes());package["figures"][0]["original_path"]=str(outside)
    else:package["figures"][0]["original_path"]="missing.png"
    data,log=render(tmp_path,package,decision)
    assert log["missing_figures"][0]["figure_id"]=="F01"
    assert not log["image_regions"] and b"/Width 160" not in data


def test_three_line_table_has_no_grid_or_color(tmp_path):
    pdf._fonts();table=pdf._three_line_table(["列一","列二"],[["事实","依据"]],[100,200],pdf._styles())
    assert [(c[0],c[1],c[2],c[3]) for c in table._linecmds]==[("LINEABOVE",(0,0),(-1,0),1),("LINEBELOW",(0,0),(-1,0),.5),("LINEBELOW",(0,-1),(-1,-1),1)]
    assert all(command[4]==pdf.colors.black for command in table._linecmds)
    assert not table._bkgrndcmds
    assert all(style.textColor==pdf.colors.black for style in pdf._styles().values())


def test_review_appends_and_keeps_original_untouched(tmp_path,monkeypatch):
    package,decision=mock_case(tmp_path);base,_=render(tmp_path,package,decision);(tmp_path/"analysis.pdf").write_bytes(base)
    original_json=(tmp_path/"analysis_pdf_render.json").read_bytes();texts=[];original_p=pdf._p
    def collect(value,*a,**kw):texts.append(str(value));return original_p(value,*a,**kw)
    monkeypatch.setattr(pdf,"_p",collect)
    reviewed,log=render(tmp_path,package,decision,review={"reviewer":"测试复核人","opinion":"<b>为普通文字</b>","conclusion":"需补证","report_sha256":"c"*64})
    assert log["report_type"]=="analysis_review" and reviewed!=base
    assert (tmp_path/"analysis.pdf").read_bytes()==base and (tmp_path/"analysis_pdf_render.json").read_bytes()==original_json
    assert texts.index("八、人工复核意见")>texts.index("附录三：图件目录")


def test_large_images_compress_without_resolution_change(tmp_path,monkeypatch):
    package,decision=mock_case(tmp_path)
    # 固定随机像素模拟照片，保证 PNG 足够大而 JPEG 有可检验的节省。
    import random
    noise=Image.frombytes("RGB",(800,600),random.Random(12).randbytes(800*600*3));noise.save(tmp_path/"noise.png")
    package["figures"][0].update(original_path="noise.png",original_sha256=sha256((tmp_path/"noise.png").read_bytes()).hexdigest())
    monkeypatch.setattr(pdf,"MAX_PDF_BYTES",100000)
    data,log=render(tmp_path,package,decision)
    assert log["compression"]["triggered"] and log["compression"]["applied"]
    assert log["compression"]["jpeg_quality"]==90 and not log["compression"]["resolution_changed"]
    assert log["file_size_bytes"]<log["initial_file_size_bytes"]
    assert b"/Width 800" in data and b"/Height 600" in data


def test_long_tables_and_sensitive_paths_are_safe(tmp_path):
    package,decision=mock_case(tmp_path,"source_r1")
    e={row["item_id"]:row["data"] for row in package["items"]}
    e["E06"]["candidates"]=[{"rank":i,"candidate_name":"测试"*80+".png","final_score":.5,"open_set_state":"review"} for i in range(40)]
    content=build_template_content(package,decision);content["process"]["identity"]="请核对 C:\\private\\secret.jpg 与 /home/private/secret.jpg。"
    assert "C:" not in pdf._display(content["process"]["identity"])
    assert "/home/" not in pdf._display(content["process"]["identity"])
    assert pdf._display("https://example.org/a/b")=="https://example.org/a/b"
    _,log=render(tmp_path,package,decision,content)
    assert log["page_count"]>7


def test_pipeline_source_manifest_digest_is_shown(tmp_path,monkeypatch):
    package,decision=mock_case(tmp_path)
    package.pop("manifest_sha256")
    package["source_manifest"]={"file":"analysis_input_manifest.json","sha256":"d"*64}
    texts=[];original_p=pdf._p
    def collect(value,*a,**kw):texts.append(str(value));return original_p(value,*a,**kw)
    monkeypatch.setattr(pdf,"_p",collect)
    render(tmp_path,package,decision)
    assert "d"*64 in texts


def test_body_tables_summarize_nested_records_and_do_not_duplicate_identity(tmp_path,monkeypatch):
    thresholds={"low_max":-2,"medium_max":-1,"calibration_set":{"description":"校准明细"*200}}
    assert "calibration_set" not in pdf._thresholds({"thresholds":thresholds})
    assert pdf._thresholds({"thresholds":thresholds},full=True)==thresholds
    assert pdf._pair_metrics({"background":{"ssim_median":.987654,"changed_fraction":.123456},"color_normalization":[{"gain":1}]} )=="背景 SSIM：0.99；背景变化比例：0.12"
    values=[]
    monkeypatch.setattr(pdf,"_table",lambda story,title,headers,data,*a:values.append((title,data)))
    candidate={"rank":1,"identity_id":"person_001","score":.8}
    e={"E07":{"candidates":[candidate],"faces":[{"face_index":0,"candidates":[candidate],"open_set":{"state":"high_confidence"}}]}}
    pdf._module_tables([],"identity",e,pdf._styles())
    assert len(values[0][1])==1
    assert values[0][1][0][-1]=="高可信候选"


def test_highest_face_template_preserves_selection_difference(tmp_path):
    package,decision=mock_case(tmp_path,"face")
    next(e['data'] for e in package['items'] if e['item_id']=='E04')['highest_face']={"face_id":2,"score":3.4567,"used_for_band":False,"bbox":[1,2,3,4]}
    text=build_template_content(package,decision)['process']['detection']
    assert "最高分来自人脸2" in text and "该脸未用于本次分档" in text
    assert "bbox" not in text


def test_reliability_appendix_preserves_separate_denominators(tmp_path,monkeypatch):
    package,decision=mock_case(tmp_path)
    next(e['data'] for e in package['items'] if e['item_id']=='E08')['reliability_statistics']={"status":"available","basis":"开发验证，非盲测","notes":["分母不同"],"records":[{"title":"阳性","numerator":35,"denominator":54,"scope":"限定子集","source":"subset.json","source_sha256":"e"*64},{"title":"对照","numerator":0,"denominator":140,"scope":"原开发全量","source":"all.json","source_sha256":"f"*64}]}
    texts=[];original_p=pdf._p
    def collect(value,*a,**kw):texts.append(str(value));return original_p(value,*a,**kw)
    monkeypatch.setattr(pdf,"_p",collect)
    render(tmp_path,package,decision)
    assert '35 / 54' in texts and '0 / 140' in texts
    assert '限定子集' in texts and '原开发全量' in texts
    assert 'e'*64 in texts and 'f'*64 in texts


def test_title_hierarchy_and_upload_transmission_audit(tmp_path,monkeypatch):
    package,decision=mock_case(tmp_path)
    package['llm_processing']={'upload_scope':'text_only','judge_send_images':False,'stages':{
        'judge':{'transmission_status':'not_sent','planned_figure_ids':[],'included_request_figure_ids':[],
                 'attempted_figure_ids':[],'uploaded_figure_ids':[],'delivery_unknown_figure_ids':[],'omitted_for_request_size':[]},
        'writer':{'transmission_status':'delivery_unknown','planned_figure_ids':['F01','F02'],
                  'included_request_figure_ids':['F01'],'attempted_figure_ids':['F01'],'uploaded_figure_ids':[],
                  'delivery_unknown_figure_ids':['F01'],'omitted_for_request_size':['F02']}}}
    audit=dict(pdf._audit_rows(tmp_path,decision,{},package))
    assert audit['配置上传范围'].startswith('text_only')
    assert audit['裁决图件传输状态']=='未发送请求'
    assert audit['撰写图件传输状态']=='已尝试发送，送达未知'
    assert audit['撰写送达未知图件数量/编号']=='1 / F01'
    assert audit['撰写请求过大而减去的图件数量/编号']=='1 / F02'
    assert audit['上传图件清单 SHA-256']=='无上传图件'
    for state,label in [('response_received','接口已有返回'),('cached_response','复用已保存响应，本次未新发送')]:
        package['llm_processing']['stages']['writer']['transmission_status']=state
        assert dict(pdf._audit_rows(tmp_path,decision,{},package))['撰写图件传输状态']==label
    seen=[];original_p=pdf._p
    def collect(value,styles,kind='body',*a,**kw):
        seen.append((str(value),kind));return original_p(value,styles,kind,*a,**kw)
    monkeypatch.setattr(pdf,'_p',collect)
    render(tmp_path,package,decision)
    assert seen[:2]==[('研判分析报告','title'),('模拟数据验收（非真实检测）','subtitle')]
    assert any(v.startswith('（一）') and k=='h2' for v,k in seen)
    assert pdf._styles()['title'].fontSize==16 and pdf._styles()['subtitle'].fontSize<10.5


def test_video_sampling_summary_does_not_dump_internal_records(tmp_path):
    package,decision=mock_case(tmp_path,'video')
    process=next(e['data'] for e in package['items'] if e['item_id']=='E10')
    process.update(candidate_frame_count=24,selected_frame_count=5,frame_selection={'sampling':{'strategy':'uniform_trimmed','candidate_indices':list(range(24))},'selection':{'rule':'按质量与最小时间间隔选择'},'source_sha256':'f'*64})
    text=build_template_content(package,decision)['process']['fixation']
    assert '候选帧数：24' in text and '入选关键帧数：5' in text
    assert 'candidate_indices' not in text and 'f'*64 not in text and '避开首尾的均匀时间采样' in text
