"""研判报告三版式：事实模板、经校验的撰写内容和可审计的原始图件。"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlparse

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, KeepTogether,
                               LongTable, PageBreak, PageTemplate, Paragraph,
                               Spacer, Table, TableStyle)

from .analysis_templates import (PROCESS_FIELDS, STEP_DESCRIPTIONS, band_label,
    build_template_content, default_figure_placement, index_items, mapping,
    number, paragraphs, rows, safe_text)


PAGE_W, PAGE_H = A4
MARGIN_X, MARGIN_Y = 2.2 * cm, 2.5 * cm
WIDTH = PAGE_W - 2 * MARGIN_X
FONT_REGULAR, FONT_BOLD = "AnalysisSong", "AnalysisHei"
FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
MAX_PDF_BYTES = 30 * 1024 * 1024
_FONT_LOCK = threading.RLock()
_FONT_NOTES = []
_SECTIONS = ("summary", *PROCESS_FIELDS, "synthesis")
_REF = re.compile(r"\[(F\d+)\]")
_REF_RANGE = re.compile(r"\[(F[0-9]+)[ \t]*-[ \t]*(F[0-9]+)\]")
_TITLES = {"fixation": "材料固定与预处理", "provenance": "文件标识与内容凭证",
    "visible_text": "画面文字识别", "detection": "模型分支检测", "source": "素材来源检索与配对",
    "identity": "人物身份比对", "clues": "多路线索组合"}
_BOUNDARIES = (
    "AIGC 可疑度评分不是概率。低档位不能证明材料真实，高档位也不等于已确认伪造。",
    "素材候选不等于已确认来源，人物候选不等于已确认身份；需结合取得过程和独立证据核实。",
    "人脸检测的可靠性受后端、数据来源和人脸质量影响；当前能力仍可能漏检或误报。",
    "大模型可能产生事实或推理错误，其结论和文字均须人工审阅。",
    "本报告只提供辅助研判线索，不能替代人工复核或司法鉴定。",
)


def _fonts():
    with _FONT_LOCK:
        registered = set(pdfmetrics.getRegisteredFontNames())
        for name, preferred, fallback in ((FONT_REGULAR, Path("C:/Windows/Fonts/simsun.ttc"), FONT_DIR / "LXGWWenKai-Regular.ttf"),
                                          (FONT_BOLD, Path("C:/Windows/Fonts/simhei.ttf"), FONT_DIR / "LXGWWenKai-Bold.ttf")):
            if name in registered:
                continue
            try:
                pdfmetrics.registerFont(TTFont(name, str(preferred), subfontIndex=0))
            except Exception as exc:
                # 字体注册异常统一转为静态随附字体；没有中文字体则明确失败。
                if not fallback.is_file():
                    raise RuntimeError("缺少可用中文字体，不能生成研判 PDF") from exc
                pdfmetrics.registerFont(TTFont(name, str(fallback)))
                _FONT_NOTES.append("系统宋体或黑体不可用，已采用随附 LXGWWenKai 字体。")
    return list(dict.fromkeys(_FONT_NOTES))


def _styles():
    common = dict(fontName=FONT_REGULAR, textColor=colors.black, wordWrap="CJK", splitLongWords=True)
    return {
        "body": ParagraphStyle("正文", fontSize=10.5, leading=15.75, spaceAfter=5, rightIndent=9, **common),
        "small": ParagraphStyle("说明", fontSize=9, leading=13, spaceAfter=4, rightIndent=12, **common),
        "table": ParagraphStyle("表格", fontSize=9, leading=12, rightIndent=6, **common),
        "caption": ParagraphStyle("图表题", fontSize=9, leading=13, alignment=TA_CENTER, spaceAfter=4, **common),
        "title": ParagraphStyle("标题", fontName=FONT_BOLD, fontSize=16, leading=23, alignment=TA_CENTER, textColor=colors.black, spaceAfter=15, wordWrap="CJK"),
        "subtitle": ParagraphStyle("软件名称", fontSize=9, leading=13, alignment=TA_CENTER, spaceAfter=12, **common),
        "h1": ParagraphStyle("一级", fontName=FONT_BOLD, fontSize=12, leading=18, spaceBefore=12, spaceAfter=7, textColor=colors.black, keepWithNext=True),
        "h2": ParagraphStyle("二级", fontName=FONT_BOLD, fontSize=10.5, leading=16, spaceBefore=9, spaceAfter=5, textColor=colors.black, keepWithNext=True),
    }


def _display(value, fallback="未记录"):
    if isinstance(value, dict):
        if value.get("kind") == "material_content":
            return _display(value.get("value"), fallback)
        return "；".join(f"{k}：{_display(v)}" for k,v in value.items()
                        if not str(k).endswith("path") and k not in ("api_key", "authorization", "token")) or fallback
    if isinstance(value, (list, tuple)):
        return "；".join(_display(v) for v in value) or fallback
    text = safe_text(value, fallback)
    # 正文禁止完整本机路径；URL 不在此模式内。
    return re.sub(r"(?<![:/\w])/(?:[^\s/]+/)+[^\s，；。]*", "[本机路径已省略]", text)


def _p(value, styles, kind="body"):
    return Paragraph(escape(_display(value)).replace("\n", "<br/>"), styles[kind])


def _three_line_table(headers, values, widths, styles):
    data = [[_p(v, styles, "table") for v in headers]]
    data += [[_p(v, styles, "table") for v in row] for row in values]
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="CENTER", splitInRow=1)
    table.setStyle(TableStyle([
        ("LINEABOVE", (0,0), (-1,0), 1, colors.black),
        ("LINEBELOW", (0,0), (-1,0), .5, colors.black),
        ("LINEBELOW", (0,-1), (-1,-1), 1, colors.black),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    return table


def _table(story, title, headers, values, fractions, styles):
    caption=_p(title, styles, "caption")
    caption.keepWithNext=True
    table=_three_line_table(headers, values or [["无可用记录"] + ["-"]*(len(headers)-1)], [WIDTH*f/sum(fractions) for f in fractions], styles)
    story.extend([caption,table,Spacer(1,7)])


def _inside_file(root, value):
    """允许任务目录内的绝对/相对原图，拒绝链接、越界和上传图替代。"""
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("未提供原图路径")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("原图不在任务目录内")
    probe = candidate
    while probe != root and probe != probe.parent:
        if probe.is_symlink() or (hasattr(probe, "is_junction") and probe.is_junction()):
            raise ValueError("原图经过文件系统链接")
        probe = probe.parent
    return resolved


def _prepare(package, decision, content, root, band_labels):
    template = mapping(content.get("_template")) or build_template_content(package, decision, band_labels)
    cleaned = deepcopy(content)
    warnings, catalog, placements = [], {}, []
    for figure in rows(package.get("figures")):
        fid = figure.get("figure_id", figure.get("id"))
        if not isinstance(fid, str) or not re.fullmatch(r"F\d+", fid) or fid in catalog:
            warnings.append({"code":"invalid_catalog_id", "figure_id":str(fid)})
            continue
        catalog[fid] = figure
    writer = mapping(content.get("_writer"))
    uploaded = set(writer.get("uploaded_figure_ids", []))
    uploaded.update(fid for fid, f in catalog.items() if f.get("uploaded") is True)
    seen = set()
    for placement in rows(content.get("figures")):
        fid, section = placement.get("figure_id"), placement.get("section")
        error = "unknown_figure" if fid not in catalog else "duplicate_figure" if fid in seen else "invalid_section" if section not in _SECTIONS else "not_uploaded" if not placement.get("automatic") and fid not in uploaded else None
        if error:
            warnings.append({"code":error,"figure_id":str(fid)})
            continue
        p = deepcopy(placement)
        default = default_figure_placement(catalog[fid])
        for key, limit in (("caption",30),("reading_guide",120)):
            if not isinstance(p.get(key),str) or not p[key].strip() or len(p[key])>limit:
                p[key] = default[key]
                warnings.append({"code":"invalid_"+key,"figure_id":fid})
        placements.append(p); seen.add(fid)
    for fid, figure in catalog.items():
        if figure.get("required") and fid not in seen:
            placements.append(default_figure_placement(figure)); seen.add(fid)
    # 整节退回事实模板，不能把不存在的图号悄悄替换为看似有效的证据。
    for section in _SECTIONS:
        container = cleaned if section in ("summary","synthesis") else cleaned.setdefault("process",{})
        refs = set(_REF.findall(str(container.get(section,""))))
        if refs - seen:
            fallback = template if section in ("summary","synthesis") else mapping(template.get("process"))
            container[section] = fallback.get(section,"未记录可核验的章节内容。")
            cleaned.setdefault("_section_sources",{})[section] = "template"
            warnings.append({"code":"invalid_reference_section_fallback","section":section,"figure_ids":sorted(refs-seen)})
    for p in placements:
        section = p["section"]
        value = cleaned.get(section) if section in ("summary","synthesis") else mapping(cleaned.get("process")).get(section)
        count = len(paragraphs(value))
        after = p.get("after_paragraph")
        if isinstance(after,bool) or not isinstance(after,int) or after<1:
            after=1
            warnings.append({"code":"invalid_paragraph","figure_id":p["figure_id"]})
        p["after_paragraph"] = min(after,count)
    placements.sort(key=lambda p: (_SECTIONS.index(p["section"]),p["after_paragraph"]))
    missing = []
    for index, p in enumerate(placements,1):
        f = catalog[p["figure_id"]]
        p["number"], p["figure"] = index, f
        try:
            original = _inside_file(root,f.get("original_path"))
            digest = sha256(original.read_bytes()).hexdigest()
            if f.get("original_sha256") and digest.lower()!=str(f["original_sha256"]).lower():
                raise ValueError("原图 SHA-256 校验不一致")
            with PILImage.open(original) as image:
                image.verify()
            p["path"], p["original_sha256"] = original,digest
        except (OSError,ValueError,SyntaxError) as exc:
            p["missing"] = str(exc) if isinstance(exc,ValueError) else "原图不存在或无法解码"
            p["original_sha256"] = f.get("original_sha256","未记录")
            missing.append({"figure_id":p["figure_id"],"reason":p["missing"],"required":bool(f.get("required"))})
    numbers={p["figure_id"]:p["number"] for p in placements}
    return cleaned,placements,numbers,warnings,missing


class _EvidenceImage(Flowable):
    def __init__(self, placement, width, max_height, compress=False):
        super().__init__()
        self.placement=placement
        with PILImage.open(placement["path"]) as img:
            self.original_size=img.size
            scale=min(width/img.width,max_height/img.height)
            self.width,self.height=img.width*scale,img.height*scale
            if compress:
                buffer=BytesIO()
                if img.mode in ("RGBA","LA") or "transparency" in img.info:
                    rgba=img.convert("RGBA"); rgb=PILImage.new("RGB",img.size,"white");rgb.paste(rgba,mask=rgba.getchannel("A"))
                else: rgb=img.convert("RGB")
                rgb.save(buffer,format="JPEG",quality=90,subsampling=0)
                self.reader=ImageReader(BytesIO(buffer.getvalue()))
            else:
                self.reader=ImageReader(str(placement["path"]))
        self.hAlign="CENTER"

    def draw(self):
        c=self.canv
        c.drawImage(self.reader,0,0,width=self.width,height=self.height,mask="auto")
        x,y=c.absolutePosition(0,0)
        c.image_regions.append({"page":c.getPageNumber(),"figure_id":self.placement["figure_id"],
                               "x0":round(x,2),"y0":round(y,2),"x1":round(x+self.width,2),"y1":round(y+self.height,2),
                               "original_width":self.original_size[0],"original_height":self.original_size[1]})
        f=self.placement["figure"]
        bbox=f.get("bbox")
        if isinstance(bbox,(list,tuple)) and len(bbox)==4 and all(isinstance(v,(int,float)) for v in bbox):
            iw,ih=self.original_size
            x1,y1,x2,y2=bbox
            if 0<=x1<x2<=iw and 0<=y1<y2<=ih:
                c.saveState();c.setStrokeColor(colors.black);c.setFillColor(colors.black);c.setLineWidth(.65)
                c.rect(x1/iw*self.width,(ih-y2)/ih*self.height,(x2-x1)/iw*self.width,(y2-y1)/ih*self.height,fill=0,stroke=1)
                c.setFont(FONT_REGULAR,8)
                face_label=f.get("face_id",f.get("face_index"))
                c.drawString(x1/iw*self.width,min(self.height-9,(ih-y1)/ih*self.height+2),"人脸 "+str(face_label if face_label is not None else "编号未记录"))
                c.restoreState()


class _NumberedCanvas(Canvas):
    def __init__(self,*args,task_id="",metrics=None,**kwargs):
        super().__init__(*args,**kwargs)
        self._page_states=[];self.task_id=task_id;self.metrics=metrics;self.image_regions=[]

    def showPage(self):
        self._page_states.append(dict(self.__dict__));self._startPage()

    def save(self):
        count=len(self._page_states)
        if self.metrics is not None:
            self.metrics["page_count"]=count
            self.metrics["image_regions"]=list(self.image_regions)
        for state in self._page_states:
            self.__dict__.update(state)
            self.saveState();self.setFillColor(colors.black);self.setStrokeColor(colors.black)
            self.setFont(FONT_REGULAR,9)
            self.drawString(MARGIN_X,PAGE_H-MARGIN_Y+20,"研判分析报告")
            # 长任务号按同一行缩小，正文表格保留完整值。
            header=_display(self.task_id)
            width=pdfmetrics.stringWidth(header,FONT_REGULAR,9)
            self.setFont(FONT_REGULAR,min(9,9*(WIDTH-100)/max(width,1)))
            self.drawRightString(PAGE_W-MARGIN_X,PAGE_H-MARGIN_Y+20,header)
            self.setLineWidth(.5);self.line(MARGIN_X,PAGE_H-MARGIN_Y+12,PAGE_W-MARGIN_X,PAGE_H-MARGIN_Y+12)
            self.setFont(FONT_REGULAR,9);self.drawCentredString(PAGE_W/2,MARGIN_Y-25,f"第 {self.getPageNumber()} 页 共 {count} 页")
            self.restoreState();super().showPage()
        super().save()


def _replace_refs(text,numbers):
    def replace_range(match):
        first, last = match.groups()
        # 先限制数字长度和跨度，不能因异常引用展开巨幅范围。
        if len(first) > 7 or len(last) > 7:
            return "未归档图件"
        start, end = int(first[1:]), int(last[1:])
        if (start < 1 or start >= end or first != f"F{start:02d}" or last != f"F{end:02d}"
                or end - start + 1 > min(len(numbers), 256)):
            return "未归档图件"
        ids = [f"F{i:02d}" for i in range(start, end + 1)]
        if any(fid not in numbers for fid in ids):
            return "未归档图件"
        # 图件编号不等于报告图号；必须逐项使用实际编排后的映射。
        return "、".join(f"图{numbers[fid]}" for fid in ids)

    text = _REF_RANGE.sub(replace_range, str(text))
    return _REF.sub(lambda m:f"图{numbers[m.group(1)]}" if m.group(1) in numbers else "未归档图件",text)


def _figure_rows(story, group, styles, compress):
    # 对齐/差异三联图同一行；其余多个图件两图一行。
    def triple(start):
        part=group[start:start+3]
        if len(part)!=3 or any(p["section"]!="source" for p in part):return False
        types={p["figure"].get("type") for p in part}
        return (types=={"pair_reference","pair_aligned","pair_difference"} and len({p["figure"].get("candidate_rank") for p in part})==1) or types=={"source_candidate","aligned_pair","difference_heatmap"}
    start=0
    while start<len(group):
        columns=3 if triple(start) else 1 if triple(start+1) else min(2,len(group)-start)
        batch=group[start:start+columns];start+=columns
        cell_width=(WIDTH-(columns-1)*10)/columns
        cells=[]
        for p in batch:
            cell=[]
            if p.get("missing"):
                cell.append(_p("图件不可用："+p["missing"],styles,"small"))
            else:
                cell.append(_EvidenceImage(p,cell_width-4,240 if columns==1 else 155,compress))
            cell.append(Spacer(1,4))
            note="（系统自动补入）" if p.get("automatic") else ""
            cell.append(_p(f"图{p['number']}　{p['caption']}{note}",styles,"caption"))
            cell.append(_p(p["reading_guide"],styles,"small"))
            cells.append(cell)
        if columns==1:
            story.append(KeepTogether(cells[0]+[Spacer(1,6)]))
        else:
            # 无框线排版容器，不是数据表格。
            expanded=[];widths=[]
            for n,cell in enumerate(cells):
                if n:expanded.append("");widths.append(10)
                expanded.append(cell);widths.append(cell_width)
            table=Table([expanded],colWidths=widths,hAlign="LEFT")
            table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0)]))
            story.extend([table,Spacer(1,6)])


def _section(story, section, content, placements, numbers, styles, compress):
    value=content.get(section) if section in ("summary","synthesis") else mapping(content.get("process")).get(section)
    placed=defaultdict(list)
    for p in placements:
        if p["section"]==section:placed[p["after_paragraph"]].append(p)
    for i,text in enumerate(paragraphs(value),1):
        story.append(_p(_replace_refs(text,numbers),styles))
        if placed[i]:_figure_rows(story,placed[i],styles,compress)


def _model_name(value):
    m=mapping(value)
    return m.get("name") or m.get("model_name") or m.get("model_id") or m.get("id") or value or "未记录"


def _thresholds(data, *, full=False):
    t=mapping(data.get("thresholds"))
    if full:return t or "未记录"
    return "\n".join(f"{label}：{t[k]}" for k,label in (("low_max","95%线"),("medium_max","99%线")) if k in t) or "未记录"


def _statistics(data):
    stats=mapping(data.get("statistics"))
    records=rows(stats.get("rows"))
    if not records:return "无对应统计"
    return "\n".join(f"{r.get('scenario','已记录场景')}：真实 {r.get('real_percent','未记录')}%，伪造 {r.get('fake_percent','未记录')}%" for r in records)


def _raw(value):
    return str(value) if isinstance(value,(int,float)) and not isinstance(value,bool) else "未记录"


def _state(value):
    return {"high_confidence":"高可信候选","insufficient_evidence":"证据不足，建议复核","no_reliable_candidate":"未发现可靠候选","unavailable":"无法完成比对","completed":"已完成","success":"已完成","failed":"失败","matched":"已对应","review":"建议复核","low":"低区段","medium":"中区段","high":"高区段"}.get(value,value) if isinstance(value,str) else value


def _pair_metrics(value):
    data=mapping(value);background=mapping(data.get("background"));parts=[]
    for title,number_value in (("背景 SSIM",background.get("ssim_median",data.get("background_ssim"))),
                               ("背景变化比例",background.get("changed_fraction")),
                               ("人脸差异集中度",data.get("face_change_concentration"))):
        if isinstance(number_value,(int,float)):parts.append(f"{title}：{number(number_value)}")
    return "；".join(parts) or "无可用配对指标"


def _pair_faces(value):
    if not isinstance(value,list):return _display(value)
    parts=[]
    for face in rows(value):
        parts.append("查询脸 "+_display(face.get("query_face_id"))+" / 参考脸 "+_display(face.get("reference_face_id"))+
                     "；相似度 "+number(face.get("cosine_similarity"))+"；"+_display(_state(face.get("match_status"))))
    return "\n".join(parts) or "未记录人脸对应"


def _identity_candidate_rows(identity):
    """仅为展示附加字段；开放集结果只属于每张查询脸的首候选。"""
    known_states = {"high_confidence", "insufficient_evidence", "no_reliable_candidate", "unavailable"}

    def display(rank, state):
        # 排序须为明确的正整数；不按列表位置、分数或布尔值猜测。
        if type(rank) is not int or rank < 1:
            return "未记录"
        if rank > 1:
            return "未单独判定"
        return _state(state) if isinstance(state, str) and state in known_states else "未记录"

    candidates = []
    for face in rows(identity.get("faces")):
        state = mapping(face.get("open_set")).get("state")
        candidates.extend({**r, "face_index": face.get("face_index"),
                           "open_set_display": display(r.get("rank"), state)}
                          for r in rows(face.get("candidates")))
    if not candidates:
        state = identity.get("open_set_state") or mapping(identity.get("open_set")).get("state")
        candidates = [{**r, "open_set_display": display(r.get("rank"), state)}
                      for r in rows(identity.get("candidates"))]
    return candidates


def _module_tables(story,key,e,styles):
    if key=="fixation":
        frames=rows(e.get("E10",{}).get("key_frames"))
        if frames:
            _table(story,"表 3　关键帧记录",["序号","时间（秒）","选择原因"],[[r.get("frame_index",i),r.get("timestamp_seconds",r.get("timestamp")),r.get("reason",r.get("selection_reason","未记录"))] for i,r in enumerate(frames,1)],[1,2,6],styles)
    elif key=="detection":
        values=[]
        for eid,name,role in (("E03","通用","主分支"),("E04","人脸","主分支"),("E05","整图辅助","辅助分支")):
            d=e.get(eid,{})
            values.append([name,role,_model_name(d.get("model")),_raw(d.get("score")),_thresholds(d),d.get("interval","无可用区段"),_statistics(d)])
        _table(story,"表 4　检测分支与对应统计",["分支","角色","模型","原始分数","分界线","区段","对应区段真实/伪造比例"],values,[.7,.8,1.2,1,1.25,1.25,1.8],styles)
        story.append(_p("表内百分比沿用证据包中的场景、分支和精度，表示历史样本落入对应区段的比例，不是本案概率。",styles,"small"))
        face=e.get("E04",{})
        per_face=rows(face.get("per_face"))
        if per_face:
            _table(story,"表 5　逐个人脸记录",["人脸","帧/时间","分数","本次分档用途","尺寸/位置"],[[f.get("face_id",f.get("face_index",i)),f.get("frame_index",f.get("timestamp_seconds","-")),number(f.get("score",f.get("logit"))),"用于分档" if f.get("used_for_band") is True else "未用于分档" if f.get("used_for_band") is False else _state(f.get("band",f.get("interval","未记录"))),f.get("bbox",f.get("size","未记录"))] for i,f in enumerate(per_face,1)],[.7,1.2,1.2,1.5,3],styles)
        fs=mapping(face.get("frame_statistics"))
        if fs:
            _table(story,"表 6　人脸分支帧级统计",["记录帧数","最小值","中位数","最大值","超过低分界线比例","超过高分界线比例"],[[fs.get("n"),number(fs.get("min")),number(fs.get("median")),number(fs.get("max")),number(fs.get("above_low_max_ratio")),number(fs.get("above_medium_max_ratio"))]],[1,1,1,1,1.6,1.6],styles)
    elif key=="source":
        sources=rows(e.get("E06",{}).get("candidates"))
        _table(story,"表 7　素材候选",["排序","候选名称","全局分数","局部分数","融合分数","开放集状态"],[[r.get("rank"),r.get("candidate_name"),number(r.get("global_score")),number(r.get("local_score")),number(r.get("final_score")),_state(r.get("open_set_state"))] for r in sources],[.6,2,1,1,1,1.3],styles)
        pairs=rows(e.get("E08",{}).get("candidates"))
        _table(story,"表 8　候选配对核查",["候选","对齐状态","差异/背景指标","人脸对应","说明"],[[r.get("candidate_name"),_state(r.get("alignment_status",mapping(r.get("alignment")).get("status"))),_pair_metrics(r.get("metrics")),_pair_faces(r.get("faces")),r.get("reason",r.get("code"))] for r in pairs],[1.4,1,1.6,1.6,2],styles)
    elif key=="identity":
        candidates=_identity_candidate_rows(mapping(e.get("E07")))
        _table(story,"表 9　人物身份候选",["人脸/排序","候选","相似度","参考最高值","首候选开放集结果"],[[f"{r.get('face_index','-')}/{r.get('rank','-')}",r.get("identity_name",r.get("identity_id")),number(r.get("score",r.get("mean_similarity"))),number(r.get("best_reference_score")),r["open_set_display"]] for r in candidates],[1,2,1.2,1.2,1.8],styles)
        story.append(_p("开放集结果仅针对每张查询脸的首候选；其余候选仅作排名展示，候选不等于身份确认。",styles,"small"))


def _audit_rows(root, decision, writer, package):
    processing=mapping(package.get("llm_processing"));stages=mapping(processing.get("stages"))
    scope=processing.get("upload_scope")
    scope_labels={"full":"full：结构化证据及按配置选择的图件", "text_only":"text_only：去标识的结构化证据，不上传图片"}
    result=[("配置上传范围",scope_labels.get(scope,"未记录；不能根据图件数量反推上传范围")),
            ("研判带图配置",{True:"启用",False:"关闭"}.get(processing.get("judge_send_images"),"未记录"))]
    for title,block,prefix in (("裁决",decision,"llm_judge"),("撰写",writer,"llm_writer")):
        transmission=mapping(stages.get("judge" if prefix=="llm_judge" else "writer")) or block
        transmission_note=transmission.get("transmission_note") or {"not_sent":"未发送请求", "delivery_unknown":"已尝试发送，送达未知", "response_received":"接口已有返回", "cached_response":"复用已保存响应，本次未新发送"}.get(transmission.get("transmission_status"),"未记录")
        unknown_ids=transmission.get("delivery_unknown_figure_ids")
        if transmission.get("transmission_status")=="not_sent" and isinstance(unknown_ids,list) and unknown_ids:
            # 最近调用未发出，不能覆盖此前物理尝试的送达未知记录。
            transmission_note="此前图件已尝试发送、送达未知；最近一次调用未发送"
            if transmission.get("transmission_note"):
                transmission_note+="；原记录说明："+str(transmission["transmission_note"])
        host=str(block.get("host", ""))
        host=urlparse(host if "://" in host else "https://"+host).hostname if host else "未记录"
        prompt=mapping(block.get("prompt"))
        result += [(title+"实际模型",block.get("actual_model",block.get("model","未参与"))),
                   (title+"请求主机",host),(title+"请求时间",block.get("time")),
                   (title+"提示词版本",block.get("prompt_version",prompt.get("version"))),
                   (title+"提示词 SHA-256",block.get("prompt_sha256",prompt.get("sha256"))),
                   (title+"请求 SHA-256",block.get("request_sha256")),
                   (title+"响应 SHA-256",block.get("response_sha256")),
                   (title+"校验/退回原因",{"status":block.get("status",block.get("source")),"invalid_sections":block.get("invalid_sections",[]),"reason":block.get("reason")}),
                    (title+"图件传输状态",transmission_note)]
        for field,label in (("planned_figure_ids","计划图件"),("included_request_figure_ids","最终请求所含图件"),
                            ("attempted_figure_ids","已尝试发送图件"),("uploaded_figure_ids","已有返回或缓存支持的图件"),
                            ("delivery_unknown_figure_ids","送达未知图件"),("omitted_for_request_size","请求过大而减去的图件")):
            ids=transmission.get(field)
            result.append((title+label+"数量/编号",f"{len(ids)} / "+("、".join(map(str,ids)) or "无") if isinstance(ids,list) else "未记录"))
        for suffix in ("request","response","check"):
            filename=f"{prefix}_{suffix}.json"
            try:
                path=_inside_file(root,filename)
                result.append((title+{"request":"请求文件","response":"响应文件","check":"校验文件"}[suffix]+" SHA-256",sha256(path.read_bytes()).hexdigest()))
            except (OSError,ValueError):pass
    uploaded=set((mapping(stages.get("judge")) or decision).get("uploaded_figure_ids",[])) | set((mapping(stages.get("writer")) or writer).get("uploaded_figure_ids",[]))
    upload_manifest=[{"figure_id":f["figure_id"],"sha256":f.get("upload_sha256",f.get("sha256"))} for f in rows(package.get("figures")) if f.get("figure_id") in uploaded]
    upload_manifest.sort(key=lambda f:f["figure_id"])
    result.append(("上传图件清单 SHA-256",sha256(json.dumps(upload_manifest,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest() if uploaded else "无上传图件"))
    result.append(("上传图件清单摘要方法","按图件编号排序，以 figure_id 和上传文件 sha256 的 UTF-8 紧凑 JSON 计算。"))
    result.append(("图件留痕边界","清单仅含接口已有返回或已保存响应支持的图件；送达未知单独列示。视频仅按配置发送关键帧，不发送视频本体。"))
    for f in rows(package.get("figures")):
        if f.get("figure_id") in uploaded:
            result.append(("上传图件 "+f["figure_id"]+" SHA-256",f.get("upload_sha256",f.get("sha256"))))
    return result


def _build(package,decision,content,root,product_name,band_labels,review,placements,numbers,font_notes,compress,metrics):
    styles=_styles();story=[];e=index_items(package);material=e.get("E09",{});process=e.get("E10",{});writer=mapping(content.get("_writer"))
    story.extend([_p("研判分析报告",styles,"title"),_p(product_name,styles,"subtitle")])
    story.append(_p("一、基本情况",styles,"h1"))
    _table(story,"表 1　材料与任务信息",["项目","记录"],[
        ["任务编号",material.get("task_id",package.get("task_id"))],["文件名",material.get("file_name")],
        ["材料类型",{"image":"图片","video":"视频"}.get(material.get("type"),material.get("type"))],
        ["大小 / 尺寸 / 时长",f"字节：{_display(material.get('size_bytes',material.get('file_size')))}；尺寸：{_display(material.get('width'))} × {_display(material.get('height'))}；时长（秒）：{_display(material.get('duration_seconds'),'不适用' if material.get('type')=='image' else '未记录')}"],
        ["输入 SHA-256",material.get("input_sha256")],["分析时间",process.get("analysis_time")],["所选模块",process.get("selected_modules")]], [1.2,5],styles)
    story.append(_p("二、研判结论",styles,"h1"))
    llm=decision.get("source")=="llm"
    mode=("大模型参与："+_display(decision.get("actual_model",decision.get("model")))) if llm else "大模型未参与："+_display(decision.get("reason"),"本次采用规则结果")
    _table(story,"表 2　最终结论",["项目","记录"],[
        ["最终档位",band_label(decision.get("final_band"),band_labels)],
        ["AIGC 可疑度评分",("-（未评分）" if decision.get("score") is None else _raw(decision["score"])+"（不是概率）")],
        ["规则参考档位",band_label(decision.get("rule_band",package.get("rule_band")),band_labels)],
        ["研判方式",mode]], [1.2,5],styles)
    _section(story,"summary",content,placements,numbers,styles,compress)
    story.append(_p("三、分析过程",styles,"h1"))
    for i,key in enumerate(PROCESS_FIELDS,1):
        story.append(_p(f"（{'一二三四五六七'[i-1]}）{_TITLES[key]}",styles,"h2"))
        story.append(_p("本步做了什么："+STEP_DESCRIPTIONS[key],styles))
        _section(story,key,content,placements,numbers,styles,compress)
        _module_tables(story,key,e,styles)
    story.append(_p("四、综合研判",styles,"h1"));_section(story,"synthesis",content,placements,numbers,styles,compress)
    story.append(_p("五、评分依据",styles,"h1"))
    assessment=rows(mapping(decision.get("judgment")).get("evidence_assessment"))
    _table(story,"表 10　逐项证据权重",["证据项","观察","方向","权重","理由"],[[r.get("item_id"),r.get("observation"),{"supports_aigc":"支持生成合成","against_aigc":"反向线索","neutral":"中性"}.get(r.get("direction"),r.get("direction")),{"high":"高","medium":"中","low":"低"}.get(r.get("weight"),r.get("weight")),r.get("reason")] for r in assessment],[.8,2,1, .7,2.3],styles)
    if not assessment:story.append(_p("本次没有可用的逐项大模型评分依据，不补造证据权重。",styles))
    story.append(_p("六、不确定性与使用边界",styles,"h1"))
    uncertainty=content.get("uncertainties",[])
    for text in (uncertainty if isinstance(uncertainty,list) else [uncertainty]):story.append(_p(text,styles))
    for text in _BOUNDARIES:story.append(_p(text,styles))
    next_step_story=[_p("七、建议核查事项",styles,"h1")]
    steps=content.get("next_steps",[])
    for i,text in enumerate(steps if isinstance(steps,list) else [steps],1):next_step_story.append(_p(f"{i}. {text}",styles))
    story.append(KeepTogether(next_step_story))
    story.append(_p("附录一：技术信息",styles,"h1"))
    tech=[["系统版本",package.get("system_version",mapping(package.get("configuration")).get("system_version"))],
          ["人脸后端 / 汇总方式",package.get("configuration")], ["素材库 / 身份库",package.get("libraries",package.get("library_versions"))],
          ["证据包清单 SHA-256",package.get("manifest_sha256") or mapping(package.get("source_manifest")).get("sha256")]]
    for eid,name in (("E03","通用"),("E04","人脸"),("E05","整图辅助")):
        d=e.get(eid,{})
        tech += [[name+"模型及版本",d.get("model")],[name+"原始分数",_raw(d.get("score"))],[name+"分界线与校准信息",_thresholds(d,full=True)],[name+"可靠性说明",d.get("known_reliability")]]
    if material.get("type")=="video":
        selection=mapping(process.get("frame_selection"));sampling=mapping(selection.get("sampling"))
        tech += [["视频帧率",material.get("fps")],["首要帧 SHA-256",material.get("primary_frame_sha256")],
                 ["候选帧数 / 入选关键帧数",f"{_display(process.get('candidate_frame_count'))} / {_display(process.get('selected_frame_count'))}"],
                 ["候选采样方式",sampling.get("strategy")],["关键帧选择规则",mapping(selection.get("selection")).get("rule")],
                 ["选帧记录文件",selection.get("source_file")],["选帧记录 SHA-256",selection.get("source_sha256")]]
    tech += [["字体记录", "；".join(font_notes) if font_notes else "正文宋体；标题黑体"],
             ["PDF 图像处理", "原尺寸 JPEG 质量 90 压缩，证据原文件不变" if compress else "使用任务目录内的原分辨率图件，未使用上传缩图"],
             ["章节来源",content.get("_section_sources")]]
    _table(story,"表 A1　技术记录",["项目","记录"],tech,[1.4,4.8],styles)
    face_stats=mapping(e.get("E04",{}).get("reliability_statistics"))
    if face_stats.get("status")=="available":
        _table(story,"表 A1-2　对应人脸后端的历史验证",["项目","记录"],[
            ["适用后端与汇总",f"{face_stats.get('backend','未记录')} / {face_stats.get('band_aggregation','未记录')}"],
            ["AUC（原精度）",face_stats.get("face_auc")],["样本总数 / 可用人脸",{"总数":face_stats.get("n_total"),"可用人脸":face_stats.get("face_available")}],
            ["范围与限制",face_stats.get("basis")],["来源",face_stats.get("source")],["来源 SHA-256",face_stats.get("source_sha256")]], [1.4,4.8],styles)
    pair_stats=mapping(e.get("E08",{}).get("reliability_statistics"))
    if pair_stats.get("status")=="available":
        story.append(_p(pair_stats.get("basis"),styles,"small"))
        for note in pair_stats.get("notes",[]):story.append(_p(note,styles,"small"))
        references={}
        result_rows=[]
        for record in rows(pair_stats.get("records")):
            source=(record.get("source"),record.get("source_sha256"))
            if source not in references:references[source]=f"S{len(references)+1}"
            result_rows.append([record.get("title"),f"{record.get('numerator','未记录')} / {record.get('denominator','未记录')}",record.get("scope"),references[source]])
        _table(story,"表 A1-3　配对规则历史开发统计（分母分别列示）",["指标","计数 / 分母","适用范围","来源编号"],result_rows,[1.6,1.2,3, .8],styles)
        _table(story,"表 A1-4　历史开发统计来源",["编号","来源文件","SHA-256"],[[code,source,digest] for (source,digest),code in references.items()],[.5,2.5,3.6],styles)
    story.append(_p("附录二：研判记录",styles,"h1"))
    _table(story,"表 A2　调用和校验记录",["项目","记录"],_audit_rows(root,decision,writer,package),[1.6,4.6],styles)
    appendix=[_p("附录三：图件目录",styles,"h1")]
    _table(appendix,"表 A3　报告图件",["图号/编号","图题","来源模块","编排来源","原图 SHA-256"],[[f"图{p['number']} / {p['figure_id']}",p["caption"],p["figure"].get("module"),"系统自动补入" if p.get("automatic") else "撰写模型",p["original_sha256"]] for p in placements],[1,1.5,1,1,2.5],styles)
    if len(placements)<=8:story.append(KeepTogether(appendix))
    else:story.extend(appendix)
    if review is not None:
        story.append(PageBreak());story.append(_p("八、人工复核意见",styles,"h1"))
        records=rows(review) if isinstance(review,(tuple,list)) else [mapping(review)]
        names={"reviewer":"复核人（手填）","conclusion":"复核结论","opinion":"复核意见","reviewed_at":"复核时间","report_sha256":"所复核报告 SHA-256","manual_judgment":"人工判定","basis_candidate_rank":"依据候选排序","target_face_index":"目标人脸编号","basis_candidate_name":"依据候选名称","basis_candidate_sha256":"依据候选 SHA-256"}
        for i,r in enumerate(records,1):_table(story,f"表 R{i}　人工复核记录",["项目","记录"],[[title,r.get(k)] for k,title in names.items() if r.get(k) is not None],[1.5,4.7],styles)
        story.append(_p("复核人姓名为手填，没有账号认证；本记录不构成可信时间戳或不可篡改存证。人工意见作为追加内容，不覆盖机器档位、评分及原始研判。",styles))
    buffer=BytesIO();doc=BaseDocTemplate(buffer,pagesize=A4,leftMargin=MARGIN_X,rightMargin=MARGIN_X,topMargin=MARGIN_Y,bottomMargin=MARGIN_Y,title="研判分析报告",author=_display(product_name))
    doc.addPageTemplates(PageTemplate(id="report",frames=[Frame(MARGIN_X,MARGIN_Y,WIDTH,PAGE_H-2*MARGIN_Y,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)]))
    doc.build(story,canvasmaker=lambda *a,**kw:_NumberedCanvas(*a,task_id=material.get("task_id",package.get("task_id","未记录")),metrics=metrics,**kw))
    return buffer.getvalue()


def render_analysis_pdf(package, final_decision, content, task_dir, *, product_name, band_labels=None, review=None) -> bytes:
    """返回 PDF 字节并写独立生成记录；不改证据包、原图或原报告。"""
    started=time.perf_counter();root=Path(task_dir).resolve(strict=True)
    if not root.is_dir():raise ValueError("任务目录不存在")
    package,decision=mapping(package),mapping(final_decision)
    content=mapping(content) or build_template_content(package,decision,band_labels)
    font_notes=_fonts()
    cleaned,placements,numbers,warnings,missing=_prepare(package,decision,content,root,band_labels)
    metrics={}
    data=_build(package,decision,cleaned,root,product_name,band_labels,review,placements,numbers,font_notes,False,metrics)
    initial_bytes=len(data);compressed=False
    if initial_bytes>MAX_PDF_BYTES and any(p.get("path") for p in placements):
        candidate_metrics={}
        candidate=_build(package,decision,cleaned,root,product_name,band_labels,review,placements,numbers,font_notes,True,candidate_metrics)
        if len(candidate)<len(data):data,metrics,compressed=candidate,candidate_metrics,True
    report={"schema_version":"1.0","report_type":"analysis_review" if review is not None else "analysis",
            "generation_seconds":round(time.perf_counter()-started,3),"file_size_bytes":len(data),
            "pdf_sha256":sha256(data).hexdigest(),"initial_file_size_bytes":initial_bytes,
            "compression":{"triggered":initial_bytes>MAX_PDF_BYTES,"applied":compressed,"jpeg_quality":90 if compressed else None,"resolution_changed":False,"over_30mb":len(data)>MAX_PDF_BYTES},
            "font_notes":font_notes,"figure_numbers":numbers,"placement_warnings":warnings,
            "missing_figures":missing,"section_sources":cleaned.get("_section_sources",{}),**metrics}
    name="analysis_pdf_review_render.json" if review is not None else "analysis_pdf_render.json"
    output=root/name
    if output.exists() and (output.is_symlink() or not output.resolve().is_relative_to(root)):
        raise ValueError("生成记录路径不安全")
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return data
