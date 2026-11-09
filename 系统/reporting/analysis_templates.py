"""从证据包生成研判分析模板，按章节合并已经校验的撰写结果。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
import re
from typing import Any

from .text_safety import redact_local_paths


PROCESS_FIELDS = ("fixation", "provenance", "visible_text", "detection", "source", "identity", "clues")
DEFAULT_BAND_LABELS = {"none": "未见明显 AIGC 痕迹", "review": "建议人工复核", "obvious": "明显 AIGC 痕迹", "undetermined": "无法完成判断"}
STEP_DESCRIPTIONS = {
    "fixation": "本步记录输入材料及其完整性摘要，并整理图像预处理或视频关键帧选择情况。",
    "provenance": "本步读取文件中的生成合成声明、工具痕迹和内容凭证结构；声明本身仍需核实。",
    "visible_text": "本步识别画面中的生成合成提示文字，记录命中内容和位置。",
    "detection": "本步分别读取通用、人脸和整图辅助分支的分数、分界线及对应统计。分数来自不同模型，不直接相加。",
    "source": "本步检索候选素材并核查局部几何对齐和配对差异；候选与像素变化用于人工核查来源。",
    "identity": "本步将可用人脸与当前身份库比对，并根据开放集状态说明候选的适用范围。",
    "clues": "本步依据固定条件组合多路线索，说明证据之间的关系和仍待核实的解释。",
}
_STATUS = {"success": "已完成", "partial_success": "部分完成", "no_face": "未检出可用人脸", "skipped": "未运行", "error": "未完成", "failed": "未完成", "unavailable": "不可用", "unrecorded": "未记录"}
_OPEN_STATES = {"high_confidence":"高可信候选", "insufficient_evidence":"证据不足，建议人工复核", "no_reliable_candidate":"当前库中未发现可靠候选", "unavailable":"本次无法完成可比对检索", "review":"建议人工复核"}


def mapping(value: Any) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def rows(value: Any) -> list[dict]:
    return [dict(v) for v in value if isinstance(v, Mapping)] if isinstance(value, (list, tuple)) else []


def safe_text(value: Any, fallback: str = "未记录") -> str:
    """材料字符串只按文字显示；本机路径不进入报告。"""
    if value is None or value == "":
        return fallback
    if isinstance(value, Mapping) and value.get("kind") == "material_content":
        return safe_text(value.get("value"), value.get("notice", fallback))
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    result = redact_local_paths(str(value))
    return "".join(c for c in result if c in "\n\t" or ord(c) >= 32)


def number(value: Any, digits: int = 2) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "未记录"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") if digits else str(round(value))


def raw_score_number(value: Any) -> str:
    """原始模型分数保留八位有效数字，避免小分数在正文中变为零。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "未记录"
    return format(value, ".8g")


def index_items(package: Mapping[str, Any]) -> dict[str, dict]:
    return {safe_text(v.get("item_id"), ""): mapping(v.get("data")) for v in rows(package.get("items"))}


def labels(band_labels=None) -> dict:
    return {**DEFAULT_BAND_LABELS, **{k: str(v) for k,v in mapping(band_labels).items() if v}}


def band_label(band, band_labels=None) -> str:
    return labels(band_labels).get(band, DEFAULT_BAND_LABELS["undetermined"])


def paragraphs(value: Any) -> list[str]:
    return [p.strip() for p in str(value or "").splitlines() if p.strip()] or ["未记录该节分析内容。"]


def _status(data: dict, name: str) -> str:
    state = data.get("status")
    if state == "success":
        return f"{name}已完成。"
    reason = data.get("error_message") or data.get("reason") or data.get("note")
    return f"本次{name}{_STATUS.get(state, '未完成或未记录')}" + (f"，原因为{safe_text(reason)}。" if reason else "，没有可用结果支持进一步判断。")


def figure_section(figure: Mapping[str, Any]) -> str:
    if figure.get("section") in ("summary", *PROCESS_FIELDS, "synthesis"):
        return str(figure["section"])
    kind = str(figure.get("type", "")).lower()
    for terms, section in [(("identity", "身份", "人物"), "identity"), (("pair", "source", "candidate", "aligned", "diff", "mask", "sift", "素材", "配对", "差异"), "source"), (("visible", "ocr", "文字"), "visible_text"), (("face", "score", "curve", "detection", "人脸", "分数", "曲线"), "detection")]:
        if any(term in kind for term in terms):
            return section
    return "fixation"


def default_figure_placement(figure: Mapping[str, Any]) -> dict:
    titles = {"original":"输入原图", "key_frame":"视频关键帧", "face_crop":"人脸裁剪", "detection_visual":"检测可视化", "source_candidate":"素材候选", "pair_reference":"候选底图", "pair_aligned":"对齐查询图", "pair_difference":"像素差异图", "pair_change_mask":"变化掩码", "sift_matches":"局部匹配", "identity_query":"身份查询脸", "identity_candidate":"身份候选", "identity_comparison":"人物身份对照", "visible_text":"文字命中区域", "branch_scores":"分支分数与分界线", "video_scores":"关键帧分数曲线", "aligned_pair":"对齐查询图", "difference_heatmap":"像素差异图"}
    title = figure.get("caption") or figure.get("title") or titles.get(figure.get("type"), "证据图件")
    return {"figure_id": figure.get("figure_id", figure.get("id")), "section": figure_section(figure),
            "after_paragraph": 1, "caption": safe_text(title)[:30],
            "reading_guide": safe_text(figure.get("description"), "结合本节记录核查图中材料。")[:120], "automatic": True}


def build_template_content(package, final_decision, band_labels=None) -> dict:
    """只复述已有事实；没有大模型长文时不拼凑不存在的推理或成绩。"""
    package, decision = mapping(package), mapping(final_decision)
    e = index_items(package)
    material, process = e.get("E09", {}), e.get("E10", {})
    label = band_label(decision.get("final_band"), band_labels)
    score = "—" if decision.get("score") is None else number(decision["score"], 0)
    reason = safe_text(decision.get("reason"), "未提供额外研判理由")
    rule = band_label(decision.get("rule_band", package.get("rule_band")), band_labels)
    summary = f"本次最终档位为“{label}”，AIGC 可疑度评分为{score}。" + ("评分不是概率。" if decision.get("score") is not None else "未提供评分。")
    summary += "各模块记录与使用边界详见下文，结论须经人工复核。"
    fixed = "已记录完整性摘要" if material.get("fixed") else "未记录完整性固定结果"
    if material.get("hash_mismatch"):
        fixed += "；现有摘要核对存在不一致"
    fixation = f"{fixed}（证据编号 E09）。"
    if material.get("type") == "video":
        frames = rows(process.get("key_frames"))
        times = [number(f.get("timestamp_seconds")) for f in frames]
        fixation += "关键帧时间点（秒）：" + ("、".join(times) if times else "未记录") + "。"
        selection = mapping(process.get("frame_selection"))
        if selection:
            sampling=mapping(selection.get("sampling"));selected=mapping(selection.get("selection"))
            candidate_count=process.get("candidate_frame_count",sampling.get("candidate_frames_decoded"))
            selected_count=process.get("selected_frame_count",selected.get("selected_frames"))
            fixation += f"候选帧数：{number(candidate_count,0)}；入选关键帧数：{number(selected_count,0)}。"
            strategy=sampling.get("strategy")
            if strategy:
                fixation += "候选采样方式："+safe_text({"uniform_trimmed":"避开首尾的均匀时间采样"}.get(strategy,strategy))+"；选帧规则与记录摘要见附录技术信息。"
    file_data, visible = e.get("E01", {}), e.get("E02", {})
    provenance = _status(file_data, "文件标识核查")
    if file_data.get("status") == "success":
        provenance += safe_text(file_data.get("verdict_label"), "记录了生成合成声明。" if file_data.get("declared_generated") else "未记录明确的生成合成声明。")
        provenance += "文件声明可能被添加或移除，内容凭证的结构识别不等于签名验证（证据编号 E01）。"
    visible_text = _status(visible, "画面文字识别")
    if visible.get("status") == "success":
        visible_text += ("检出了生成合成相关字样；字样可能被添加，仍需核查来源。" if visible.get("visible_label_found") else "本次未记录生成合成字样，不能据此排除生成内容。") + "（证据编号 E02）"
    detection_parts=[]
    for key,name in (("E03","通用分支"),("E04","人脸分支"),("E05","整图辅助分支")):
        data=e.get(key,{})
        detection_parts.append(_status(data,name) + (f"原始分数{raw_score_number(data.get('score'))}，{safe_text(data.get('interval'),'无可用区段')}（证据编号 {key}）。" if data.get("status")=="success" else f"（证据编号 {key}）"))
    face=e.get("E04",{})
    if face.get("highest_face"):
        highest=mapping(face["highest_face"])
        face_id=highest.get("face_id",highest.get("face_index","未记录"))
        detection_parts.append(f"最高分来自人脸{face_id}，分数为{number(highest.get('score',highest.get('logit')))}。"+
                               ("该脸未用于本次分档，分档汇总方式见技术信息。" if highest.get("used_for_band") is False else ""))
    if material.get("type")=="video":
        fs=mapping(face.get("frame_statistics"))
        detection_parts.append("人脸分支帧级统计："+(f"最小值{number(fs.get('min'))}，中位数{number(fs.get('median'))}，最大值{number(fs.get('max'))}。" if fs.get("status")=="recorded" else "未记录。"))
    source=e.get("E06",{}); pair=e.get("E08",{}); identity=e.get("E07",{})
    source_text=_status(source,"素材来源检索")+f"开放集状态：{safe_text(_OPEN_STATES.get(source.get('open_set_state'),source.get('open_set_state')))}（证据编号 E06）。"
    source_text+=_status(pair,"候选配对")+"几何对齐和像素差异用于辅助核查；对齐失败时暂不能判断是否同源（证据编号 E08）。"
    identity_text=_status(identity,"人物身份比对")+f"开放集状态：{safe_text(_OPEN_STATES.get(identity.get('open_set_state'),identity.get('open_set_state')))}；候选不等于确认身份（证据编号 E07）。"
    matched=[r for r in rows(package.get("clues")) if r.get("matched")]
    clues="\n".join(safe_text(r.get("rule_id"))+"："+safe_text(r.get("description",r.get("meaning",r.get("note"))),"命中固定线索条件，具体证据仍需复核") for r in matched) or "本次未记录命中的线索组合。"
    if any(r.get("rule_id")=="R1" for r in matched):
        clues+="\n背景一致而人脸不同提示需要核查换脸等解释，也可能是同一地点拍摄的不同人。该线索不单独形成自动伪造类型结论。"
    judgment=mapping(decision.get("judgment"))
    synthesis=[f"最终档位为“{label}”，规则参考档位为“{rule}”。", f"已记录的研判理由：{reason}"]
    if decision.get("score") is None:
        synthesis.append("当前采用规则结果，未生成 AIGC 可疑度评分；没有额外评分数值可以解释。")
    elif judgment.get("score_reason"):
        synthesis.append("评分依据："+safe_text(judgment["score_reason"]))
    if matched:
        synthesis.append("应结合以下多路线索核查："+clues)
    synthesis.append("本节为证据包确定性模板说明。未记录的跨模块关系不作推断，尚不能代替人工核查材料的形成过程。")
    missing=[f"{key} {_STATUS.get(v.get('status'),'未完成或未记录')}。" for key,v in e.items() if v.get("status") not in ("success","partial_success")]
    uncertainties=missing[:5] or ["检测分数与既有统计受来源分布影响，不能排除漏检或误报。"]
    uncertainties.append("未记录的证据关系仍需结合原始材料核实。")
    return {"summary":summary[:200],"process":{"fixation":fixation,"provenance":provenance,"visible_text":visible_text,
            "detection":"\n".join(detection_parts),"source":source_text,"identity":identity_text,"clues":clues},
            "synthesis":"\n\n".join(synthesis),"uncertainties":uncertainties[:6],
            "next_steps":["核对原始文件和取得过程，确认材料完整性。","结合候选素材与配对差异核查来源、时间和编辑过程。","由人员审阅各模块证据和不确定性，必要时进一步检验。"],
            "figures":[default_figure_placement(f) for f in rows(package.get("figures")) if f.get("required")],
            "_section_sources":{k:"template" for k in ("summary",*PROCESS_FIELDS,"synthesis","uncertainties","next_steps")},
            "_writer":{"status":"fallback","reason":reason}}


def merge_writer_content(template, writer_result) -> dict:
    """只覆盖服务层保留的有效章节；档位、评分不在本结构中且不能被撰写改动。"""
    output=deepcopy(mapping(template)); result=mapping(writer_result)
    wrapped="content" in result
    incoming=mapping(result.get("content")) if wrapped else result
    invalid=set(result.get("invalid_sections",[])) if wrapped else set()
    if wrapped and result.get("status")=="fallback":invalid.add("all")
    sources=mapping(output.get("_section_sources"))
    if "all" not in invalid:
        for key in ("summary","synthesis"):
            if key not in invalid and isinstance(incoming.get(key),str) and incoming[key].strip():
                output[key]=incoming[key];sources[key]="writer"
        process=mapping(incoming.get("process"))
        for key in PROCESS_FIELDS:
            if key not in invalid and "process" not in invalid and isinstance(process.get(key),str) and process[key].strip():
                output.setdefault("process",{})[key]=process[key];sources[key]="writer"
        for key in ("uncertainties","next_steps"):
            value=incoming.get(key)
            if key not in invalid and isinstance(value,list) and value and all(isinstance(v,str) and v.strip() for v in value):
                output[key]=deepcopy(value);sources[key]="writer"
        if "figures" not in invalid and isinstance(incoming.get("figures"),list):
            output["figures"]=[{**f,"automatic":False} for f in rows(incoming["figures"])]
    output["_section_sources"]=sources
    output["_writer"]={k:deepcopy(v) for k,v in result.items() if k!="content"} if wrapped else {"status":"success"}
    output["_template"]=deepcopy({k:v for k,v in mapping(template).items() if not k.startswith("_")})
    return output
