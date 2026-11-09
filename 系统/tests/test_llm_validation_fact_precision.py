"""原始分数舍入及明确事实边界的纯夹具正反例。"""
from copy import deepcopy
from decimal import Decimal
import pytest
from reporting import llm_validation as candidate

SCALAR_CASES = [
    ("2有效数字", "0.005202112567123126", "0.0052", True),
    ("3有效数字含尾零", "0.005202112567123126", "0.00520", True),
    ("4有效数字", "0.005202112567123126", "0.005202", True),
    ("5有效数字", "0.005202112567123126", "0.0052021", True),
    ("6有效数字", "0.005202112567123126", "0.00520211", True),
    ("7有效数字", "0.005202112567123126", "0.005202113", True),
    ("8有效数字", "0.005202112567123126", "0.0052021126", True),
    ("精确来源仍允许", "0.005202112567123126", "0.005202112567123126", True),
    ("9有效数字不扩展", "0.005202112567123126", "0.00520211257", False),
    ("相邻错误数字", "0.005202112567123126", "0.005202114", False),
    ("截断代替舍入", "0.005202112567123126", "0.005202112", False),
    ("科学计数等价", "0.005202112567123126", "5.202113e-3", True),
    ("科学计数相邻值", "0.005202112567123126", "5.202114e-3", False),
    ("科学计数指数错位", "0.005202112567123126", "5.202113e-2", False),
    ("负数同精度", "-0.005202112567123126", "-0.005202113", True),
    ("正负号不能交换", "-0.005202112567123126", "0.005202113", False),
    ("正数半数进位", "0.0012345675", "0.001234568", True),
    ("负数半数远离零", "-0.0012345675", "-0.001234568", True),
    ("不采用同位截断", "0.0012345675", "0.001234567", False),
    ("跨量级进位", "0.0099999995", "0.01", True),
    ("极小非零不靠小数截断", "1.23456789e-100", "1.234568e-100", True),
    ("大非整数精确派生", "1234567.89", "1.234568e6", True),
    ("大非整数相邻错误", "1234567.89", "1.234569e6", False),
    ("整数计数不新增近似", "599", "600", False),
    ("整数精确值保留", "599", "599", True),
    ("年份不新增近似", "20261107", "20000000", False),
    ("零精确值保留", "0", "0", True),
    ("旧8g小阈值仍允许", "0.000072013129572372", "7.201313e-5", True),
    ("阈值错误尾数不允许", "0.000072013129572372", "7.2013131e-5", False),
    ("百分比不扩展新形式", "0.005202112567123126", "0.5202113%", False),
    ("带空格百分号不扩展", "0.005202112567123126", "0.005202113 %", False),
    ("全角百分号不扩展", "0.005202112567123126", "0.005202113％", False),
    ("百分比精确旧形式保留", "0.005202112567123126", "0.5202112567123126%", True),
    ("同值混合也不偷渡百分比", "0.005202112567123126", "0.005202113，另称0.5202113%", False),
    ("没有来源就不允许", "", "0.005202113", False),
]


@pytest.mark.parametrize("name,source,output,valid", SCALAR_CASES)
def test_fixed_scalar_cases(name, source, output, valid):
    allowed = {Decimal(source)} if source else set()
    assert (not candidate._unmatched_numbers(output, allowed)) is valid, name


def fixtures():
    evidence = {"rule_band": "review", "constraints": {"floor_band": "review", "allow_obvious": False},
        "items": [{"item_id": "E04", "data": {"score": 0.4}},
                  {"item_id": "E05", "data": {"score": 0.005202112567123126}}], "figures": [{"figure_id": "F01"}]}
    judge = {"final_band": "review", "score": 50,
        "evidence_assessment": [{"item_id": "E05", "observation": "原始分数为0.005202113", "direction": "neutral", "weight": "low", "reason": "整图分支只作辅助参考"}],
        "visual_observations": [], "band_reason": "综合已有线索，建议人工复核", "score_reason": "用于安排复核优先级",
        "relation_to_rule": "same", "adjustment_basis_ids": [], "adjustment_reason": "", "uncertainties": ["实际适用范围仍需核查"]}
    paragraph = "本次分析根据检测分支给出的风险提示展开，现有结果只提供辅助线索。各项检测的适用范围有限，缺失的材料需要后续人工核查。"
    writer = {"summary": "最终档位为建议人工复核，评分50。", "process": {k: "本步骤仍需人工核查。" for k in candidate.PROCESS_FIELDS},
        "synthesis": paragraph * 15, "uncertainties": ["来源仍需核实", "适用范围仍需核查", "结论需要人工复核"],
        "next_steps": ["核查原始材料", "核对候选关系", "组织人工复核"], "figures": []}
    writer["process"]["detection"] = "整图原始分数为0.005202113。"
    return evidence, judge, writer


def test_public_judge_and_writer_accept_same_source_rounding():
    view, judge, writer = fixtures()
    assert candidate.validate_judge(judge, view)["valid"]
    assert candidate.validate_writer(writer, view, judge)["valid"]


def test_local_evidence_item_still_cannot_borrow_another_score():
    view, judge, _ = fixtures()
    judge["evidence_assessment"][0]["item_id"] = "E04"
    checked = candidate.validate_judge(judge, view)
    assert not checked["valid"] and "E04 观察数字与该证据项不一致" in checked["errors"]


def test_ocr_material_does_not_supply_trusted_numeric_rounding():
    view, judge, writer = fixtures()
    view["items"][1]["data"]["score"] = 0.1
    view["items"].append({"item_id": "E02", "data": {"material_content": {"kind": "material_content", "text": "0.005202112567123126"}}})
    assert not candidate.validate_judge(judge, view)["valid"]
    assert not candidate.validate_writer(writer, view, judge)["valid"]


@pytest.mark.parametrize("metric,text,valid", [
    ("recall", "整图召回率0.005202113", True),
    ("fpr", "整图召回率0.005202113", False),
    ("fake_percent", "整图假阴性率0.005202113", False),
    ("auc", "整图误报率0.005202113", False),
    ("fpr", "整图误报率0.5202112567123126%", True),
    ("fpr", "整图误报率0.5202113%", False),
])
def test_metric_identity_and_percent_policy_unchanged(metric, text, valid):
    view, judge, writer = fixtures()
    view["items"][1]["data"]["statistics"] = {metric: 0.005202112567123126}
    judge["evidence_assessment"][0]["observation"] = text
    writer["process"]["detection"] = text
    assert candidate.validate_judge(judge, view)["valid"] is valid
    assert candidate.validate_writer(writer, view, judge)["valid"] is valid


def test_writer_metric_cannot_borrow_from_another_item():
    view, judge, writer = fixtures()
    view["items"][0]["data"]["statistics"] = {"recall": 0.005202112567123126}
    writer["process"]["detection"] = "E05召回率0.005202113"
    assert not candidate.validate_writer(writer, view, judge)["valid"]


@pytest.mark.parametrize("text,valid", [("整图原始分数0.005202113", True), ("整图召回率0.005202113", False)])
def test_figure_text_shares_numeric_but_not_metric_permission(text, valid):
    view, judge, writer = fixtures()
    writer["figures"] = [{"figure_id": "F01", "section": "detection", "after_paragraph": 1, "caption": text,
                          "reading_guide": "分数只供辅助筛查，不能替代人工核查"}]
    assert candidate.validate_writer(writer, view, judge, uploaded_figure_ids=["F01"])["valid"] is valid



def view():
    return {"rule_band": "review", "constraints": {"floor_band": "review", "allow_obvious": False}, "items": [
        {"item_id": "E03", "data": {"status": "success", "score": -2.3}},
        {"item_id": "E04", "data": {"status": "no_face", "known_reliability": "人脸分支大约三分之二的换脸图进不了复核。", "reliability_statistics": {"face_auc": 0.6627}}},
        {"item_id": "E06", "data": {"status": "skipped"}},
        {"item_id": "E07", "data": {"status": "skipped"}},
        {"item_id": "E10", "data": {"status": "unrecorded", "analysis_time": None, "module_status": {"deepfake": "skipped", "source": "skipped", "identity": "skipped"}}}]}


@pytest.mark.parametrize("text,valid", [
    ("鉴于AUC指标的限制，约三分之一的换脸样本可能无法被有效识别，反之亦然。", False),
    ("人脸检测低分仍存在约三成伪造样本漏检可能", False),
    ("人脸分支有30%的伪造样本漏检。", False),
    ("人脸分支可能存在漏检，仍须人工复核。", True),
    ("AUC不能推算漏检率，也不能说明约三分之一的换脸样本未被检出。", True),
    ("不能表述为约三成伪造样本漏检。", True),
    ("不能表述为约三成伪造样本漏检，但实际约三成伪造样本漏检。", False),
    ("人脸分支大约三分之二的换脸图进不了复核。", True),
    ("通用分支大约三分之二的换脸图进不了复核。", False),
    ("人脸分支大约三分之二的换脸图漏检。", False),
    ("人脸分支大约三分之一的换脸图进不了复核。", False),
    ("人脸99%分界线上下均可能存在样本漏检。", True),
])
def test_quantified_performance_source_boundary(text, valid):
    assert (not candidate._performance_errors(text, view(), path="detection")) is valid


@pytest.mark.parametrize("metric,value,text,valid", [
    ("fnr", 0.3, "人脸分支约三成伪造样本漏检。", True),
    ("fnr", 0.3, "人脸分支30%的伪造样本漏检。", True),
    ("recall", 0.3, "人脸分支约三成伪造样本漏检。", False),
    ("fake_percent", 30, "人脸分支约三成伪造样本漏检。", False),
    ("face_auc", 0.7, "人脸分支约三成伪造样本漏检。", False),
    ("fnr", 1 / 3, "人脸分支三分之一的伪造样本漏检。", True),
    ("fnr", 0.3, "通用分支约三成伪造样本漏检。", False),
])
def test_same_branch_same_metric_quantity(metric, value, text, valid):
    evidence = view(); evidence["items"][1]["data"]["statistics"] = {metric: value}
    assert (not candidate._performance_errors(text, evidence, path="detection")) is valid


@pytest.mark.parametrize("section,text,valid", [
    ("source", "因模块未执行，不能表述为已检索且未发现，仅说明当前无素材来源线索。", True),
    ("identity", "因模块未执行，不能表述为已比对且未命中，仅说明当前无身份关联线索。", True),
    ("source", "本步骤未执行素材比对，不能表述为未检出可信候选或排除同源可能。", True),
    ("source", "不能表述为已检索且未发现相似源。", True),
    ("identity", "因模块未成功运行，不能得出无人脸匹配或排除特定身份的结论。", True),
    ("identity", "因未检出人脸且模块未执行，无身份比对记录。不能推断库内有无此人。", True),
    ("identity", "人脸检测未发现人脸[E04]。身份候选模块跳过[E07]。", True),
    ("source", "素材候选模块跳过，无开放集状态[E06]。未检出同源候选或编辑线索。", False),
    ("source", "不能表述为已检索且未发现，但实际未检出可信候选。", False),
    ("source", "不能表述为已检索且未发现，检索已执行并未命中。", False),
    ("source", "不能表述为已检索且未发现。然而检索未命中。", False),
    ("source", "并非不能表述为已检索且未发现。", False),
    ("source", "不是不能表述为已检索且未发现。", False),
    ("source", "素材检索未命中且不是没有执行。", False),
    ("source", "未发现执行记录，不能判断是否命中。", True),
    ("identity", "因未检出人脸匹配且模块未执行，无身份比对记录。", False),
    ("visible_text", "画面文字模块跳过，未识别到生成相关字样或水印[E02]。", False),
    ("visible_text", "不能报告识别已完成，也不能得出未发现文字的结论。", True),
    ("source", "不能报告是否检出候选，也不能得出库内无匹配的结论。", True),
    ("source", "不能报告是否检出候选，但实际未检索到任何候选。", False),
    ("visible_text", "OCR模块跳过，无法据此排除水印。", True),
    ("visible_text", "OCR模块跳过，无法据此排除水印，但实际排除了水印。", False),
    ("source", "素材候选模块跳过，无开放集状态[E06]。未检索到可信候选素材。", False),
    ("source", "素材候选模块E06状态为跳过，未执行检索。无候选结果不代表库中无底图，仅说明本次未进行比对。", True),
    ("source", "无候选输出，未执行检索。", True),
    ("source", "无可信候选，仅说明本次未进行比对。", False),
    ("source", "无候选结果，但实际检索已完成，未命中。", False),
    ("source", "未检出可信候选不等于库中无底图，仅表示本次未执行或未返回结果。", False),
])
def test_execution_negation_scope_and_cross_subject(section, text, valid):
    evidence = view(); evidence["items"].append({"item_id": "E02", "data": {"status": "skipped"}})
    assert (not candidate._execution_fact_errors(text, evidence, path=section)) is valid


@pytest.mark.parametrize("status,text,valid", [
    ("no_face", "人脸检测失败，无法评估局部篡改或换脸风险。", False),
    ("no_face", "人脸分支执行失败。", False),
    ("no_face", "人脸检测未检出人脸，不等于检测执行失败。", True),
    ("no_face", "不能表述为人脸检测失败。", True),
    ("no_face", "如果人脸检测失败，应保留错误记录。", True),
    ("no_face", "未检出人脸导致身份比对未执行。", True),
    ("no_face", "不能说人脸检测失败，但实际人脸检测失败。", False),
    ("failed", "人脸检测失败，需补查执行日志。", True),
    ("failed", "人脸检测未检出人脸。", False),
    ("failed", "不能表述为人脸检测未检出人脸。", True),
])
def test_no_face_is_not_execution_failure(status, text, valid):
    evidence = view(); evidence["items"][1]["data"]["status"] = status
    assert (not candidate._execution_fact_errors(text, evidence, path="detection")) is valid


@pytest.mark.parametrize("text,valid", [
    ("处理过程未记录分析时间与模块状态[E10]。", False),
    ("模块状态均未记录。", False),
    ("没有任何模块状态记录。", False),
    ("未记录任何模块状态。", False),
    ("分析时间未记录。", True),
    ("处理过程总状态为unrecorded。", True),
    ("部分模块状态未记录。", True),
    ("模块状态未完整记录。", True),
    ("完整处理日志未记录。", True),
    ("不能说没有任何模块状态记录。", True),
    ("处理过程模块状态未记录，部分子模块显示跳过[E10]。", True),
    ("处理过程模块状态均未记录。", False),
])
def test_e10_total_missing_is_not_partial_missing(text, valid):
    assert (not candidate._execution_fact_errors(text, view(), path="provenance")) is valid


def ratio_view():
    evidence = view()
    evidence["items"].append({"item_id": "E08", "data": {"reliability_statistics": {"records": [
        {"metric": "original_face_replaced", "title": "原图检出", "numerator": 35, "denominator": 54, "rate": 35 / 54}]}}})
    return evidence


@pytest.mark.parametrize("text,valid", [
    ("该规则基于开发实验，原图检出率为35/54，非盲测验证[E08]。", True),
    ("原图检出率为35/54 [E08]。", True),
    ("原图检出率为36/54 [E08]。", False),
    ("原图检出率为35/53 [E08]。", False),
    ("退化图检出率为35/54 [E08]。", False),
    ("原图误报率为35/54 [E08]。", False),
    ("通用分支检出率为35/54 [E03]。", False),
    ("配对模块基于构造约束提示人脸替换线索，原图检出率为35/54。", True),
])
def test_fraction_requires_same_row_and_metric(text, valid):
    assert (not candidate._performance_errors(text, ratio_view(), path="clues")) is valid


def test_fraction_cannot_mix_row_numerator_denominator_or_inconsistent_rate():
    for updates in ({"numerator": 34}, {"denominator": 55}, {"rate": 0.7}, {"metric": "false_negative"}):
        evidence = ratio_view()
        evidence["items"][-1]["data"]["reliability_statistics"]["records"][0].update(updates)
        assert candidate._performance_errors("原图检出率35/54[E08]。", evidence, path="clues")


def test_ban_phrases_contract_remains_literal_even_when_negated():
    for text in ("不能确认真实。", "高档位不代表确认伪造。"):
        assert any("含禁用说法" in e for e in candidate._common_errors(text, set(), evidence_view=view()))


@pytest.mark.parametrize("text,valid", [
    ("所有结论需经人工复核后方可作为侦查依据。", True),
    ("所有结论必须经过人工审阅后才能使用。", True),
    ("所有结论已经人工复核后方可作为侦查依据。", False),
    ("所有结论经人工复核后已经形成。", False),
    ("所有结论需经人工复核后方可使用，但已经过人工核查。", False),
])
def test_future_human_review_condition_is_not_completion(text, valid):
    assert (not candidate._execution_fact_errors(text, view(), path="synthesis")) is valid


def test_public_writer_rejects_contradiction_in_uncertainties_not_all_sections():
    _, _, writer = fixtures()
    writer["process"]["detection"] = "人脸分支未检出人脸。"
    writer["uncertainties"][0] = "人脸检测失败，无法评估局部风险。"
    checked = candidate.validate_writer(writer, view(), {"final_band": "review", "score": 50})
    assert "uncertainties" in checked["invalid_sections"]
    assert "detection" not in checked["invalid_sections"]


def test_public_judge_rejects_same_no_face_failure_fact():
    _, judge, _ = fixtures()
    judge["evidence_assessment"] = [{"item_id": "E04", "observation": "人脸检测失败。", "direction": "neutral", "weight": "low", "reason": "需要复核。"}]
    assert any("no_face" in e for e in candidate.validate_judge(judge, view())["errors"])
