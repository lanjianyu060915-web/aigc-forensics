"""撰写结论与评分局部作用域的固定正反例。"""
import pytest
from reporting import llm_validation as candidate

LABELS = {"none": "未见明显AIGC痕迹", "review": "建议人工复核", "obvious": "明显AIGC痕迹", "undetermined": "无法完成判断"}

CASES = [
    ("空评分不跨句取尺寸", "评分未给出。材料为512像素人像图。", "summary", "none", None, None, True),
    ("空评分不跨逗号取尺寸", "评分未给出，图宽512像素。", "summary", "none", None, None, True),
    ("空评分不跨分号", "评分未给出；材料为512像素。", "summary", "none", None, None, True),
    ("空评分不跨换行", "评分未给出\n材料为512像素。", "summary", "none", None, None, True),
    ("空评分不跨英文句点", "评分未给出.材料为512像素。", "summary", "none", None, None, True),
    ("真评分512仍拒", "本次最终评分为512。", "summary", "none", None, None, False),
    ("同句真评分仍拒", "材料512像素，最终评分为512。", "summary", "none", None, None, False),
    ("明确赋值带逗号仍拒", "最终评分为，512。", "summary", "none", None, None, False),
    ("明确赋值带分号仍拒", "最终评分为；512。", "summary", "none", None, None, False),
    ("前句为空后句真评分仍拒", "评分未给出。最终评分为512。", "summary", "none", None, None, False),
    ("同句先空后真仍拒", "评分未给出，但最终评分为512。", "summary", "none", None, None, False),
    ("同分合法", "最终评分为45。", "summary", "review", 45, None, True),
    ("不同分拒", "最终评分为46。", "summary", "review", 45, None, False),
    ("小数同分保留", "最终评分：45.5。", "summary", "review", 45.5, None, True),
    ("多个评分逐一校验", "评分45。复核后的最终评分46。", "summary", "review", 45, None, False),
    ("next_steps明确动作", "建议人工复核原始材料。", "next_steps", "none", None, None, True),
    ("next_steps短动作", "建议人工复核。", "next_steps", "none", None, None, True),
    ("同句正确结论与动作", "最终档位未见明显AIGC痕迹，建议人工复核原始材料。", "summary", "none", None, None, True),
    ("分号正确结论与动作", "最终档位未见明显AIGC痕迹；建议人工复核原始材料。", "summary", "none", None, None, True),
    ("不同句正确结论与短动作", "最终结论为未见明显AIGC痕迹。建议人工复核。", "summary", "none", None, None, True),
    ("近真实缺失后的动作", "最终结论为未见明显AIGC痕迹，评分未给出。因多模块缺失，无法排除风险，建议人工复核。", "summary", "none", None, None, True),
    ("图件说明明确动作", "建议人工复核原图。", "图件说明", "none", None, None, True),
    ("最终调整为review拒", "最终档位调整为建议人工复核。", "summary", "none", None, None, False),
    ("next_steps也不能改档", "最终档位调整为建议人工复核。", "next_steps", "none", None, None, False),
    ("同句先正确后新研判拒", "最终档位未见明显AIGC痕迹，研判为建议人工复核。", "summary", "none", None, None, False),
    ("跨句先动作后改档拒", "建议人工复核原始材料。最终档位为建议人工复核。", "summary", "none", None, None, False),
    ("跨句先正确后改档拒", "最终档位为未见明显AIGC痕迹。现调整结论为建议人工复核。", "summary", "none", None, None, False),
    ("动作不能掩盖obvious", "建议人工复核原始材料，最终定档为明显AIGC痕迹。", "next_steps", "none", None, None, False),
    ("归为review拒", "本案应归为建议人工复核。", "summary", "none", None, None, False),
    ("设为review拒", "将档位设为建议人工复核。", "summary", "none", None, None, False),
    ("逗号拆开结论锚点仍拒", "最终档位为，建议人工复核原始材料。", "summary", "none", None, None, False),
    ("跨句拆开结论锚点仍拒", "最终档位为。建议人工复核原始材料。", "next_steps", "none", None, None, False),
    ("字面档位语境拒", "建议人工复核档位。", "summary", "none", None, None, False),
    ("无上下文短句保守拒", "建议人工复核。", "summary", "none", None, None, False),
    ("当前review结论不改变", "最终档位为建议人工复核，评分45。", "summary", "review", 45, None, True),
    ("正确规则参考仍允许", "规则参考档位为建议人工复核。", "synthesis", "none", None, "review", True),
    ("伪造规则参考仍拒", "规则参考档位为建议人工复核。", "synthesis", "none", None, "none", False),
    ("改名review不泛豁免", "最终档位未见明显AIGC痕迹，建议人工复核后重新定档。", "summary", "none", None, None, False),
]


@pytest.mark.parametrize("name,text,path,selected,score,rule_band,valid", CASES)
def test_fixed_scope_cases(name, text, path, selected, score, rule_band, valid):
    errors = candidate._decision_text_errors(text, LABELS, selected, score, path=path, rule_band=rule_band)
    assert (not errors) is valid, (name, errors)
