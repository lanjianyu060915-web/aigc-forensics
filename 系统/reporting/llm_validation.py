"""大模型输出的确定性约束；文字事实仍须人工复核。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
import math
import re
from typing import Any

from .llm_client import DEFAULT_LLM_CONFIG, assert_upload_safe, LLMClientError


BAN_PHRASES = ("确认伪造", "确认真实", "就是原图", "就是此人", "百分之百", "毫无疑问", "一定是", "肯定是", "对不齐，所以不是同一张", "库里没有底图", "脸不在身份库")
BANDS = {"none": (0, 39), "review": (40, 69), "obvious": (70, 100)}
SCORE_POLICY_BOUNDARIES = (0, 15, 16, 29, 30, 39, 40, 54, 55, 69, 70, 84, 85, 100)
PROCESS_FIELDS = ("fixation", "provenance", "visible_text", "detection", "source", "identity", "clues")
WRITER_SECTIONS = ("summary", *PROCESS_FIELDS, "synthesis")


def _string(maximum=None, enum=None):
    value = {"type": "string"}
    if maximum is not None:
        value["maxLength"] = maximum
    if enum is not None:
        value["enum"] = list(enum)
    return value


def _object(properties):
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


def _array(item, maximum=40):
    return {"type": "array", "maxItems": maximum, "items": item}


JUDGE_SCHEMA = _object({
    "final_band": _string(enum=BANDS), "score": {"type": "integer", "minimum": 0, "maximum": 100},
    "evidence_assessment": _array(_object({"item_id": _string(20), "observation": _string(600),
        "direction": _string(enum=("supports_aigc", "against_aigc", "neutral")),
        "weight": _string(enum=("high", "medium", "low")), "reason": _string(400)})),
    "visual_observations": _array(_object({"figure_id": _string(20), "observation": _string(80)}), 24),
    "band_reason": _string(400), "score_reason": _string(300),
    "relation_to_rule": _string(enum=("same", "raised", "lowered")),
    "adjustment_basis_ids": _array(_string(20)), "adjustment_reason": _string(200),
    "uncertainties": _array(_string(80), 12),
})
WRITER_SCHEMA = _object({
    "summary": _string(200),
    "process": _object({key: _string(limit) for key, limit in zip(PROCESS_FIELDS, (150, 250, 200, 600, 450, 250, 250))}),
    "synthesis": _string(1500), "uncertainties": _array(_string(80), 6), "next_steps": _array(_string(60), 6),
    "figures": _array(_object({"figure_id": _string(20), "section": _string(enum=WRITER_SECTIONS),
        "after_paragraph": {"type": "integer", "minimum": 1}, "caption": _string(30), "reading_guide": _string(120)}), 24),
})


def _schema_errors(value, schema, path="$"):
    errors = []
    kind = schema.get("type")
    valid_type = {"object": isinstance(value, dict), "array": isinstance(value, list), "string": isinstance(value, str),
                  "integer": isinstance(value, int) and not isinstance(value, bool)}.get(kind, True)
    if not valid_type:
        return [f"{path} 类型应为 {kind}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} 不在允许值内")
    if kind == "object":
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}.{name} 缺少字段")
        for name, item in value.items():
            if name not in schema.get("properties", {}):
                errors.append(f"{path}.{name} 不允许新增字段")
            else:
                errors.extend(_schema_errors(item, schema["properties"][name], path + "." + name))
    elif kind == "array":
        if len(value) > schema.get("maxItems", math.inf):
            errors.append(f"{path} 条目过多")
        for index, item in enumerate(value):
            errors.extend(_schema_errors(item, schema["items"], f"{path}[{index}]"))
    elif kind == "string" and len(value) > schema.get("maxLength", math.inf):
        errors.append(f"{path} 超过长度限制")
    elif kind == "integer" and not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
        errors.append(f"{path} 超出数值范围")
    return errors


_NUMBER = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?%?")
_ALIAS = re.compile(r"材料(?:代号)?\d{3,}")
_IDS = re.compile(r"\b(?:E|F|R)\d+\+?\b|\b[0-9a-fA-F]{32,64}\b|材料(?:代号)?\d{3,}")


def _numbers(value):
    result = set()
    if isinstance(value, bool) or value is None:
        return result
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return result
        result.add(Decimal(str(value)))
    elif isinstance(value, str):
        for found in _NUMBER.findall(_IDS.sub("", value)):
            try:
                result.add(Decimal(found.rstrip("%")) / (100 if found.endswith("%") else 1))
            except InvalidOperation:
                pass
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if key not in ("item_id", "figure_id", "sha256", "prompt_sha256", "file_sha256"):
                result.update(_numbers(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            result.update(_numbers(item))
    return result


def _unmatched_numbers(value, allowed):
    # 接纳旧图的小数舍入和新图的八位有效数字；都从实际来源数值精确派生。
    rounded = allowed | {number.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
                         for number in allowed if abs(number) < Decimal("1e20") for digits in range(2, 7)}
    rounded.update(Decimal(format(float(number), ".8g")) for number in allowed
                   if number.is_finite() and abs(number) < Decimal("1e308"))
    # 新增范围：非整数来源的二至八位有效数字，使用Decimal精确四舍五入。
    # 不近似整数计数，不引入数值容差；百分比仍沿用既有规则，不扩大单位转换权限。
    significant = {number.quantize(Decimal(1).scaleb(number.adjusted() - digits + 1), rounding=ROUND_HALF_UP)
                   for number in allowed if number.is_finite() and abs(number) < Decimal("1e308")
                   and number != number.to_integral_value() for digits in range(2, 9)}
    percentage_numbers = set()
    for text in _strings(value):
        cleaned = _IDS.sub("", text)
        for match in _NUMBER.finditer(cleaned):
            if match.group().endswith("%") or re.match(r"\s*[%％]", cleaned[match.end():]):
                percentage_numbers.update(_numbers(match.group()))
    return _numbers(value) - rounded - (significant - percentage_numbers)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


_TRUSTED_NUMBER_FIELDS = {
    "score", "logit", "thresholds", "statistics", "reliability", "reliability_statistics", "per_face", "frame_scores",
    "frame_statistics", "highest_face", "selected_face", "records", "candidates", "faces", "open_set", "metrics",
    "alignment", "face_count", "width", "height", "duration_seconds", "fps", "size_bytes", "runtime_seconds",
    "frame_selection", "key_frames", "video_selection", "rank", "face_index", "confidence", "bbox", "polygon",
}
_UNTRUSTED_FIELDS = {"material_content", "metadata", "aigc_labels", "digital_source_type", "generator_hints",
                     "text", "raw", "notice", "reason", "explanation", "description", "error_message", "notes"}


def _structured_numbers(value, key=""):
    """只采纳受信结构中的数值叶子，材料文本和任意字符串不转为系统事实。"""
    if isinstance(value, Mapping):
        if value.get("kind") == "material_content":
            return set()
        return set().union(*(_structured_numbers(item, name) for name, item in value.items()
                              if name not in _UNTRUSTED_FIELDS)) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(_structured_numbers(item, key) for item in value)) if value else set()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        values = _numbers(value)
        if "percent" in key.lower():
            values |= {number / 100 for number in values}
        return values
    return set()


def _item_numbers(item):
    data = item.get("data", item)
    if not isinstance(data, Mapping):
        return set()
    allowed = set().union(*(_structured_numbers(value, key) for key, value in data.items()
                            if key in _TRUSTED_NUMBER_FIELDS)) if data else set()
    # 此字段由本地证据包从版本化说明生成，区别于OCR/文件元数据材料。
    allowed.update(_numbers(data.get("known_reliability", "")))
    allowed.update(_numbers(data.get("interval", "")))
    timestamp = data.get("analysis_time")
    if isinstance(timestamp, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ][0-9:.+Z-]+", timestamp):
        allowed.update(_numbers(timestamp))
    return allowed


def _evidence_numbers(evidence_view):
    allowed = set().union(*(_item_numbers(item) for item in _items(evidence_view).values()))
    allowed.update(_structured_numbers(evidence_view.get("video_selection", {})))
    return allowed


_PERFORMANCE_METRICS = {
    "accuracy": (r"准确率|正确率|accuracy", r"(?:^|_)(?:accuracy|acc)(?:_|$)"),
    "auc": (r"(?<![A-Za-z])AUC(?![A-Za-z])", r"(?:^|_)auc(?:_|$)"),
    "recall": (r"检出率|召回率|敏感度|recall|sensitivity", r"(?:^|_)(?:recall|sensitivity|tpr)(?:_|$)"),
    "fpr": (r"误报率|误判率|假阳性率|false_positive_rate|FPR", r"(?:^|_)(?:fpr|false_positive_rate)(?:_|$)"),
    "fnr": (r"假阴性率|漏检率|漏报率|漏判率|false_negative_rate|FNR", r"(?:^|_)(?:fnr|false_negative_rate|miss_rate)(?:_|$)"),
}
_METRIC_NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?![\d.eE])[ \t]*%?"
_NAMED_RATE_METRICS = {
    "recall": {"original_face_replaced", "degraded_face_replaced"},
    "fpr": {"false_positive", "copy_false_swap", "beauty_false_swap", "negative_false_same_source"},
    "fnr": {"false_negative"},
}


def _negated_or_hypothetical(text, start, end, *, human=False):
    """只识别局部明确否定或假设，不声称能穷尽中文语义。"""
    boundary = max(text.rfind(mark, 0, start) for mark in ("，", ",", "。", "；", ";", "\n")) + 1
    prefix = re.split(r"但是|然而|不过|但|而是|事实上|实际上", text[boundary:start])[-1]
    if re.search(r"(?:如果|假如|假设|即使|即便|若|建议|需要|应当|有待|须由)[^，。；\n]{0,24}$", prefix):
        return True
    if re.search(r"(?:不是|并非|不等于|不等同于?|不代表|(?:不能|不可|不应|不得|无法)[^，。；\n]{0,24}(?:称|当|视|写|说|声称|解释|推算|换算|推断|排除)[^，。；\n]{0,24})[‘“\"'\s]*$", prefix):
        return True
    if (re.search(r"(?:不能|不可|不应|不得|无法)(?:据此|由此|直接)?\s*$", prefix)
            and re.match(r"推算|换算|推断|排除|断言|认定", text[start:end])):
        return True
    if human and re.search(r"(?:尚未|未经|未曾|没有|尚无|未)[^，。；\n]{0,6}$", prefix) and not re.search(r"并非未|不是未", prefix):
        return True
    return bool(re.search(r"不是|并非|不等于|不代表", text[start:end]))


def _metric_occurrences(text, pattern):
    forward = r"(?:" + pattern + r")[^\d，。；\n]{0,12}(" + _METRIC_NUMBER + r")"
    reverse = r"(?<![A-Za-z0-9_])(" + _METRIC_NUMBER + r")[ \t]*的?[ \t]*(?:" + pattern + r")"
    return [match for expression in (forward, reverse)
            for match in re.finditer(expression, text, re.I)
            if not _negated_or_hypothetical(text, match.start(), match.end())]


def _metric_mentions(text, pattern):
    return [match.group(1).replace(" ", "").replace("\t", "") for match in _metric_occurrences(text, pattern)]


def _performance_values(evidence_view):
    allowed = {name: set() for name in _PERFORMANCE_METRICS}
    def collect(value):
        if isinstance(value, Mapping):
            metric_name = str(value.get("metric", "")).lower()
            named = {name for name, (_, field_pattern) in _PERFORMANCE_METRICS.items()
                     if re.search(field_pattern, metric_name, re.I) or metric_name in _NAMED_RATE_METRICS.get(name, set())}
            title = str(value.get("title", ""))
            for key, child in value.items():
                if key in _UNTRUSTED_FIELDS:
                    continue
                for name, (pattern, field_pattern) in _PERFORMANCE_METRICS.items():
                    # real_percent/fake_percent只表示当前档位占比，不能自动借作二元性能率。
                    matching_rate = key == "rate" and (name in named if named else re.search(pattern, title, re.I))
                    if re.search(field_pattern, key, re.I) or matching_rate:
                        allowed[name].update(_structured_numbers(child, key))
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
    for item in _items(evidence_view).values():
        data = item.get("data", item)
        for field in ("statistics", "reliability", "reliability_statistics"):
            collect(data.get(field, {}))
        for name, (pattern, _) in _PERFORMANCE_METRICS.items():
            for number in _metric_mentions(str(data.get("known_reliability", "")), pattern):
                allowed[name].update(_numbers(number))
    return allowed


def _performance_reference_ids(text, evidence_view):
    """只取当前语句的明确分支归属；不借别的分支数字凑性能率。"""
    items = _items(evidence_view)
    ids = set(re.findall(r"(?<![A-Za-z0-9_])E\d{2}(?!\d)", text)) & set(items)
    for identifier, subject in (("E03", r"通用(?:分支|检测|生成)|融合分支"),
                                ("E04", r"人脸(?:分支|检测|AUC|召回率|检出率|漏检率|误报率)|换脸检测"),
                                ("E05", r"整图(?:分支|辅助|检测)"),
                                ("E08", r"配对(?:规则|检出|实验|模块)")):
        if identifier in items and re.search(subject, text):
            ids.add(identifier)
    if not ids and len(items) == 1:
        ids = set(items)
    return ids


def _chinese_integer(text):
    digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if text.isdigit():
        return int(text)
    if text in digits:
        return digits[text]
    if text == "百":
        return 100
    if "十" in text and text.count("十") == 1:
        high, low = text.split("十")
        if high in ("", *digits) and low in ("", *digits):
            return (digits[high] if high else 1) * 10 + (digits[low] if low else 0)
    return None


_RATE_QUANTITY = r"\d+(?:\.\d+)?\s*[%％]|百分之[零一二两三四五六七八九十百\d.]+|[一二两三四五六七八九十百\d]+分之[一二两三四五六七八九十百\d]+|[一二两三四五六七八九十]+成"


def _rate_quantity_value(text):
    text = re.sub(r"\s", "", text)
    if text.endswith(("%", "％")):
        return Decimal(text[:-1]) / 100
    if "分之" in text and not text.startswith("百分之"):
        denominator, numerator = (_chinese_integer(x) for x in text.split("分之"))
        return Decimal(numerator) / denominator if denominator and numerator is not None else None
    number = text[3:] if text.startswith("百分之") else text[:-1]
    value = Decimal(number) if re.fullmatch(r"\d+(?:\.\d+)?", number) else _chinese_integer(number)
    return Decimal(value) / (100 if text.startswith("百分之") else 10) if value is not None else None


def _quantified_sample_rates(text):
    """限定为数量、样本群体和检出结果同时出现，不拦无量化的风险说明。"""
    results = []
    for sentence in re.split(r"[。；;\n]", text):
        for clause in re.split(r"[，,]", sentence):
            for match in re.finditer(_RATE_QUANTITY, clause):
                tail = clause[match.end():]
                if re.match(r"\s*(?:分界线|阈值|置信区间|线)", tail):
                    continue
                neighborhood = clause[max(0, match.start() - 18):]
                if not re.search(r"样本|图像|图片|换脸图|伪造图|真实图", neighborhood):
                    continue
                metric = None
                for name, pattern in (("fnr", r"漏检|漏报|漏判|逃脱检测|(?:无法|不能|未能|未被)[^，。；\n]{0,8}(?:识别|检出|捕获)"),
                                      ("fpr", r"误报|误判为(?:生成|伪造|换脸)"),
                                      ("not_enter_review", r"进不了复核|(?:未|无法|不能)[^，。；\n]{0,5}进入复核"),
                                      ("recall", r"成功(?:识别|检出)|被(?:有效)?(?:识别|检出)|有效捕获")):
                    if re.search(pattern, tail):
                        metric = name
                        break
                if metric and not _execution_claim_negated(clause, match.start(), match.end()):
                    value = _rate_quantity_value(match.group())
                    if value is not None:
                        results.append((metric, value, sentence))
    return results


def _quantified_performance_errors(text, evidence_view, *, path):
    errors = []
    items = _items(evidence_view)
    for metric, value, sentence in _quantified_sample_rates(text):
        refs = _performance_reference_ids(sentence, evidence_view)
        supported = bool(refs)
        for identifier in refs:
            item = items[identifier]
            data = item.get("data", item)
            allowed = _performance_values({"items": [item]}).get(metric, set())
            # 中文分数只作确定的分子/分母值；不新增百分比误差窗口。
            numeric = value in allowed or Decimal(str(float(value))) in allowed
            # 原可靠性原句例外必须同证据、同结果类型、同数值；进不了复核不能改称漏检。
            quoted = any(kind == metric and amount == value
                         for kind, amount, _ in _quantified_sample_rates(str(data.get("known_reliability", ""))))
            supported = supported and (numeric or quoted)
        if not supported:
            errors.append(f"{path} 的具体样本检出/漏检比例没有对应受信指标或同义原文来源")
    return errors


def _mask_grounded_metric_fractions(text, evidence_view):
    """只接纳同证据同统计行的分子/分母；不把比值泛化为其他性能指标。"""
    items = _items(evidence_view)
    pattern = r"(?P<metric>检出率|召回率|漏检率|误报率)\s*(?:为|是|[:：])?\s*(?P<n>\d+)\s*/\s*(?P<d>\d+)"
    def replace(match):
        left = max(text.rfind(mark, 0, match.start()) for mark in ("。", "；", ";", "\n")) + 1
        right = min([p for mark in ("。", "；", ";", "\n") if (p := text.find(mark, match.end())) >= 0] or [len(text)])
        sentence = text[left:right]
        refs = _performance_reference_ids(sentence, evidence_view)
        metric = {"检出率": "recall", "召回率": "recall", "漏检率": "fnr", "误报率": "fpr"}[match.group("metric")]
        numerator, denominator = int(match.group("n")), int(match.group("d"))
        if not refs or denominator <= 0:
            return match.group()
        def rows(value):
            if isinstance(value, Mapping):
                if {"numerator", "denominator", "rate"} <= value.keys():
                    yield value
                for key, child in value.items():
                    if key not in _UNTRUSTED_FIELDS:
                        yield from rows(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    yield from rows(child)
        prefix = text[max(left, match.start() - 8):match.start()]
        qualifiers = [word for word in ("原图", "退化图", "无害副本", "模拟美颜", "难负") if word in prefix]
        for identifier in refs:
            data = items[identifier].get("data", items[identifier])
            candidates = [row for field in ("statistics", "reliability_statistics", "reliability") for row in rows(data.get(field, {}))]
            valid = False
            for row in candidates:
                named = str(row.get("metric", ""))
                title = str(row.get("title", ""))
                kind = named in _NAMED_RATE_METRICS.get(metric, set()) or re.search(_PERFORMANCE_METRICS[metric][1], named, re.I)
                if (kind and row.get("numerator") == numerator and row.get("denominator") == denominator
                        and isinstance(row.get("rate"), (int, float)) and not isinstance(row.get("rate"), bool)
                        and float(row["rate"]) == numerator / denominator
                        and all(word in title for word in qualifiers)):
                    valid = True
                    break
            if not valid:
                return match.group()
        return "对应指标按受信分子分母引用"
    return re.sub(pattern, replace, text)


def _execution_claim_negated(text, start, end):
    """否定只支配当前分句及其并列补语，转折后的实际断言另行检查。"""
    boundary = max(text.rfind(mark, 0, start) for mark in ("，", ",", "。", "；", ";", "\n")) + 1
    prefix = re.split(r"但是|然而|不过|而是|事实上|实际上|实际|但", text[boundary:start])[-1]
    if re.search(r"(?:并非|不是|并不是|不能说|不能认为)(?:不能|不可|不应|不得|无法)", prefix):
        return False
    if _negated_or_hypothetical(text, start, start):
        return True
    if (re.search(r"(?:不能|不可|不应|不得|无法)(?:据此|由此|直接)?\s*$", prefix)
            and re.match(r"推算|换算|推断|排除|断言|认定", text[start:end])):
        return True
    return bool(re.search(r"(?:不能|不可|不应|不得|无法)(?:直接)?(?:表述为|称为?|说成|写成|解释为|报告|得出|认定|声称|断言|认为)[^，。；;\n]{0,48}$", prefix))


def _specific_status_fact_errors(text, evidence_view, *, path):
    errors = []
    items = _items(evidence_view)
    face = items.get("E04", {})
    face = face.get("data", face)
    if face.get("status") == "no_face":
        pattern = r"(?:人脸(?:检测|分支|模块)|E04)(?:模块)?(?:已|已经)?(?:执行|运行)?(?:失败|出错|异常)"
        for match in re.finditer(pattern, text):
            if not _execution_claim_negated(text, match.start(), match.end()):
                errors.append(f"{path} 将E04的no_face写成检测执行失败")
                break
    elif face.get("status") in {"failed", "error"}:
        for match in re.finditer(r"人脸检测(?:未检出|未发现|没有检测到)(?:有效)?人脸", text):
            if not _execution_claim_negated(text, match.start(), match.end()):
                errors.append(f"{path} 将E04执行失败写成未检出人脸")
                break
    process = items.get("E10", {})
    process = process.get("data", process)
    statuses = process.get("module_status", {})
    if isinstance(statuses, Mapping) and any(isinstance(s, str) and s not in ("", "unrecorded") for s in statuses.values()):
        patterns = (r"未记录分析时间(?:与|和|及)模块状态", r"模块状态(?:均|都|全部)?(?:未记录|没有记录)",
                    r"(?:没有|未有|无)任何模块状态记录", r"未记录任何模块状态")
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                prefix = text[max(0, match.start() - 8):match.start()]
                if (match.group() == "模块状态未记录" and process.get("status") == "unrecorded"
                        and re.search(r"处理过程$", prefix)):
                    continue
                if re.search(r"部分|有些|个别", prefix) or _execution_claim_negated(text, match.start(), match.end()):
                    continue
                errors.append(f"{path} 在E10已有子模块状态时声称模块状态全无记录")
                break
    return errors


def _performance_errors(value, evidence_view, *, path):
    original_text = _material_quote_mask(value, evidence_view)
    text = _mask_grounded_metric_fractions(original_text, evidence_view)
    allowed = _performance_values(evidence_view)
    errors = []
    items = _items(evidence_view)
    per_item = {identifier: _performance_values({"items": [item]}) for identifier, item in items.items()}
    for name, (pattern, _) in _PERFORMANCE_METRICS.items():
        for match in _metric_occurrences(text, pattern):
            start = max(text.rfind(mark, 0, match.start()) for mark in ("，", ",", "。", "；", ";", "\n", "、")) + 1
            endings = [p for mark in ("，", ",", "。", "；", ";", "\n", "、") if (p := text.find(mark, match.end())) >= 0]
            clause = text[start:min(endings) if endings else len(text)]
            referenced = set(re.findall(r"(?<![A-Za-z0-9_])E\d{2}(?!\d)", clause)) & set(items)
            if not referenced:
                for identifier, subject in (("E03", "通用|融合"), ("E04", "人脸|换脸"), ("E05", "整图"), ("E08", "配对")):
                    if identifier in items and re.search(subject, clause):
                        referenced.add(identifier)
            sources = [per_item[identifier][name] for identifier in referenced] or [allowed[name]]
            if any(_unmatched_numbers(match.group(1), source) for source in sources):
                errors.append(f"{path} 的性能数字没有对应受信统计指标")
    for sentence in re.split(r"[。；;\n]", text):
        if not re.search(r"(?<![A-Za-z])AUC(?![A-Za-z])", sentence, re.I):
            continue
        # AUC与漏检可以作定性讨论，不能从AUC推出具体检出/漏检比例。
        inference = re.search(r"意味着|相当于|等同于|也就是|即表示|因此|所以|推算(?:出|为)?|换算(?:成|为)", sentence)
        if not inference or _negated_or_hypothetical(sentence, inference.start(), inference.end()):
            continue
        if re.search(r"(?:不|不能|不可|不应|不得|并非)\s*$", sentence[:inference.start()]):
            continue
        tail = sentence[inference.end():]
        if re.match(r"\s*(?:不能|不可|不应|不得|无法).{0,16}(?:推算|换算|断言|认定|解释|说)", tail):
            continue
        antecedent = re.split(r"[，,]", sentence[:inference.start()].rstrip("，, "))[-1]
        if (not re.search(r"(?<![A-Za-z])AUC(?![A-Za-z])", antecedent, re.I)
                and any(_metric_occurrences(antecedent, pattern)
                        for name, (pattern, _) in _PERFORMANCE_METRICS.items() if name != "auc")):
            # 相邻分句若明确引用另一实测指标，不能仅因同句提到AUC而改判推导来源。
            continue
        quantity = r"\d+(?:\.\d+)?\s*%|百分之[零一二两三四五六七八九十百\d.]+|[一二两三四五六七八九十百]+分之[一二两三四五六七八九十百]+|[一二两三四五六七八九十]+成"
        if re.search(quantity, tail) and re.search(r"漏检|漏报|漏判|未.{0,3}检出|检出率|召回率|敏感度|假阴性|误报率|准确率", tail):
            errors.append(f"{path} 将AUC换算为未经提供的具体性能比例")
    errors.extend(_quantified_performance_errors(original_text, evidence_view, path=path))
    return errors


_UNFINISHED_MODULE_STATUSES = {"skipped", "unrecorded", "unavailable", "failed", "error", "not_applicable", "not_run"}
_MODULE_TEXT = {
    "E01": ("provenance", r"文件标识|生成声明|元数据|C2PA|签名|来源标识"),
    "E02": ("visible_text", r"画面文字|文字识别|OCR|水印|可见字样|显性标记"),
    "E06": ("source", r"素材|来源候选|素材库|检索"),
    "E07": ("identity", r"身份|人员候选|身份库"),
}


def _execution_fact_errors(value, evidence_view, *, path, item_id=None):
    """检查已证实的执行状态和人工主体错写；不把视觉未见等同模块检索结论。"""
    if isinstance(value, Mapping) and "evidence_assessment" in value:
        # 逐证据观测在调用方按E编号核对；视觉观察保留其字段归属。
        return [error for key, child in value.items() if key != "evidence_assessment"
                for error in _execution_fact_errors(child, evidence_view,
                    path="visual_observations" if key == "visual_observations" else path)]
    text = _material_quote_mask(value, evidence_view)
    errors = []
    # 当前LLM证据包不携带人工审阅凭证，人工复核记录由独立本地章节展示。
    human_claim = r"经(?:过)?人工(?:观察|核查|核验|复核|审阅|确认)|(?:人工|审核人员|办案人员)(?:已|已经|完成了?|观察(?:发现|显示)|复核(?:发现|确认))"
    for match in re.finditer(human_claim, text):
        # 明确的未来前提不是已完成审阅；后续肯定句仍分别检查。
        if (re.search(r"(?:需|须|必须|应当|有待)(?:先)?$", text[:match.start()])
                and re.match(r"后(?:方可|才可|才能)", text[match.end():])):
            continue
        if not _negated_or_hypothetical(text, match.start(), match.end(), human=True):
            errors.append(f"{path} 无人工审阅凭证却声称已由人工观察或核查")
            break
    for identifier, (section, subject) in _MODULE_TEXT.items():
        item = _items(evidence_view).get(identifier, {})
        data = item.get("data", item)
        if not isinstance(data, Mapping) or data.get("status") not in _UNFINISHED_MODULE_STATUSES:
            continue
        for sentence in re.split(r"[。；;\n，,、]", text):
            if path != section and item_id != identifier and identifier not in sentence and not re.search(subject, sentence, re.I):
                continue
            # 假设或建议不表示本次已执行；程序已跳过也不妨碍模型描述上传图中可见内容。
            if (evidence_view.get("uploaded_figure_ids") and
                    (path == "visual_observations" or re.search(r"(?:图中|上传图|视觉观察|可见画面).{0,8}(?:未见|没有看到)", sentence))
                    and not re.search(r"已执行|已完成|排除", sentence)):
                continue
            patterns = [r"(?:模块|核查|检查|检测|识别|检索|查询|比对)(?:已|已经)?(?:执行|完成|成功)",
                        r"(?:已|已经)(?:完成|执行|开展).{0,12}(?:核查|检查|检测|识别|检索|查询|比对)",
                        r"(?:未检出|未识别到|未检测到|未检索到|未命中|未找到|未发现|无(?:可信|可靠)?候选|无匹配)[^，。；\n]{0,20}",
                        r"排除(?:了)?(?:显性标记|生成标识|生成声明|水印).{0,8}(?:线索|可能)?"]
            if identifier in {"E01", "E02"}:
                patterns.append(r"(?:未见|没有).{0,10}(?:标识|声明|水印|字样|签名)")
            for pattern in patterns:
                for match in re.finditer(pattern, sentence):
                    if re.match(r"(?:无(?:可信|可靠)?候选|无匹配|未找到|未发现)(?:的)?(?:记录|结果记录|执行记录)", match.group()):
                        # 缺失记录是执行缺口，不能误作已运行后的阴性结论。
                        continue
                    if re.match(r"(?:无候选|无匹配)(?:的)?(?:结果|输出)(?!为|是)", match.group()):
                        # 没有返回输出不等于检索已经执行；不能推广为“无可信候选”。
                        continue
                    face = _items(evidence_view).get("E04", {})
                    face = face.get("data", face)
                    if (identifier != "E04" and face.get("status") == "no_face"
                            and re.match(r"(?:未检出|未发现|未检测到)(?:有效|可用)?人脸(?:区域)?(?=\[E04\]|且|并|而|导致|$)", sentence[match.start():])):
                        # 此处明确是已有E04无人脸结果，不移作E06/E07的执行结论。
                        continue
                    if not _execution_claim_negated(sentence, match.start(), match.end()):
                        errors.append(f"{path} 将{identifier}的{data['status']}状态写成已执行或检出阴性")
                        break
                else:
                    continue
                break
    errors.extend(_specific_status_fact_errors(text, evidence_view, path=path))
    return errors


def _material_strings(evidence_view):
    values = set()
    def collect(value, material=False):
        if isinstance(value, Mapping):
            material = material or value.get("kind") == "material_content"
            for key, item in value.items():
                collect(item, material or key in {"material_content", "metadata", "hits", "aigc_labels"})
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item, material)
        elif material and isinstance(value, str):
            values.add(value)
    collect(evidence_view)
    return values


def _material_quote_mask(value, evidence_view):
    """只允许明确归属材料的逐字引文；引文数字不能移作性能断言。"""
    sources = _material_strings(evidence_view)
    def mask(text):
        pattern = r"(?:识别文字|画面文字|元数据内容|材料原文)(?:为|是|记载为|包含)?\s*[:：]?\s*[“\"]([^”\"]+)[”\"]"
        return re.sub(pattern, lambda match: "材料逐字引文" if match.group(1) in sources else match.group(0), text)
    return "\n".join(mask(text) for text in _strings(value))


def _explicit_numeric_reference_mask(text, evidence_view):
    """仅豁免明确引用的实际约束值与OCR程序置信度，不扩大全局数字白名单。"""
    distance = evidence_view.get("constraints", {}).get("max_band_distance")
    if isinstance(distance, int) and not isinstance(distance, bool):
        # 必须读完数值，不能把百分号或数量单位前面的前缀当作整数档距。
        pattern = (r"(?P<prefix>(?<![A-Za-z0-9_])max_band_distance\s*(?:为|是|[:=：])\s*)"
                   r"(?P<number>[-+]?\d+(?:\.\d+)?)(?=$|[\s，。；;、）)\]}]|且|并|但|所以|档(?:$|[\s，。；;、）)\]}]))")
        text = re.sub(pattern, lambda m: m["prefix"] + "已记录约束值" if Decimal(m["number"]) == Decimal(distance) else m[0], text)
    item = _items(evidence_view).get("E02", {})
    data = item.get("data", item)
    hits = data.get("hits", {}) if isinstance(data, Mapping) else {}
    records = hits.get("value", []) if isinstance(hits, Mapping) and hits.get("kind") == "material_content" else []
    confidences = {Decimal(str(row["confidence"])) for row in records if isinstance(row, Mapping)
                   and isinstance(row.get("confidence"), (int, float)) and not isinstance(row.get("confidence"), bool)
                   and 0 <= row["confidence"] <= 1 and math.isfinite(row["confidence"])}
    if not confidences:
        return text
    pattern = (r"(?P<prefix>(?:OCR|文字识别|画面文字|字样|识别文字)[^。；\n]{0,80}?"
               r"(?:识别)?置信度\s*(?:为|是|[:：])?\s*)(?P<number>" + _METRIC_NUMBER
               + r")(?=$|[\s，。；;、）)\]}]|但|且|并|即|可|不|位于|用于|是|为)")
    def confidence_reference(match):
        if _unmatched_numbers(match["number"], confidences):
            return match[0]
        left = max(text.rfind(mark, 0, match.start()) for mark in ("。", "；", "\n")) + 1
        ends = [p for mark in ("。", "；", "\n") if (p := text.find(mark, match.end())) >= 0]
        clause = text[left:min(ends) if ends else len(text)]
        # 只接受识别质量语境；同句混入概率/系统成绩时保守拒绝，另句解释边界不受影响。
        metric = r"(?:系统|模型|本系统)?(?:准确率|正确率|召回率|检出率|误报率|误判率)|(?:AIGC|生成|伪造|换脸|真实)[^，。；\n]{0,6}(?:概率|置信度)|AUC"
        safe_boundary = re.sub(r"(?:不是|不等于|不代表|并非|不能作为|不能解释为)\s*(?:" + metric + r")", "", clause, flags=re.I)
        if re.search(metric, safe_boundary, re.I):
            return match[0]
        return match["prefix"] + "已记录OCR置信度"
    return re.sub(pattern, confidence_reference, text, flags=re.I)


def _common_errors(value, allowed, *, path="输出", evidence_view=None):
    errors = []
    text = "\n".join(_strings(value))
    for phrase in BAN_PHRASES:
        if phrase in text:
            errors.append(f"{path} 含禁用说法：{phrase}")
    number_text = _material_quote_mask(value, evidence_view) if evidence_view is not None else value
    if evidence_view is not None:
        number_text = _explicit_numeric_reference_mask(number_text, evidence_view)
        errors.extend(_execution_fact_errors(value, evidence_view, path=path))
    if _unmatched_numbers(number_text, allowed):
        errors.append(f"{path} 含证据中不存在的数字")
    try:
        assert_upload_safe(value)
    except LLMClientError:
        errors.append(f"{path} 含本机路径或凭据字段")
    return errors


def _items(evidence_view):
    return {item["item_id"]: item for item in evidence_view.get("items", []) if isinstance(item, Mapping) and isinstance(item.get("item_id"), str)}


def validate_judge(value: Any, evidence_view: Mapping[str, Any], *, uploaded_figure_ids=()) -> dict[str, Any]:
    evidence_view = {**evidence_view, "uploaded_figure_ids": list(uploaded_figure_ids)}
    errors = _schema_errors(value, JUDGE_SCHEMA)
    if errors:
        return {"valid": False, "errors": errors}
    rule = evidence_view.get("rule_band", "undetermined")
    band = value["final_band"]
    if rule not in BANDS:
        errors.append("无法完成判断时不允许大模型定档")
    else:
        distance = list(BANDS).index(band) - list(BANDS).index(rule)
        if abs(distance) > 1:
            errors.append("最终档位与规则档位相差超过一档")
        expected = "same" if distance == 0 else "raised" if distance > 0 else "lowered"
        if value["relation_to_rule"] != expected:
            errors.append("与规则档位的关系不一致")
    if not BANDS[band][0] <= value["score"] <= BANDS[band][1]:
        errors.append("档位与评分区间不一致")
    constraints = evidence_view.get("constraints", {})
    floor = constraints.get("floor_band", "none")
    if floor in BANDS and list(BANDS).index(band) < list(BANDS).index(floor):
        errors.append("最终档位低于证据要求的最低档位")
    if band == "obvious" and constraints.get("allow_obvious") is not True:
        errors.append("没有允许明显档位的依据")
    allowed = _evidence_numbers(evidence_view) | {Decimal(value["score"])}
    remaining = {key: item for key, item in value.items() if key != "score_reason"}
    errors.extend(_common_errors(remaining, allowed, evidence_view=evidence_view))
    errors.extend(_common_errors(value["score_reason"], allowed | {Decimal(n) for n in SCORE_POLICY_BOUNDARIES},
                                 path="评分理由", evidence_view=evidence_view))
    errors.extend(_performance_errors(value, evidence_view, path="研判"))
    errors.extend(_entity_errors(value, evidence_view, path="研判"))
    items = _items(evidence_view)
    uploaded = set(uploaded_figure_ids)
    assessments = {}
    for item in value["evidence_assessment"]:
        item_id = item["item_id"]
        if item_id in assessments:
            errors.append("证据评估编号重复")
        assessments[item_id] = item
        if item_id in items:
            local_view = {"items": [items[item_id]]}
            number_text = _explicit_numeric_reference_mask(_material_quote_mask(item["observation"], local_view), local_view)
            if _unmatched_numbers(number_text, _item_numbers(items[item_id])):
                errors.append(f"{item_id} 观察数字与该证据项不一致")
            errors.extend(_performance_errors({key: item[key] for key in ("observation", "reason")},
                                              {"items": [items[item_id]]}, path=item_id))
            errors.extend(_execution_fact_errors({key: item[key] for key in ("observation", "reason")},
                                                 evidence_view, path=item_id, item_id=item_id))
        elif item_id in uploaded:
            errors.extend(_execution_fact_errors({key: item[key] for key in ("observation", "reason")},
                                                 evidence_view, path="visual_observations"))
            if item["weight"] == "high":
                errors.append("视觉观察分量不能为高")
        else:
            errors.append("证据评估引用不存在或未上传的编号")
    visual_ids = set()
    for item in value["visual_observations"]:
        if item["figure_id"] not in uploaded:
            errors.append("视觉观察引用未实际上传的图件")
        if item["figure_id"] in visual_ids:
            errors.append("视觉观察图件编号重复")
        visual_ids.add(item["figure_id"])
    basis_ids = value["adjustment_basis_ids"]
    if any(item_id not in items and item_id not in uploaded for item_id in basis_ids):
        errors.append("调整依据编号不存在")
    if band != rule:
        if not value["adjustment_reason"].strip():
            errors.append("调整档位缺少理由")
        if not any(item_id in items and item_id.startswith("E") and assessments.get(item_id, {}).get("weight") in ("medium", "high") for item_id in basis_ids):
            errors.append("调整档位缺少中高分量的非图件证据")
    return {"valid": not errors, "errors": list(dict.fromkeys(errors))}


_FILE_NAME = re.compile(r"[^\s，。；：、（）()\[\]<>\"']+\.(?:jpe?g|png|webp|bmp|mp4|mov|avi|pdf|json|txt)\b", re.I)
_PERSON = re.compile(r"(?:姓名|人员姓名|人员编号|身份候选|候选姓名)\s*[:：为是]\s*[\"'“]?([\w\u3400-\u9fff\-]{2,24})")
_PERSON_SUBJECT = re.compile(r"(?:^|[，。；！？\n])\s*([A-Za-z\u3400-\u9fff][A-Za-z\u3400-\u9fff·\-]{1,23})(?=出现在|出镜|在画面中|在视频中|在图片中|的人脸|的身份)")
_GENERIC_SUBJECTS = {"人物", "人员", "有人", "某人", "同一人", "不同人", "查询人物", "画面人物", "视频人物", "候选人物", "图中人物", "人脸", "画面", "图像", "本图", "查询图", "候选图", "素材", "目标", "参考人物"}


def _entity_errors(value, evidence_view, *, path):
    """实体检查为启发式防线；自由中文不承诺穷尽人名识别。"""
    available_names, available_files, available_aliases = set(), set(), set()
    def collect(item):
        if isinstance(item, Mapping):
            if item.get("kind") == "material_content":
                return
            for key, child in item.items():
                if key in _UNTRUSTED_FIELDS or key == "hits":
                    continue
                if isinstance(child, str):
                    if key in {"identity_id", "identity_name", "person_id", "person_name"}:
                        available_names.add(child)
                    if key in {"file_name", "original_file_name", "candidate_name", "reference_name"}:
                        available_files.add(child)
                    if key in {"identity_id", "identity_name", "person_id", "person_name", "file_name", "original_file_name", "candidate_name", "reference_name", "task_id", "library_name"}:
                        available_aliases.update(_ALIAS.findall(child))
                collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)
    collect(evidence_view)
    joined = _material_quote_mask(value, evidence_view)
    identity = _items(evidence_view).get("E07", {})
    identity = identity.get("data", identity)
    state = identity.get("open_set_state") if isinstance(identity, Mapping) else None
    if state in {"high_confidence", "low_confidence", "uncertain", "no_reliable_candidate", "unavailable", "no_face"}:
        # 只豁免当前E07的实际状态引用；姓名/人员编号语境和未知姓名继续检查。
        pattern = (r"身份候选(?:状态)?\s*(?:为|是|[:：])\s*(?P<state_quote>[\"']?)" + re.escape(state)
                   + r"(?P=state_quote)(?=$|[\s，。；;、（）()\]}])")
        joined = re.sub(pattern, "身份候选状态与记录一致", joined)
    errors = []
    if not set(_ALIAS.findall(joined)) <= available_aliases:
        errors.append(f"{path} 引用证据之外的材料代号")
    files_masked = joined
    for filename in sorted(available_files, key=len, reverse=True):
        files_masked = files_masked.replace(filename, "证据文件")
    if _FILE_NAME.search(files_masked):
        errors.append(f"{path} 含证据之外的文件名")
    people = set(_PERSON.findall(joined)) | set(_PERSON_SUBJECT.findall(joined))
    if any(person not in available_names and person not in _GENERIC_SUBJECTS for person in people):
        errors.append(f"{path} 含证据之外的人员标识")
    return errors


def _review_action_in_clause(joined, match, labels, selected_band, path):
    # 仅识别有明确动作对象的复核建议；不是通用中文语义判别。
    if match.group(0) != "建议人工复核":
        return False
    boundary = r"[，,；;。！？!?\n]"
    left = re.split(boundary, joined[:match.start()])
    prefix = left[-1].strip()
    suffix = re.split(boundary, joined[match.end():], maxsplit=1)[0].strip()
    # 结论锚点在当前分句仍然优先；不能借 next_steps 或动作句绕过改档检查。
    if re.search(r"最终|档位|结论|研判|定档|归[为入]|调整|改[为成]|升[为至]|降[为至]|判[为定]", prefix):
        return False
    previous = next((part.strip() for part in reversed(left[:-1]) if part.strip()), "")
    if re.search(r"(?:最终|档位|结论|研判|定档).*(?:为|是|[:：])$", previous):
        return False
    if not re.fullmatch(r"(?:后续|仍|进一步|因此|为此|故|同时|另外|并|请|需|应|可)?", prefix):
        return False
    action_object = re.match(r"(?:原始(?:材料|文件|视频|图像)|原图|材料|图像|图片|视频|证据|来源|人物|人脸|细节|元数据|哈希|结果|相关材料)", suffix)
    if action_object and not re.search(r"最终|档位|结论|研判|定档|调整|改[为成]", suffix):
        return True
    if suffix:
        return False
    if path == "next_steps":
        return True
    # 无动作对象的短句仅在同段已明确给出当前正确结论时接纳。
    label = labels.get(selected_band, "")
    compact_before = re.sub(r"\s+", "", joined[:match.start()])
    compact_label = re.escape(re.sub(r"\s+", "", label))
    return bool(compact_label and re.search(r"(?:最终(?:结论|档位)?|规则定档|定档|研判结论|结论)(?:为|是|[:：])?" + compact_label, compact_before))


def _decision_text_errors(value, labels, selected_band, score, *, path, rule_band=None):
    joined = "\n".join(_strings(value))
    errors = []
    # 最长显示名优先，避免“未见明显…”中的子串被误认成“明显…”。
    label_keys = {label: key for key, label in labels.items() if label}
    expression = "|".join(re.escape(label) for label in sorted(label_keys, key=len, reverse=True))
    for match in re.finditer(expression, joined) if expression else ():
        other = label_keys[match.group(0)]
        if other != selected_band:
            if other == "review" and _review_action_in_clause(joined, match, labels, selected_band, path):
                continue
            prefix = joined[max(0, match.start() - 40):match.start()]
            grounded_rule = other == rule_band and re.search(r"规则(?:参考)?(?:档位|定档|结论|判断)[^，。；\n]{0,12}$", prefix)
            if not grounded_rule:
                errors.append(f"{path} 出现其他最终档位名称")
    for match in re.finditer(r"(?:可疑度评分|最终评分|评分)(?:[^\d\n，,。；;！？!?\.]{0,8}|\s*(?:为|是|[:：])[\s，,；;]*)(\d+(?:\.\d+)?)", joined):
        if score is None or Decimal(match.group(1)) != Decimal(str(score)):
            errors.append(f"{path} 评分与研判结论不一致")
    return errors


def validate_writer(value: Any, evidence_view: Mapping[str, Any], final_decision: Mapping[str, Any], *,
                    uploaded_figure_ids=(), band_labels=None) -> dict[str, Any]:
    evidence_view = {**evidence_view, "uploaded_figure_ids": list(uploaded_figure_ids)}
    errors = _schema_errors(value, WRITER_SCHEMA)
    invalid = set()
    normalized = deepcopy(value) if isinstance(value, dict) else {}
    if not isinstance(value, dict):
        return {"valid": False, "errors": errors, "invalid_sections": ["all"], "normalized": {}}
    if errors:
        for error in errors:
            path = error.split(" ", 1)[0]
            parts = path.removeprefix("$.").split(".")
            part = parts[0].split("[")[0]
            invalid.add(parts[1].split("[")[0] if part == "process" and len(parts) > 1 and parts[1].split("[")[0] in PROCESS_FIELDS else part)
        # 单段结构错误只使该段回退；其余段仍须逐段经过全部事实校验。
    labels = dict(DEFAULT_LLM_CONFIG["band_labels"])
    labels.update(band_labels or {})
    sent_decision = {key: final_decision[key] for key in ("final_band", "score", "source", "reason", "judgment") if key in final_decision}
    # 已有研判的自由文字不能扩大数值权限，只加入最终结构化评分。
    allowed = _evidence_numbers(evidence_view) | _numbers(sent_decision.get("score"))
    process = value.get("process") if isinstance(value.get("process"), dict) else {}
    sections = {"summary": value.get("summary"), **{key: process.get(key) for key in PROCESS_FIELDS},
                "synthesis": value.get("synthesis"), "uncertainties": value.get("uncertainties"), "next_steps": value.get("next_steps")}
    selected_band = final_decision.get("final_band", "undetermined")
    score = final_decision.get("score")
    for name, text in sections.items():
        expected_type = isinstance(text, list) and all(isinstance(item, str) for item in text) if name in ("uncertainties", "next_steps") else isinstance(text, str)
        if not expected_type:
            invalid.add(name)
            continue
        section_errors = _common_errors(text, allowed, path=name, evidence_view=evidence_view)
        section_errors.extend(_entity_errors(text, evidence_view, path=name))
        section_errors.extend(_performance_errors(text, evidence_view, path=name))
        section_errors.extend(_decision_text_errors(text, labels, selected_band, score, path=name, rule_band=evidence_view.get("rule_band")))
        joined = "\n".join(_strings(text))
        if name == "synthesis" and len(joined) < 800:
            section_errors.append("synthesis 少于要求的八百字")
        if name in ("uncertainties", "next_steps") and not 3 <= len(text) <= 6:
            section_errors.append(f"{name} 条目数须为三至六条")
        if section_errors:
            errors.extend(section_errors)
            invalid.add(name)
    uploaded = set(uploaded_figure_ids)
    figure_ids = set()
    figures = value.get("figures") if isinstance(value.get("figures"), list) else []
    for index, figure in enumerate(figures):
        if _schema_errors(figure, WRITER_SCHEMA["properties"]["figures"]["items"]):
            invalid.add("figures")
            continue
        figure_id = figure["figure_id"]
        figure_text = {key: figure[key] for key in ("caption", "reading_guide")}
        figure_errors = _common_errors(figure_text, allowed, path="图件说明", evidence_view=evidence_view)
        figure_errors.extend(_entity_errors(figure_text, evidence_view, path="图件说明"))
        figure_errors.extend(_performance_errors(figure_text, evidence_view, path="图件说明"))
        figure_errors.extend(_decision_text_errors(figure_text, labels, selected_band, score, path="图件说明", rule_band=evidence_view.get("rule_band")))
        if figure_id not in uploaded:
            figure_errors.append("撰写图件未实际上传")
        if figure_id in figure_ids:
            figure_errors.append("撰写图件重复放置")
        figure_ids.add(figure_id)
        section_text = sections[figure["section"]]
        paragraphs = [p for p in section_text.split("\n") if p.strip()] if isinstance(section_text, str) else []
        normalized["figures"][index]["after_paragraph"] = min(figure["after_paragraph"], max(1, len(paragraphs)))
        if figure_errors:
            errors.extend(figure_errors)
            invalid.add("figures")
    for name, text in sections.items():
        references = set(re.findall(r"\[(F\d+)\]", "\n".join(_strings(text))))
        if not references <= figure_ids:
            errors.append(f"{name} 的正文图号没有对应放置记录")
            invalid.add(name)
    return {"valid": not errors, "errors": list(dict.fromkeys(errors)), "invalid_sections": sorted(invalid), "normalized": normalized}
