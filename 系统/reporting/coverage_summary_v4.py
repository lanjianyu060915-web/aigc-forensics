"""Read-only coverage wording shared by local reports and the UI."""
from collections.abc import Mapping


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _count(value):
    return str(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else "未记录"


def face_coverage_note(audit):
    audit = _mapping(audit)
    coverage = _mapping(audit.get("coverage"))
    if coverage.get("count_status") == "unrecorded" or not coverage:
        return "人脸检查覆盖未记录；不能据旧裁剪数量推定已检查全部人脸。"
    return (f"人脸观察记录：检出 {_count(coverage.get('detected_total'))}，"
            f"裁剪成功 {_count(coverage.get('cropped_count'))}，已评分 {_count(coverage.get('scored_count'))}，"
            f"参与定档 {_count(coverage.get('used_for_band_count'))}，"
            f"超出数量上限 {_count(coverage.get('truncated_count'))}，裁剪失败 {_count(coverage.get('failed_count'))}。"
            "视频按每帧的人脸观察计数，不等于独立人数；未检查部分不推定为低风险。")


def module_face_coverage_rows(audit):
    """Keep each module's denominator and definition of processing explicit."""
    audit = _mapping(audit)
    face = _mapping(audit.get("coverage"))
    identity = _mapping(audit.get("identity_coverage"))
    own_identity = _mapping(identity.get("module_reported_coverage"))
    pair = _mapping(audit.get("pair_coverage"))
    return [
        {"模块": "人脸鉴伪", "检出记录": _count(face.get("detected_total")),
         "处理记录": _count(face.get("scored_count")), "处理口径": "实际评分的人脸观察",
         "失败": f"裁剪 {_count(face.get('failed_count'))} / 模型 {_count(face.get('model_failed_count'))}",
         "数量上限外": _count(face.get("truncated_count"))},
        {"模块": "身份候选", "检出记录": _count(own_identity.get("detected_total")),
         "处理记录": _count(own_identity.get("processed_count")), "处理口径": "模块尝试记录，包含失败",
         "失败": _count(own_identity.get("failed_count")), "数量上限外": _count(own_identity.get("truncated_count"))},
        {"模块": "素材配对", "检出记录": _count(pair.get("module_detected_total")),
         "处理记录": _count(pair.get("processed_bound_count")), "处理口径": "明确对应本表查询脸且完成测量，跨候选去重",
         "失败": f"对应脸 {_count(pair.get('failed_bound_count'))} / 候选 {_count(pair.get('candidate_failed_count'))}",
         "数量上限外": "未记录"},
    ]


def face_binding_coverage_rows(audit):
    """Counts refer to the face-audit observations, never the identity detector's total."""
    audit = _mapping(audit)
    result = []
    for key, label in (("identity_coverage", "身份候选"), ("pair_coverage", "素材配对")):
        coverage = _mapping(audit.get(key))
        result.append({"模块": label, "明确对应": _count(coverage.get("bound_count")),
                       "已处理对应脸": _count(coverage.get("processed_bound_count")),
                       "失败对应脸": _count(coverage.get("failed_bound_count")),
                       "部分完成": _count(coverage.get("partial_count")) if key == "pair_coverage" else "不适用",
                       "歧义": _count(coverage.get("ambiguous_count")), "未对应": _count(coverage.get("unmatched_count")),
                       "未知": _count(coverage.get("unknown_count")), "跳过": _count(coverage.get("skipped_count"))})
    return result


def module_face_coverage_notes(audit):
    audit = _mapping(audit)
    identity = _mapping(audit.get("identity_coverage"))
    own_identity = _mapping(identity.get("module_reported_coverage"))
    pair = _mapping(audit.get("pair_coverage"))
    def correspondence(value):
        return (f"鉴伪查询脸记录 {_count(value.get('query_observation_count'))} 中，明确对应 {_count(value.get('bound_count'))}，"
                f"已处理对应脸 {_count(value.get('processed_bound_count'))}，失败对应脸 {_count(value.get('failed_bound_count'))}，"
                f"歧义 {_count(value.get('ambiguous_count'))}，未对应 {_count(value.get('unmatched_count'))}，"
                f"未知 {_count(value.get('unknown_count'))}，跳过 {_count(value.get('skipped_count'))}。")
    return {
        "identity": (f"身份候选覆盖：模块检出 {_count(own_identity.get('detected_total'))}，"
                     f"尝试 {_count(own_identity.get('processed_count'))}（含失败 {_count(own_identity.get('failed_count'))}），"
                     f"超出数量上限 {_count(own_identity.get('truncated_count'))}。" + correspondence(identity)
                     + "模块检出与鉴伪查询脸的统计范围可能不同，不能据此声称全部人脸完成身份检索。"),
        "source": (f"素材配对覆盖：模块检出 {_count(pair.get('module_detected_total'))}，"
                   f"候选 {_count(pair.get('candidate_count'))}（失败 {_count(pair.get('candidate_failed_count'))}）。"
                   + correspondence(pair) + f"部分完成 {_count(pair.get('partial_count'))} 已包含在已处理对应脸中；"
                   "多候选按同一查询脸去重，失败/歧义/未知的逐候选记录仍保留。"
                   "已处理仅表示有明确对应并完成测量，不证明同一身份或伪造类型；未检查部分不推定为低风险。"),
    }


def append_coverage_notes(content, package):
    face = next((v.get("data", {}) for v in package.get("items", []) if v.get("item_id") == "E04"), {})
    process = next((v.get("data", {}) for v in package.get("items", []) if v.get("item_id") == "E10"), {})
    notes = [face_coverage_note(face.get("face_audit"))]
    labels = {"deepfake": "鉴伪", "identity": "身份", "source": "素材"}
    for module, data in (process.get("module_frame_coverage") or {}).items():
        if not isinstance(data, Mapping) or "requested_count" not in data:
            continue
        notes.append(f"{labels.get(module, module)}模块计划处理 {_count(data.get('requested_count'))} 帧，"
                     f"成功 {_count(data.get('success_count'))} 帧，失败 {_count(data.get('failure_count'))} 帧；"
                     "这是离散帧检查，未证明帧间时段已检查。")
    section = content.setdefault("process", {})
    # This append happens after writer validation; the writer cannot replace
    # measured coverage counts or reinterpret missing observations as success.
    section["detection"] = str(section.get("detection") or "") + "\n" + "\n".join(notes)
    for key, note in module_face_coverage_notes(face.get("face_audit")).items():
        section[key] = str(section.get(key) or "") + "\n" + note
    return content
