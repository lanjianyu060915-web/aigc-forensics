"""人工复核记录只追加，机器报告和原始证据保持不变。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any


REVIEW_FILE = "review_records.jsonl"
REVIEW_CONCLUSIONS = ("同意系统结论", "不同意", "无法判断")
MANUAL_JUDGMENTS = (
    "未作类型判断", "倾向换脸（同一底图，脸被替换）", "倾向同地点的不同人",
    "倾向未改动副本（仅压缩或缩放）", "倾向美颜或局部修改",
    "倾向生成内容（没有对应底图）", "无法判断",
)
REVIEW_NOTICE = "复核人姓名是手填的，没有账号认证；这不是可信时间戳，也不是不可篡改存证。"
_LOCK = threading.RLock()


class ReviewRecordError(ValueError):
    """复核数据无法安全读取或保存。"""


def review_report_mismatch(task_dir: str | Path, record: dict) -> bool:
    """以当前原报告的真实字节检查复核依据，读取失败也不声称一致。"""
    try:
        actual = hashlib.sha256(_task_file(task_dir, "final_report.json").read_bytes()).hexdigest()
    except (OSError, ReviewRecordError):
        return True
    return record.get("report_sha256") != actual


def _task_file(task_dir: str | Path, name: str) -> Path:
    root = Path(task_dir).resolve(strict=True)
    candidate = (root / name).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ReviewRecordError("复核文件路径越出任务目录。") from error
    return candidate


def load_review_records(task_dir: str | Path) -> list[dict[str, Any]]:
    """按保存顺序读取；损坏的历史记录不会被静默忽略。"""
    path = _task_file(task_dir, REVIEW_FILE)
    if not path.exists():
        return []
    with _LOCK:
        try:
            text = path.read_text(encoding="utf-8")
            records = []
            for line in text.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict) or not all(item.get(key) for key in ("reviewer", "conclusion", "opinion", "reviewed_at", "report_sha256")):
                    raise ValueError("复核记录字段不完整")
                if item["conclusion"] not in REVIEW_CONCLUSIONS:
                    raise ValueError("复核结论不在允许范围内")
                _validate_manual_fields(item)
                records.append(item)
            return records
        except (OSError, UnicodeError, ValueError) as error:
            raise ReviewRecordError("复核记录无法读取，请检查记录文件；原报告没有修改。") from error


def _validate_manual_fields(record: dict) -> None:
    if record.get("manual_judgment", MANUAL_JUDGMENTS[0]) not in MANUAL_JUDGMENTS:
        raise ReviewRecordError("请选择有效的人工判断。")
    rank = record.get("basis_candidate_rank")
    if rank is not None and (type(rank) is not int or not 1 <= rank <= 3):
        raise ReviewRecordError("依据候选名次必须为1到3或空。")
    index = record.get("target_face_index")
    if index is not None and (type(index) is not int or index < 0):
        raise ReviewRecordError("请选择有效的人脸编号。")


def review_context(task_dir: str | Path, report: dict) -> tuple[dict, dict]:
    """读取已保存的公开结果，不查询外部素材或重跑任何模型。"""
    pair = report.get("pair_compare") or {}
    identity = ((report.get("modules") or {}).get("identity") or {}).get("result") or {}
    for name, target in (("pair_compare_result.json", pair), ("identity_result.json", identity)):
        if not target:
            path = _task_file(task_dir, name)
            if path.is_file():
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    target.update(value)
    return pair, identity


def append_review_record(task_dir: str | Path, reviewer: str, conclusion: str, opinion: str, *,
                         manual_judgment: str = MANUAL_JUDGMENTS[0],
                         basis_candidate_rank: int | None = None,
                         target_face_index: int | None = None) -> dict[str, Any]:
    """保存一条带原报告哈希的复核意见，不提供覆盖或删除接口。"""
    reviewer, opinion = str(reviewer).strip(), str(opinion).strip()
    if not reviewer or not opinion:
        raise ReviewRecordError("复核人和意见均为必填。")
    if conclusion not in REVIEW_CONCLUSIONS:
        raise ReviewRecordError("请选择有效的复核结论。")
    manual = {"manual_judgment": manual_judgment, "basis_candidate_rank": basis_candidate_rank,
              "target_face_index": target_face_index}
    _validate_manual_fields(manual)
    path = _task_file(task_dir, REVIEW_FILE)
    report_path = _task_file(task_dir, "final_report.json")
    with _LOCK:
        load_review_records(task_dir)
        try:
            raw = report_path.read_bytes()
            report = json.loads(raw)
            if not isinstance(report, dict):
                raise ValueError("原报告格式错误")
            pair, identity = review_context(task_dir, report)
            candidate = next((row for row in pair.get("candidates", []) if row.get("rank") == basis_candidate_rank), None) if basis_candidate_rank is not None else None
            if basis_candidate_rank is not None and candidate is None:
                raise ReviewRecordError("所选素材候选未保存在该任务中。")
            if target_face_index is not None and not any(face.get("face_index") == target_face_index for face in identity.get("faces", [])):
                raise ReviewRecordError("所选人脸未保存在该任务中。")
            record = {
                "reviewer": reviewer, "conclusion": conclusion, "opinion": opinion,
                "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "report_sha256": hashlib.sha256(raw).hexdigest(),
                **manual,
                "basis_candidate_name": candidate.get("candidate_name") if candidate else None,
                "basis_candidate_sha256": candidate.get("candidate_sha256") if candidate else None,
            }
            encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            # 兼容手工归档时末尾换行被移除的合法 JSONL，只追加分隔符。
            prefix = b""
            if path.exists() and path.stat().st_size:
                with path.open("rb") as current:
                    current.seek(-1, os.SEEK_END)
                    if current.read(1) != b"\n":
                        prefix = b"\n"
            with path.open("ab") as stream:
                stream.write(prefix + encoded)
                stream.flush()
                os.fsync(stream.fileno())
            return record
        except (OSError, UnicodeError, ValueError) as error:
            raise ReviewRecordError("复核意见未能保存，请检查任务目录和原报告；原报告没有修改。") from error
