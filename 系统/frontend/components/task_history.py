"""任务目录的只读汇总：历史列表、批量结果、证据完整性校验和证据包导出。

这里不依赖 Streamlit，便于单独测试。所有分数都原样取自任务结果文件，不做跨模块合并。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any
import zipfile

from decision import decide_identity_open_set, decide_source_open_set


STATUS_LABELS = {
    "success": "完成",
    "partial_success": "部分完成",
    "error": "失败",
    "skipped": "未运行",
    "running": "运行中",
    "queued": "排队中",
    "warning": "警告",
}
EXECUTION_LABELS = {
    "service": "常驻服务",
    "subprocess": "隔离子进程",
    "preflight": "启动前检查",
    "configuration": "配置检查",
    "disabled": "已关闭",
    "deadline_exhausted": "超出时限",
}
MODULE_LABELS = {"deepfake": "鉴伪", "identity": "身份", "source": "底图"}
SUMMARY_COLUMNS = (
    ("started_at", "开始时间"),
    ("file_name", "文件"),
    ("input_type", "类型"),
    ("status_label", "状态"),
    ("deepfake_label", "鉴伪结论"),
    ("deepfake_score", "鉴伪风险分"),
    ("identity_label", "身份结论"),
    ("identity_top", "身份第一候选"),
    ("identity_score", "身份相似度"),
    ("source_label", "底图结论"),
    ("source_top", "底图第一候选"),
    ("source_score", "底图最终分"),
    ("runtime_seconds", "耗时(秒)"),
    ("task_id", "任务编号"),
)
MAX_BUNDLE_BYTES = 1024 * 1024 * 1024


def status_label(value: Any) -> str:
    text = str(value or "").strip()
    return STATUS_LABELS.get(text, text or "未知")


def execution_label(value: Any) -> str:
    text = str(value or "").strip()
    return EXECUTION_LABELS.get(text, text or "未记录")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _top_candidate(result: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = result.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], Mapping):
        return candidates[0]
    return {}


def _file_name(status: Mapping[str, Any]) -> str:
    input_status = status.get("input")
    input_status = input_status if isinstance(input_status, Mapping) else {}
    original = str(input_status.get("original_file_name") or "").strip()
    if original:
        return Path(original).name
    source = str(input_status.get("source_file") or status.get("query_source") or "").strip()
    return Path(source).name if source else "—"


def _deepfake_summary(result: Mapping[str, Any]) -> tuple[str, float | None]:
    if str(result.get("status") or "") != "success":
        return status_label(result.get("status") or "skipped"), None
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    label = str(details.get("band_label") or "").strip()
    if not label:
        verdict = result.get("is_deepfake")
        label = "疑似AI换脸" if verdict is True else "未见明显换脸特征" if verdict is False else "证据不足"
    return label, _number(result.get("score"))


def _open_set_label(result: Mapping[str, Any], decide) -> str:
    if str(result.get("status") or "") != "success":
        return status_label(result.get("status") or "skipped")
    try:
        decision = decide(result)
    except Exception:  # 判定层失败不影响列表展示。
        return "未判定"
    return str(decision.get("state_label") or "未判定")


def summarize_task(task_dir: str | Path) -> dict[str, Any] | None:
    root = Path(task_dir)
    status = _read_json(root / "task_status.json")
    if not status:
        return None
    deepfake = _read_json(root / "detection_result.json")
    identity = _read_json(root / "identity_result.json")
    source = _read_json(root / "source_result.json")
    input_status = status.get("input") if isinstance(status.get("input"), Mapping) else {}
    deepfake_label, deepfake_score = _deepfake_summary(deepfake)
    top_identity = _top_candidate(identity)
    top_source = _top_candidate(source)
    identity_provenance = identity.get("provenance")
    source_provenance = source.get("provenance")
    return {
        "task_id": str(status.get("task_id") or root.name),
        "task_dir": str(root),
        "started_at": str(status.get("started_at") or "").replace("T", " ")[:19],
        "finished_at": str(status.get("finished_at") or "").replace("T", " ")[:19],
        "file_name": _file_name(status),
        "input_type": "视频" if input_status.get("type") == "video" else "图片",
        "status": str(status.get("status") or "unknown"),
        "status_label": status_label(status.get("status")),
        "selected_modules": list(status.get("selected_modules") or []),
        "modules_label": "、".join(
            MODULE_LABELS.get(str(item), str(item)) for item in status.get("selected_modules") or []
        ),
        "deepfake_label": deepfake_label,
        "deepfake_score": deepfake_score,
        "identity_label": _open_set_label(identity, decide_identity_open_set),
        "identity_top": str(top_identity.get("identity_name") or top_identity.get("identity_id") or "—"),
        "identity_score": _number(top_identity.get("mean_similarity") or top_identity.get("score")),
        "source_label": _open_set_label(source, decide_source_open_set),
        "source_top": Path(str(top_source.get("file_path") or "—")).name,
        "source_score": _number(top_source.get("final_score")),
        "runtime_seconds": _number(status.get("runtime_seconds")),
        "identity_library": (
            str(identity_provenance.get("library_path") or "")
            if isinstance(identity_provenance, Mapping)
            else ""
        ),
        "source_library": (
            str(source_provenance.get("library_path") or "")
            if isinstance(source_provenance, Mapping)
            else ""
        ),
    }


def list_tasks(tasks_root: str | Path, *, limit: int = 200) -> list[dict[str, Any]]:
    """最新的任务在前；未写完 task_status.json 的目录直接跳过。"""

    root = Path(tasks_root)
    try:
        directories = [
            path for path in root.iterdir()
            if path.is_dir() and (path / "task_status.json").is_file()
        ]
    except OSError:
        return []
    directories.sort(
        key=lambda path: (path / "task_status.json").stat().st_mtime_ns, reverse=True
    )
    rows = []
    for directory in directories[:limit]:
        summary = summarize_task(directory)
        if summary is not None:
            rows.append(summary)
    return rows


def display_cell(key: str, value: Any) -> str:
    """界面表格用的文字：缺值显示破折号，分数统一 6 位小数（小分数不能被四舍五入成 0）。"""

    if value is None or value == "":
        return "—"
    if key == "runtime_seconds" and isinstance(value, (int, float)):
        return f"{float(value):.1f}"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def summaries_to_table(summaries: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {label: display_cell(key, summary.get(key)) for key, label in SUMMARY_COLUMNS}
        for summary in summaries
    ]


def summaries_to_csv(summaries: Iterable[Mapping[str, Any]]) -> bytes:
    """带 BOM 的 UTF-8，Excel 直接打开不乱码。"""

    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow([label for _, label in SUMMARY_COLUMNS])
    for summary in summaries:
        writer.writerow(
            ["" if summary.get(key) is None else summary.get(key) for key, _ in SUMMARY_COLUMNS]
        )
    return ("﻿" + stream.getvalue()).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, relative: str) -> Path | None:
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve(strict=False))
    except ValueError:
        return None
    return candidate


def verify_task_artifacts(task_dir: str | Path) -> dict[str, Any]:
    """按任务自带的证据清单重算 SHA-256，发现缺失或被改动的文件。"""

    root = Path(task_dir)
    manifest = _read_json(root / "artifact_manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return {
            "available": False,
            "message": "任务目录没有可校验的证据清单（可能是早期版本生成的任务）。",
            "checked": 0, "matched": 0, "problems": [],
        }
    problems: list[dict[str, str]] = []
    checked = 0
    matched = 0
    for item in artifacts:
        if not isinstance(item, Mapping):
            continue
        relative = str(item.get("path") or "").strip()
        expected = str(item.get("sha256") or "").strip().lower()
        if not relative or not expected:
            continue
        checked += 1
        path = _inside(root, relative)
        if path is None:
            problems.append({"文件": relative, "问题": "路径越出任务目录"})
        elif not path.is_file():
            problems.append({"文件": relative, "问题": "文件缺失"})
        elif _sha256(path) != expected:
            problems.append({"文件": relative, "问题": "内容与清单记录的 SHA-256 不一致"})
        else:
            matched += 1
    return {
        "available": True,
        "message": (
            f"{checked} 个证据文件全部与清单一致。"
            if not problems
            else f"{checked} 个证据文件中有 {len(problems)} 个存在问题。"
        ),
        "checked": checked, "matched": matched, "problems": problems,
    }


def build_evidence_bundle(task_dir: str | Path) -> bytes:
    """把任务目录打成 zip，运行时缓存不进包；超过上限时明确报错而不是截断。"""

    root = Path(task_dir)
    files = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and "runtime" not in path.relative_to(root).parts[:1]
    ]
    if not files:
        raise ValueError("任务目录为空，无法生成证据包。")
    total = sum(path.stat().st_size for path in files)
    if total > MAX_BUNDLE_BYTES:
        raise ValueError(f"证据文件合计 {total / 1048576:.0f} MB，超过证据包上限，请直接复制任务目录。")
    lines = ["文件\tSHA-256\t大小(字节)"]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            archive.write(path, f"{root.name}/{relative}")
            lines.append(f"{relative}\t{_sha256(path)}\t{path.stat().st_size}")
        archive.writestr(f"{root.name}/证据包文件校验清单.tsv", "﻿" + "\n".join(lines) + "\n")
    return buffer.getvalue()


__all__ = [
    "EXECUTION_LABELS",
    "STATUS_LABELS",
    "build_evidence_bundle",
    "display_cell",
    "execution_label",
    "list_tasks",
    "status_label",
    "summaries_to_csv",
    "summaries_to_table",
    "summarize_task",
    "verify_task_artifacts",
]
