# -*- coding: utf-8 -*-
"""一键自检：用内置示例真实跑一遍全部模块，确认系统可以演示。

通过标准只看结果是否与记录一致，不涉及准确率：
  integrated_query_001.jpg 的身份第一候选应为 person_001，素材第一候选应为 122.jpg，
  整图分支应走 ViT 模型，通用 AIGC 分支、人脸分支和文件标识解析都要完成，
  所有证据文件都能通过 SHA-256 校验。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("FSTS_DATA_ROOT", str(ROOT.parent / "用户数据"))
for import_root in (ROOT, ROOT / "frontend"):
    sys.path.insert(0, str(import_root))

from components.task_history import summarize_task, verify_task_artifacts  # noqa: E402
from orchestrator import load_config, run_task  # noqa: E402


QUERY = ROOT / "examples" / "integrated_query_001.jpg"
EXPECTED_IDENTITY = "person_001"
EXPECTED_SOURCE = "122.jpg"
EXPECTED_BRANCH = "vit_full_image"
# Windows 默认路径上限 260 字符。系统内最长的相对路径 189 字符，证据图约 154 字符。
MAX_ROOT_PATH_LENGTH = 60


def root_path_problem(root: Path) -> str:
    """目录太深时，证据图会写不出来，只留一条警告，这里提前说清楚。"""

    length = len(str(root))
    if length <= MAX_ROOT_PATH_LENGTH:
        return ""
    return (
        f"所在目录路径 {length} 字符，超过 {MAX_ROOT_PATH_LENGTH} 字符，"
        "部分证据图可能写不出来；请把整个目录移到短路径，例如 D:\\aigc-forensics"
    )


def identity_evidence_warnings(task_dir: Path) -> list[str]:
    warnings: list[str] = []
    for path in sorted((task_dir / "visualizations" / "identity").glob("*/identity_evidence.json")):
        evidence = json.loads(path.read_text(encoding="utf-8"))
        warnings.extend(str(item) for item in evidence.get("warnings") or [])
    return warnings


def report(results: list[tuple[str, bool, str]], name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    print(f"[{'通过' if passed else '失败'}] {name}{'：' + detail if detail else ''}", flush=True)


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    config_path = ROOT / "config.json"
    print(f"数据目录：{os.environ['FSTS_DATA_ROOT']}\n", flush=True)

    path_problem = root_path_problem(ROOT.parent)
    report(results, "安装路径长度", not path_problem, path_problem)

    environment = subprocess.run(
        [sys.executable, "-B", str(ROOT / "check_environment.py"), "--config", str(config_path)],
        cwd=str(ROOT), capture_output=True, encoding="utf-8", errors="replace",
    )
    failed_lines = [line for line in environment.stdout.splitlines() if not line.startswith("PASS")]
    report(
        results, "运行环境和模型文件", environment.returncode == 0,
        "；".join(failed_lines[:5]) if environment.returncode != 0 else "",
    )

    config, _ = load_config(config_path)
    version = config.get("display_version") or config.get("product_version")
    print(f"\n正在分析内置示例（系统版本 {version}），首次运行要建索引，可能需要几分钟……", flush=True)
    started = time.perf_counter()
    try:
        status = run_task(
            QUERY,
            selected_modules=("deepfake", "identity", "source"),
            config_path=config_path,
            original_file_name=f"自检_{QUERY.name}",
        )
    except Exception as exc:  # noqa: BLE001
        report(results, "综合分析", False, f"{type(exc).__name__}: {exc}")
        return finish(results)
    elapsed = time.perf_counter() - started
    report(results, "综合分析", status.get("status") == "success", f"状态 {status.get('status')}，耗时 {elapsed:.1f} 秒")

    task_dir = Path(str(status.get("task_dir")))
    summary = summarize_task(task_dir) or {}
    report(
        results, "身份检索结果", summary.get("identity_top") == EXPECTED_IDENTITY,
        f"第一候选 {summary.get('identity_top')}（{summary.get('identity_label')}）",
    )
    report(
        results, "素材检索结果", summary.get("source_top") == EXPECTED_SOURCE,
        f"第一候选 {summary.get('source_top')}（{summary.get('source_label')}）",
    )
    detection = json.loads((task_dir / "detection_result.json").read_text(encoding="utf-8"))
    branch = (detection.get("method") or {}).get("branch")
    report(
        results, "整图分支走 ViT 模型", branch == EXPECTED_BRANCH,
        f"分支 {branch}，结论 {summary.get('deepfake_label')}",
    )
    general = detection.get("general_aigc") or {}
    report(
        results, "通用 AIGC 分支", general.get("status") == "success",
        f"档位 {general.get('band_label')}，融合分数 {general.get('logit')}，设备 "
        f"{(general.get('runtime') or {}).get('device_name')}"
        if general.get("status") == "success"
        else str(general.get("error_message")),
    )
    face = general.get("face_branch") or {}
    report(
        results, "人脸分支", face.get("status") == "success" and bool(face.get("crops")),
        f"档位 {face.get('band_label')}，裁剪图均值分数 {face.get('logit')}"
        if face.get("status") == "success"
        else str(face.get("error_message") or face.get("status")),
    )
    provenance_path = task_dir / "file_provenance_result.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    report(results, "文件标识解析", provenance.get("status") == "success", str(provenance.get("verdict_label")))
    visible_path = task_dir / "visible_label_result.json"
    visible = json.loads(visible_path.read_text(encoding="utf-8")) if visible_path.is_file() else {}
    report(results, "画面文字识别", visible.get("status") == "success" and visible.get("verdict") != "unavailable", str(visible.get("verdict_label")))
    evidence_warnings = identity_evidence_warnings(task_dir)
    report(
        results, "身份证据图", not evidence_warnings,
        f"{len(evidence_warnings)} 条警告，例如：{evidence_warnings[0][:60]}……" if evidence_warnings else "",
    )
    verification = verify_task_artifacts(task_dir)
    report(results, "证据文件完整性", bool(verification.get("available")) and not verification.get("problems"), str(verification.get("message")))
    reports = [task_dir / "final_report.json", *task_dir.glob("检测结果_*.html"), *task_dir.glob("检测结果_*.txt")]
    report(results, "报告生成", len(reports) >= 3 and all(path.is_file() for path in reports), f"{len(reports)} 份")
    final_report = json.loads((task_dir / "final_report.json").read_text(encoding="utf-8"))
    analysis = final_report.get("analysis") or {}
    technical_name = analysis.get("technical_pdf_file")
    pdfs = [task_dir / technical_name] if technical_name else list(task_dir.glob("检测结果_*.pdf"))
    pdf_ok = len(pdfs) == 1 and pdfs[0].is_file() and pdfs[0].read_bytes().startswith(b"%PDF-")
    report(results, "PDF 能生成", pdf_ok, f"{len(pdfs)} 份")
    if isinstance(config.get("llm"), dict):
        from reporting.analysis_pipeline import verify_analysis_artifacts
        analysis_path = task_dir / "研判分析报告.pdf"
        analysis_ok = analysis.get("status") == "success" and analysis_path.is_file()
        analysis_problems = verify_analysis_artifacts(task_dir) if analysis_ok else ["研判报告未生成"]
        report(results, "研判分析报告", analysis_ok and not analysis_problems,
               "；".join(analysis_problems) if analysis_problems else "报告与补充证据清单一致")
    print(f"\n任务目录：{task_dir}", flush=True)
    return finish(results)


def finish(results: list[tuple[str, bool, str]]) -> int:
    failed = [name for name, passed, _ in results if not passed]
    print(f"\n自检结束：{len(results) - len(failed)} 项通过，{len(failed)} 项失败。", flush=True)
    if failed:
        print("未通过：" + "、".join(failed), flush=True)
    else:
        print("系统可以正常演示。", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
