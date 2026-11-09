"""从已保存的报告内容生成中文研判PDF，不修改任务文件和机器结论。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from io import BytesIO
import json
import math
from pathlib import Path, PureWindowsPath
import re
import threading
from typing import Any
from product_info import PRODUCT_NAME as _PRODUCT_NAME
from decision.aigc_evidence import UNAVAILABLE_NOTE, legacy_overall_statistics_match
from reporting.text_safety import redact_local_paths
from reporting.review_records import review_report_mismatch

from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Line, PolyLine, String, Circle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Image, KeepTogether, LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
FONT_REGULAR = "ReportChinese"
FONT_BOLD = "ReportChinese-Bold"
_FONT_LOCK = threading.RLock()
_PAGE_WIDTH, _PAGE_HEIGHT = A4
_MARGIN = 42
_WIDTH = _PAGE_WIDTH - 2 * _MARGIN
_INK = colors.HexColor("#223247")
_BLUE = colors.HexColor("#244F73")
_PALE = colors.HexColor("#EEF3F7")
_LINE = colors.HexColor("#CFD9E1")
_PATH_RE = re.compile(r"(?i)(?:[a-z]:(?:\\|/(?!/))|\\\\)[^\r\n；;，,。]+")
_BAND_NAMES = {"low": "证据不足", "medium": "需人工复核", "high": "高风险提示"}
_STATUS_NAMES = {"success": "已完成", "no_face": "不适用，没有检出可用人脸", "skipped": "未运行", "error": "未完成", "failed": "未完成", "partial_success": "部分完成"}
_DEFAULT_DISCLAIMER = (
    "本系统用于深度伪造风险筛查、身份候选检索和素材候选检索，仅提供侦查线索和辅助研判信息，"
    "不替代原始介质固定、哈希校验、人工复核或司法鉴定。"
)
_EVIDENCE_DISCLAIMER = (
    "下表是测试集上的统计：伪造样本落入这一档的比例，除以真实样本落入这一档的比例。"
    "它说明这一档的分量，不是本案的概率，也不是正式的似然比，换一批数据会变化。"
    "结果只作侦查线索，不替代人工复核和司法鉴定。"
)
_LOW_NOTE = "证据不足不能排除伪造，约三分之二的换脸测试图都落在这一档。"
_FORGERY_UNAVAILABLE = "当前版本未给出伪造类型结论，请结合上面的素材和身份候选由人工判断。"
REVIEW_NOTICE = "复核人姓名是手填的，没有账号认证；这不是可信时间戳，也不是不可篡改存证。"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list:
    return list(value) if isinstance(value, (list, tuple)) else []


def _module(report: Mapping[str, Any], name: str) -> tuple[dict, dict]:
    """沿用结构化报告中结果与解释分开存放的结构。"""
    block = _mapping(_mapping(report.get("modules")).get(name))
    return _mapping(block.get("result")), _mapping(block.get("explanation"))


def _basename(value: Any) -> str:
    if value is None:
        return "未记录"
    text = str(value).strip().replace("\\", "/").rstrip("/")
    return text.rsplit("/", 1)[-1] or "未记录"


def _text(value: Any, fallback: str = "未记录") -> str:
    if value is None:
        return fallback
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    # 完整本机路径不进入普通报告文本；用户输入仍按普通文字显示。
    text = redact_local_paths(text)
    return "".join(c for c in text if c in "\n\t" or ord(c) >= 32)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _fmt(value: Any, digits: int = 6) -> str:
    number = _number(value)
    return "未记录" if number is None else f"{number:.{digits}f}"


def _fonts() -> None:
    """只从随系统分发的静态字体读取，缺失时交给上层降级处理。"""
    regular = FONT_DIR / "LXGWWenKai-Regular.ttf"
    bold = FONT_DIR / "LXGWWenKai-Bold.ttf"
    if not regular.is_file() or not bold.is_file():
        raise RuntimeError("PDF中文字体缺失，无法生成报告")
    with _FONT_LOCK:
        registered = set(pdfmetrics.getRegisteredFontNames())
        if FONT_REGULAR not in registered:
            pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(regular)))
        if FONT_BOLD not in registered:
            pdfmetrics.registerFont(TTFont(FONT_BOLD, str(bold)))
        pdfmetrics.registerFontFamily(FONT_REGULAR, normal=FONT_REGULAR, bold=FONT_BOLD, italic=FONT_REGULAR, boldItalic=FONT_BOLD)


def check_pdf_ready() -> None:
    """检查并注册固定中文字体，不生成PDF或创建任务文件。"""
    _fonts()


def _styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle("正文", fontName=FONT_REGULAR, fontSize=9.5, leading=15, textColor=_INK, wordWrap="CJK", splitLongWords=True, spaceAfter=5)
    return {
        "body": body,
        "small": ParagraphStyle("说明", parent=body, fontSize=8, leading=12, textColor=colors.HexColor("#526172")),
        "cell": ParagraphStyle("表格", parent=body, fontSize=8, leading=12, spaceAfter=0),
        "header": ParagraphStyle("表头", parent=body, fontName=FONT_BOLD, fontSize=8, leading=12, spaceAfter=0),
        "title": ParagraphStyle("主标题", parent=body, fontName=FONT_BOLD, fontSize=19, leading=28, spaceAfter=8),
        "heading": ParagraphStyle("章节标题", parent=body, fontName=FONT_BOLD, fontSize=13, leading=19, textColor=_BLUE, spaceBefore=9, spaceAfter=8, keepWithNext=True),
        "subheading": ParagraphStyle("小节标题", parent=body, fontName=FONT_BOLD, fontSize=10, leading=16, spaceBefore=7, spaceAfter=5, keepWithNext=True),
        "band": ParagraphStyle("总体档位", parent=body, fontName=FONT_BOLD, fontSize=24, leading=32, textColor=_BLUE, spaceBefore=6, spaceAfter=10),
        "caption": ParagraphStyle("图注", parent=body, fontSize=8, leading=12, alignment=TA_CENTER, spaceAfter=4),
    }


def _p(value: Any, styles: dict, style: str = "body") -> Paragraph:
    return Paragraph(escape(_text(value), quote=True).replace("\n", "<br/>"), styles[style])


def _table(headers: Sequence[str], rows: list[list[Any]], widths: list[float], styles: dict) -> LongTable:
    content = [[_p(value, styles, "header") for value in headers]]
    content.extend([[_p(value, styles, "cell") for value in row] for row in rows])
    table = LongTable(content, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1, splitInRow=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _PALE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFBFC")]),
        ("GRID", (0, 0), (-1, -1), 0.4, _LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _safe_file(root: Path, value: Any) -> Path | None:
    """拒绝绝对路径、越界路径和指向任务外部的链接。"""
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.replace("\\", "/")
    if PureWindowsPath(value).is_absolute() or Path(value).is_absolute() or ":" in value:
        return None
    try:
        candidate = (root / value).resolve()
        candidate.relative_to(root)
        return candidate if candidate.is_file() else None
    except (ValueError, OSError, RuntimeError):
        return None


def _image(root: Path, relative: Any, width: float, height: float, styles: dict):
    path = _safe_file(root, relative)
    if path is None:
        return _p("图片缺失或未保存在任务目录内", styles, "caption")
    try:
        from PIL import Image as PILImage, ImageOps
        with PILImage.open(path) as source:
            source.load()
            photo = ImageOps.exif_transpose(source).convert("RGB")
            photo.thumbnail((1600, 1600), PILImage.Resampling.LANCZOS)
            scale = min(width / photo.width, height / photo.height)
            stream = BytesIO()
            # 报告只嵌入展示用缩略图；原始证据图片保持原样。
            photo.save(stream, format="JPEG", quality=90, optimize=True)
            stream.seek(0)
            return Image(stream, width=photo.width * scale, height=photo.height * scale, hAlign="CENTER")
    except (OSError, ValueError, SyntaxError, PILImage.DecompressionBombError):
        return _p("图片无法读取，原记录仍保留在任务证据中", styles, "caption")


def _read_task_json(root: Path, relative: str) -> dict:
    path = _safe_file(root, relative)
    if path is None:
        return {}
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8-sig")))
    except (OSError, UnicodeError, ValueError):
        return {}


def _query_relative(report: Mapping[str, Any], root: Path) -> str | None:
    input_data = _mapping(report.get("input"))
    stored = _mapping(_mapping(_mapping(report.get("provenance")).get("task_status")).get("input"))
    video = _mapping(input_data.get("video"))
    primary = video.get("primary_frame")
    if isinstance(primary, Mapping):
        primary = primary.get("path") or primary.get("file")
    for value in (stored.get("query_image"), primary, input_data.get("query_image"), input_data.get("file_name"), "query.jpg", "query.png"):
        path = _safe_file(root, value)
        if path is not None:
            return path.relative_to(root).as_posix()
    return None


class _NumberedCanvas(Canvas):
    """第二遍写页脚，确保总页数来自实际分页。"""

    def __init__(self, *args, task_id: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_pages = []
        self._task_id = task_id

    def showPage(self):
        self._saved_pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_pages)
        for page in self._saved_pages:
            self.__dict__.update(page)
            self.saveState()
            self.setStrokeColor(_LINE)
            self.line(_MARGIN, 35, _PAGE_WIDTH - _MARGIN, 35)
            self.setFont(FONT_REGULAR, 7)
            self.setFillColor(colors.HexColor("#526172"))
            task_text = "任务编号：" + _text(self._task_id)
            while task_text and pdfmetrics.stringWidth(task_text, FONT_REGULAR, 7) > _WIDTH - 105:
                task_text = task_text[:-1]
            self.drawString(_MARGIN, 23, task_text)
            self.drawRightString(_PAGE_WIDTH - _MARGIN, 23, f"第 {self._pageNumber} 页，共 {total} 页")
            self.restoreState()
            super().showPage()
        super().save()


def _frame_chart(general: dict, styles: dict):
    frames = [_mapping(frame) for frame in _list(general.get("frames"))]
    key = next((field for field in ("clip_logit", "logit", "score") if any(_number(frame.get(field)) is not None for frame in frames)), None)
    if key is None:
        return [_p("未记录可绘制的逐帧分数。", styles, "small")]
    points = []
    for index, frame in enumerate(frames):
        score = _number(frame.get(key))
        stamp = _number(frame.get("timestamp_seconds"))
        if score is not None:
            points.append((stamp if stamp is not None else float(index), score))
    points.sort()
    drawing = Drawing(_WIDTH, 184)
    left, bottom, width, height = 43, 31, _WIDTH - 60, 125
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
    if xmax == xmin:
        xmax = xmin + 1
    pad = max((ymax - ymin) * 0.12, 0.1)
    ymin, ymax = ymin - pad, ymax + pad
    xy = lambda x, y: (left + (x - xmin) / (xmax - xmin) * width, bottom + (y - ymin) / (ymax - ymin) * height)
    for index in range(5):
        y = ymin + index * (ymax - ymin) / 4
        py = xy(xmin, y)[1]
        drawing.add(Line(left, py, left + width, py, strokeColor=_LINE, strokeWidth=0.4))
        drawing.add(String(left - 6, py - 3, f"{y:.2f}", fontName=FONT_REGULAR, fontSize=7, textAnchor="end"))
    drawing.add(Line(left, bottom, left + width, bottom, strokeColor=_INK, strokeWidth=0.6))
    drawing.add(Line(left, bottom, left, bottom + height, strokeColor=_INK, strokeWidth=0.6))
    flattened = [coordinate for point in points for coordinate in xy(*point)]
    if len(points) > 1:
        drawing.add(PolyLine(flattened, strokeColor=_BLUE, strokeWidth=1.2, fillColor=None))
    for point in points:
        x, y = xy(*point)
        drawing.add(Circle(x, y, 1.7, fillColor=_BLUE, strokeColor=None))
    for position in (0, 0.5, 1):
        x = xmin + position * (xmax - xmin)
        drawing.add(String(left + position * width, bottom - 13, f"{x:.2f}", fontName=FONT_REGULAR, fontSize=7, textAnchor="middle"))
    drawing.add(String(left + width / 2, 5, "时间（秒）；缺时间时使用帧序号", fontName=FONT_REGULAR, fontSize=8, textAnchor="middle"))
    title = "逐帧CLIP分数（logit）" if key == "clip_logit" else "逐帧分数（logit）" if key == "logit" else "逐帧记录分数"
    drawing.add(String(left, 170, title, fontName=FONT_BOLD, fontSize=9))
    return [drawing, _p("曲线来自通用分支已保存的逐帧记录，不是总体档位，也不是伪造概率。", styles, "small")]


def _crop_grid(general: dict, root: Path, styles: dict) -> list:
    face = _mapping(general.get("face_branch"))
    paths = []
    for value in _list(face.get("crops")):
        value = _mapping(value).get("file") if isinstance(value, Mapping) else value
        if isinstance(value, str):
            relative = value if value.replace("\\", "/").startswith("face_crops/") else "face_crops/" + value
            if relative not in paths:
                paths.append(relative)
    if not paths:
        directory = (root / "face_crops").resolve()
        if directory.is_relative_to(root) and directory.is_dir():
            paths = [path.relative_to(root).as_posix() for path in sorted(directory.iterdir()) if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not paths:
        return [_p("不适用：没有保存可展示的人脸裁剪图。", styles, "small")]
    cells = [[_image(root, path, _WIDTH / 3 - 16, 92, styles), _p(f"人脸裁剪 {index + 1}：{_basename(path)}", styles, "caption")] for index, path in enumerate(paths[:6])]
    rows = [cells[index:index + 3] + [""] * (3 - len(cells[index:index + 3])) for index in range(0, len(cells), 3)]
    table = Table(rows, colWidths=[_WIDTH / 3] * 3, hAlign="LEFT")
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return [_p(f"本报告展示{min(6, len(paths))}张已保存裁剪图，最多展示6张。", styles, "small"), table]


def _module_empty(result: dict, explanation: dict, name: str) -> str:
    status = str(result.get("status") or explanation.get("status") or "")
    if status in {"error", "failed"}:
        return f"{name}未完成：{_text(result.get('error_message'), '未记录具体原因')}"
    if status == "skipped":
        return _text(_mapping(result.get("details")).get("reason") or explanation.get("summary"), f"{name}未运行。")
    if status == "no_face":
        return "不适用：没有可用人脸。"
    return "库内未发现可靠来源。" if name == "素材检索" else "库内未发现可靠身份候选。"


def _pair_section(report: Mapping[str, Any], root: Path, styles: dict) -> list:
    pair = _mapping(report.get("pair_compare")) or _read_task_json(root, "pair_compare_result.json")
    if not pair:
        return []
    story = [_p("配对比对（前三候选）", styles, "subheading")]
    for item in _list(pair.get("candidates"))[:3]:
        row = _mapping(item)
        comparison = _mapping(row.get("comparison"))
        artifacts = _mapping(row.get("artifacts"))
        story.append(_p(f"第{row.get('rank')}候选：{_text(row.get('candidate_name'))}", styles, "subheading"))
        if comparison.get("alignment_status") != "completed":
            story.append(_p("对齐失败，暂不能判断是否同源" if comparison.get("alignment_status") == "failed" else "未形成可比对的对齐，请人工核查。", styles))
            continue
        cells = []
        for key, label in (("reference", "参考图"), ("aligned", "对齐后的查询图"), ("difference", "差异热图")):
            cells.append([_image(root, artifacts.get(key), _WIDTH / 3 - 14, 105, styles), _p(label, styles, "caption")])
        table = Table([cells], colWidths=[_WIDTH / 3] * 3, hAlign="LEFT")
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.extend([table, _p("差异热图只提示像素变化位置，不能单凭颜色判断伪造", styles, "small")])
        rows = []
        for face in _list(comparison.get("faces")):
            face = _mapping(face)
            rows.append([face.get("query_face_id"), {"matched": "已找到几何对应", "unmatched": "未找到几何对应"}.get(face.get("match_status"), "未记录"), _fmt(face.get("cosine_similarity"))])
        if rows:
            story.append(_table(["查询脸编号", "对应状态", "余弦相似度"], rows, [95, _WIDTH-205, 110], styles))
        story.append(_p("几何对应不等于身份绑定", styles, "small"))
    return story


def _manual_judgment_text(record: Mapping[str, Any]) -> str:
    text = f"人工判断，非系统结论：{_text(record.get('manual_judgment'), '未作类型判断')}；复核人：{_text(record.get('reviewer'))}；时间：{_text(record.get('reviewed_at'))}"
    if record.get("basis_candidate_rank") is not None:
        text += f"；依据第{record['basis_candidate_rank']}候选：{_text(record.get('basis_candidate_name'))}；候选SHA-256：{_text(record.get('basis_candidate_sha256'))}"
    if record.get("target_face_index") is not None:
        text += f"；人工选定的目标脸：{record['target_face_index']}"
    return text


def _source_section(report: Mapping[str, Any], root: Path, styles: dict) -> list:
    story = [_p("三、溯源情况", styles, "heading"), _p("文件标识", styles, "subheading")]
    provenance = _mapping(report.get("file_provenance"))
    if provenance.get("status") == "success":
        story.append(_p(provenance.get("verdict_label") or "未记录文件标识结论", styles))
        for label in _list(provenance.get("aigc_labels")):
            label = _mapping(label)
            story.append(_p(f"{_text(label.get('standard'))}：{_text(label.get('label_meaning'))}", styles, "small"))
        c2pa = _mapping(provenance.get("c2pa"))
        story.append(_p("C2PA：" + ("检测到内容凭证结构；" if c2pa.get("present") else "未记录内容凭证结构；") + _text(c2pa.get("note"), "只检测结构，不验证签名"), styles, "small"))
        for limitation in _list(provenance.get("limitations")):
            story.append(_p(limitation, styles, "small"))
    else:
        story.append(_p("文件标识解析未完成或不适用：" + _text(provenance.get("error_message"), "没有可用记录"), styles))
    from modules.visible_label import visible_label_lines
    for index, line in enumerate(visible_label_lines(_mapping(report.get("visible_label")))):
        story.append(_p(line, styles, "subheading" if index == 0 else "small"))
    source, explanation = _module(report, "source")
    technical = _mapping(explanation.get("technical"))
    candidates = _list(technical.get("candidates")) or _list(source.get("candidates"))
    decision = _mapping(technical.get("open_set")).get("state_label")
    story.append(_p("素材候选前三名", styles, "subheading"))
    if not candidates:
        story.append(_p(_module_empty(source, explanation, "素材检索"), styles))
    else:
        story.append(_p(explanation.get("summary") or "素材候选仅表示当前库内排序。", styles, "small"))
        for index, item in enumerate(candidates[:3]):
            candidate = _mapping(item)
            rank = candidate.get("rank", index + 1)
            name = candidate.get("file_name") or _basename(candidate.get("file_path"))
            verdict = candidate.get("state_label") or (decision if index == 0 else None) or "库内排序候选"
            relative = candidate.get("visualization_relative_path") or _mapping(candidate.get("evidence_artifacts")).get("matches_inliers")
            block = [_p(f"第{rank}候选：{name}；判定：{verdict}", styles, "subheading")]
            block.append(_table(["全局分", "局部分", "最终分"], [[_fmt(candidate.get("global_score")), _fmt(candidate.get("local_score")), _fmt(candidate.get("final_score"))]], [_WIDTH / 3] * 3, styles))
            block.append(Spacer(1, 5))
            block.append(_image(root, relative, _WIDTH - 10, 123, styles))
            block.append(_p("查询图与候选素材的已保存对比证据；分数不是来源概率。", styles, "caption"))
            story.append(KeepTogether(block))
    story.extend(_pair_section(report, root, styles))
    identity, explanation = _module(report, "identity")
    technical = _mapping(explanation.get("technical"))
    candidates = _list(technical.get("candidates")) or _list(identity.get("candidates"))
    decision = _mapping(technical.get("open_set")).get("state_label")
    story.append(_p("身份候选前三名", styles, "subheading"))
    if not candidates:
        story.append(_p(_module_empty(identity, explanation, "身份比对"), styles))
    else:
        rows = []
        for index, item in enumerate(candidates[:3]):
            candidate = _mapping(item)
            rows.append([candidate.get("rank", index + 1), candidate.get("identity_name") or candidate.get("identity_id"), _fmt(candidate.get("mean_similarity", candidate.get("score"))), candidate.get("state_label") or (decision if index == 0 else None) or "库内排序候选"])
        story.append(_table(["排序", "身份候选", "人员级相似度", "判定"], rows, [38, 170, 100, _WIDTH - 308], styles))
        story.append(_p("候选相似度不是身份概率，高可信候选仍需人工比对确认。", styles, "small"))
    from reporting.identity_faces import face_notes, face_table_rows
    per_face_rows = face_table_rows(identity)
    if per_face_rows:
        story.append(_p("逐脸身份候选（每脸前三名）", styles, "subheading"))
        rows = [[face, rank, name, _fmt(score), verdict] for face, rank, name, score, verdict in per_face_rows]
        story.append(_table(["查询脸", "排序", "身份候选", "余弦相似度", "判定"], rows,
                            [75, 35, 135, 85, _WIDTH - 330], styles))
        for note in face_notes(identity):
            story.append(_p(note, styles, "small"))
    return story


def _source_method_text(value: Any) -> str | None:
    """方法概述使用中文，完整参数留在结构化报告中。"""
    if isinstance(value, Mapping):
        if not value:
            return None
        name = value.get("name") or value.get("description")
        if not name:
            return "全局特征检索和局部几何复核，具体参数见结构化报告"
    else:
        name = value
    if name == "face-masked global retrieval plus SIFT geometric verification":
        return "遮蔽人脸后的全局特征检索与 SIFT 局部几何复核"
    return str(name) if name is not None else None


def _appendix(report: Mapping[str, Any], general: dict, technical: dict, overall: dict, styles: dict) -> list:
    model = _mapping(general.get("model"))
    face = _mapping(general.get("face_branch"))
    identity, identity_explanation = _module(report, "identity")
    source, _ = _module(report, "source")
    rows = [
        ["系统版本", _mapping(report.get("product")).get("version")],
        ["报告结构版本", report.get("schema_version")],
        ["通用模型标识", model.get("model_id")],
        ["通用模型SHA-256", model.get("checkpoint_sha256")],
        ["校准版本", model.get("calibration_version")],
        ["整图模型", technical.get("model_name")],
        ["整图模型版本", technical.get("model_version")],
        ["整图模型SHA-256", technical.get("model_sha256") or technical.get("model_hashes")],
        ["身份模型", _mapping(identity_explanation.get("technical")).get("model_name") or _mapping(identity.get("method")).get("model_name")],
        ["素材检索方法", _source_method_text(source.get("method"))],
        ["总体规则名", overall.get("rule")],
    ]
    story = [_p("六、附录", styles, "heading"), _table(["项目", "任务记录"], rows, [115, _WIDTH - 115], styles)]
    threshold_rows = []
    for name, values in (("通用分支", general.get("band_thresholds")), ("人脸分支", face.get("band_thresholds")), ("整图分支（仅提示）", technical.get("band_thresholds"))):
        values = _mapping(values)
        threshold_rows.append([name, _fmt(values.get("low_max")), _fmt(values.get("medium_max"))])
    story.extend([_p("分档边界", styles, "subheading"), _table(["分支", "低档/中档边界", "中档/高档边界"], threshold_rows, [180, (_WIDTH - 180) / 2, (_WIDTH - 180) / 2], styles), _p("以上版本和边界来自该任务记录；缺失处不补写当前配置。", styles, "small")])
    limitations = []
    for name in ("deepfake", "source", "identity"):
        raw, explanation = _module(report, name)
        for value in _list(explanation.get("limitations")) + _list(raw.get("limitations")):
            value = _text(value).strip().rstrip("。；;")
            if value not in limitations:
                limitations.append(value)
    for value in _list(general.get("limitations")):
        value = _text(value).strip().rstrip("。；;")
        if value not in limitations:
            limitations.append(value)
    story.append(_p("局限", styles, "subheading"))
    for index, text in enumerate(limitations or [_DEFAULT_DISCLAIMER], 1):
        story.append(_p(f"{index}. {text}", styles, "small"))
    story.append(_p("完整证据清单在任务目录的 artifact_manifest.json。清单在报告之后生成，本PDF不包含清单哈希。", styles, "small"))
    return story


def render_pdf_report(report: Mapping[str, Any], task_dir: str | Path, review_records: Sequence[Mapping[str, Any]] | None = None) -> bytes:
    """返回A4中文PDF字节；仅读取任务内图片和随系统分发的字体。"""
    if not isinstance(report, Mapping):
        raise ValueError("报告内容必须为对象")
    check_pdf_ready()
    styles = _styles()
    root = Path(task_dir).resolve()
    task = _mapping(report.get("task"))
    task_id = _text(task.get("task_id"))
    input_data = _mapping(report.get("input"))
    video = _mapping(input_data.get("video"))
    product = _mapping(report.get("product"))
    deepfake, explanation = _module(report, "deepfake")
    technical = _mapping(explanation.get("technical"))
    overall = _mapping(technical.get("overall"))
    general = _mapping(deepfake.get("general_aigc")) or _read_task_json(root, "general_aigc_result.json")
    statistics_applicable = legacy_overall_statistics_match(general)
    if not statistics_applicable:
        # 已存旧任务也要遮止不适用的旧统计，只改本次渲染副本。
        overall["band_evidence"] = {"status": "unavailable", "rows": [], "note": UNAVAILABLE_NOTE}
        hints = list(overall.get("hints") or [])
        if not any(isinstance(hint, Mapping) and hint.get("code") == "overall_statistics_unavailable" for hint in hints):
            hints.append({"code": "overall_statistics_unavailable", "message": UNAVAILABLE_NOTE})
        overall["hints"] = hints
    reviews = [dict(value) for value in (review_records or []) if isinstance(value, Mapping)]
    product_name = _text(product.get("name"), _PRODUCT_NAME)
    story = [_p(product_name, styles, "title"), _p("检测研判报告" + ("（复核版）" if reviews else ""), styles, "heading")]
    digest = video.get("video_sha256") if video else input_data.get("input_sha256")
    stored_input = _mapping(_mapping(_mapping(report.get("provenance")).get("task_status")).get("input"))
    filename = _basename(stored_input.get("original_file_name") or (video.get("video_file") if video else input_data.get("file_name") or input_data.get("query_image")))
    rows = [["任务编号", task_id], ["文件名", filename], ["输入类型", "视频" if video else "图片"], ["输入文件SHA-256", digest], ["报告生成时间", task.get("report_generated_at")], ["人工复核", f"已有{len(reviews)}条复核记录" if reviews else "尚无人工复核记录"]]
    if video:
        rows.append(["视频首要帧SHA-256", input_data.get("input_sha256")])
    story.append(_table(["基本信息", "内容"], rows, [115, _WIDTH - 115], styles))
    from .final_decision_view import saved_decision_view
    final_view = saved_decision_view(report)
    if final_view is not None:
        story.extend([Spacer(1, 8), _p("最终研判档位", styles, "subheading"), _p(final_view["label"], styles, "band"),
                      _p(final_view["reason"] or "详见已保存决策。", styles),
                      _p("本页沿用已保存的最终决策；下方基础机器档位用于解释检测过程。", styles, "small")])
    story.extend([Spacer(1, 8), _p("基础机器档位" if final_view else "总体档位", styles, "subheading"), _p(overall.get("band_label") or _BAND_NAMES.get(overall.get("band")) or "未记录总体档位", styles, "band"), _p(overall.get("reason") or explanation.get("summary") or "没有可用的总体档位依据。", styles)])
    if reviews and reviews[-1].get("manual_judgment"):
        story.append(_p(_manual_judgment_text(reviews[-1]), styles))
        if review_report_mismatch(root, reviews[-1]):
            story.append(_p("该条复核所依据的报告与当前报告不一致", styles, "heading"))
    for hint in _list(overall.get("hints")):
        message = _mapping(hint).get("message")
        if message:
            story.append(_p(message, styles))
    if overall.get("band") == "low":
        story.append(_p(_LOW_NOTE if statistics_applicable else "证据不足不能排除伪造，请结合其他线索人工复核。", styles))
    story.extend([Spacer(1, 5), _image(root, _query_relative(report, root), _WIDTH, 140, styles), _p("视频首要帧缩略图" if video else "查询图缩略图", styles, "caption"), _p(report.get("disclaimer") or _DEFAULT_DISCLAIMER, styles, "small"), PageBreak()])

    story.append(_p("二、检测情况", styles, "heading"))
    story.append(_p("三个检测分支，其中两路定结论，一路只作提示。各分支分数性质不同，不相加。", styles, "small"))
    branch_rows = []
    for item in _list(overall.get("branches")):
        branch = _mapping(item)
        name = branch.get("label") or "未记录分支"
        if branch.get("key") == "full_image":
            name = "整图分支（只作提示）"
        branch_rows.append([name, _STATUS_NAMES.get(str(branch.get("status")), _text(branch.get("status"))), branch.get("band_label") or _BAND_NAMES.get(branch.get("band")) or "不适用/未记录", _fmt(branch.get("score")), branch.get("error_message") or ""])
    if branch_rows:
        story.append(_table(["分支", "状态", "档位", "记录分数", "说明"], branch_rows, [140, 78, 84, 85, _WIDTH - 387], styles))
    else:
        story.append(_p("旧任务未记录各检测分支的总体分档表。", styles))
    evidence = _mapping(overall.get("band_evidence"))
    evidence_rows = []
    for value in _list(evidence.get("rows")):
        item = _mapping(value)
        try:
            evidence_rows.append([item["label"], f"{item['real_count']}/{item['real_total']}（{100 * item['real_ratio']:.1f}%）", f"{item['fake_count']}/{item['fake_total']}（{100 * item['fake_ratio']:.1f}%）", "无法计算" if item["ratio"] is None else _fmt(item["ratio"], 2), "样本少，只能看个大概" if item["small_sample"] else ""])
        except (KeyError, TypeError, ValueError):
            continue
    if evidence_rows:
        story.extend([_p("当前档位的证据强度", styles, "subheading"), _table(["测试场景", "真实样本", "伪造样本", "比例之比", "说明"], evidence_rows, [88, 116, 116, 62, _WIDTH - 382], styles), _p(_EVIDENCE_DISCLAIMER, styles, "small")])
    if video or general.get("input_kind") == "video":
        story.append(_p("视频逐帧记录", styles, "subheading"))
        story.extend(_frame_chart(_read_task_json(root, "general_aigc_result.json") or general, styles))
    story.append(_p("人脸裁剪证据", styles, "subheading"))
    story.extend(_crop_grid(general, root, styles))
    story.append(PageBreak())
    story.extend(_source_section(report, root, styles))
    story.append(_p("四、伪造类型结论", styles, "heading"))
    forgery = report.get("forgery_type")
    if isinstance(forgery, Mapping):
        forgery = forgery.get("label") or forgery.get("conclusion") or forgery.get("description")
    from pair_integration import pair_report_lines
    from source_upgrade import case_report_lines

    pair_lines = pair_report_lines(report.get("pair_compare"))
    if not pair_lines:
        story.append(_p(forgery or _FORGERY_UNAVAILABLE, styles))
    for line in pair_lines:
        story.append(_p(line, styles))
    for record in reviews:
        if record.get("manual_judgment"):
            story.append(_p(_manual_judgment_text(record), styles))
            if review_report_mismatch(root, record):
                story.append(_p("该条复核所依据的报告与当前报告不一致", styles, "heading"))
    library_lines = case_report_lines(report.get("case_libraries"))
    if library_lines:
        story.append(_p("本次比对库", styles, "subheading"))
        story.extend(_p(line, styles, "small") for line in library_lines)
    if reviews:
        story.append(_p("五、人工复核", styles, "heading"))
        for index, record in enumerate(reviews, 1):
            story.append(_p(f"复核记录 {index}", styles, "subheading"))
            if review_report_mismatch(root, record):
                story.append(_p("该条复核所依据的报告与当前报告不一致", styles, "heading"))
            story.extend([_p(f"复核人：{_text(record.get('reviewer'))}；时间：{_text(record.get('reviewed_at'))}", styles), _p(f"结论：{_text(record.get('conclusion'))}", styles), _p("意见：" + _text(record.get("opinion")), styles), _p("所依据报告SHA-256：" + _text(record.get("report_sha256")), styles, "small")])
    else:
        story.append(KeepTogether([
            _p("五、人工复核", styles, "heading"),
            _table(["复核项目", "填写栏"], [["复核人", "________________________"], ["时间", "________________________"], ["意见", "\n\n________________________\n\n"], ["签字", "________________________"]], [90, _WIDTH - 90], styles),
            _p(REVIEW_NOTICE, styles, "small"),
        ]))
    if reviews:
        story.append(_p(REVIEW_NOTICE, styles, "small"))
    story.append(PageBreak())
    story.extend(_appendix(report, general, technical, overall, styles))
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=_MARGIN, rightMargin=_MARGIN, topMargin=36, bottomMargin=48, title=product_name + " - 研判报告", author=product_name, subject="检测结果、检索候选与人工复核记录", pageCompression=1, allowSplitting=1)
    document.build(story, canvasmaker=lambda *args, **kwargs: _NumberedCanvas(*args, task_id=task_id, **kwargs))
    return buffer.getvalue()
