# -*- coding: utf-8 -*-
"""把 文档/评测 下的汇总数据画成研究报告用的静态图（SVG）。

配色取自校验过的类别色：蓝表示检出或主方案，橙表示误报或对照。用法：
    python tools/research/make_report_figures.py 输入.json 输出目录
输入 JSON 见 build_figure_data 的产物，图的种类由键名决定。
"""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import sys

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e7e6e2"
FONT = "'Microsoft YaHei','Noto Sans CJK SC',sans-serif"


def _text(x: float, y: float, content: str, *, size: int = 15, color: str = INK_2, anchor: str = "start", weight: int = 400) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{escape(content)}</text>'
    )


def _canvas(width: int, height: int, title: str, subtitle: str, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="{FONT}">',
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
            _text(32, 42, title, size=22, color=INK, weight=700),
            _text(32, 70, subtitle, size=15),
            *body,
            "</svg>",
        ]
    )


def _legend(x: float, y: float, items: list[tuple[str, str]]) -> list[str]:
    parts: list[str] = []
    for label, color in items:
        parts.append(f'<circle cx="{x:.1f}" cy="{y - 5:.1f}" r="6" fill="{color}"/>')
        parts.append(_text(x + 12, y, label, size=15, color=INK))
        x += 26 + 15 * len(label)
    return parts


def dumbbell(
    rows: list[dict],
    *,
    title: str,
    subtitle: str,
    low_key: str,
    high_key: str,
    low_label: str,
    high_label: str,
    axis_min: float,
    axis_max: float,
    ticks: list[float],
    percent: bool,
    axis_title: str,
    low_color: str = ORANGE,
    high_color: str = BLUE,
    width: int = 960,
) -> str:
    """每行一根连杆：两端是同一数据集上的两个数。"""
    left, right = 290, width - 70
    top = 118
    step = 46
    bottom = top + step * len(rows)
    scale = lambda value: left + (value - axis_min) / (axis_max - axis_min) * (right - left)
    body = _legend(left, 100, [(low_label, low_color), (high_label, high_color)])
    for tick in ticks:
        x = scale(tick)
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{GRID}" stroke-width="1"/>')
        label = f"{tick:.0f}%" if percent else f"{tick:g}"
        body.append(_text(x, bottom + 22, label, size=14, anchor="middle"))
    body.append(_text((left + right) / 2, bottom + 48, axis_title, size=14, anchor="middle"))
    show = (lambda value: f"{value:.1f}%") if percent else (lambda value: f"{value:.3f}")
    for index, row in enumerate(rows):
        y = top + step * index + step / 2
        low, high = float(row[low_key]), float(row[high_key])
        # 类别名让出 60 像素，给贴近左端的数字留位置。
        body.append(_text(left - 60, y + 5, str(row["name"]), size=15, color=INK, anchor="end"))
        body.append(
            f'<line x1="{scale(low):.1f}" y1="{y:.1f}" x2="{scale(high):.1f}" y2="{y:.1f}" '
            f'stroke="#c9c8c3" stroke-width="2" stroke-linecap="round"/>'
        )
        for value, color in ((low, low_color), (high, high_color)):
            body.append(f'<circle cx="{scale(value):.1f}" cy="{y:.1f}" r="7" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        # 数字放在点的外侧；两点靠得太近时只往外推，不互相压住。
        first, second = (low, high) if low <= high else (high, low)
        body.append(_text(scale(first) - 12, y + 5, show(first), size=13, anchor="end"))
        body.append(_text(scale(second) + 12, y + 5, show(second), size=13))
    return _canvas(width, bottom + 66, title, subtitle, body)


def bars(
    rows: list[dict],
    *,
    title: str,
    subtitle: str,
    key: str,
    axis_min: float,
    axis_max: float,
    ticks: list[float],
    axis_title: str,
    width: int = 960,
) -> str:
    left, right = 250, width - 90
    top = 96
    step = 40
    bottom = top + step * len(rows)
    scale = lambda value: left + (value - axis_min) / (axis_max - axis_min) * (right - left)
    body: list[str] = []
    for tick in ticks:
        x = scale(tick)
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{GRID}" stroke-width="1"/>')
        body.append(_text(x, bottom + 22, f"{tick:g}", size=14, anchor="middle"))
    body.append(_text((left + right) / 2, bottom + 48, axis_title, size=14, anchor="middle"))
    for index, row in enumerate(rows):
        y = top + step * index + (step - 22) / 2
        value = float(row[key])
        end = scale(value)
        radius = 4
        body.append(_text(left - 16, y + 16, str(row["name"]), size=15, color=INK, anchor="end"))
        body.append(
            f'<path d="M {left} {y:.1f} H {end - radius:.1f} Q {end:.1f} {y:.1f} {end:.1f} {y + radius:.1f} '
            f'V {y + 22 - radius:.1f} Q {end:.1f} {y + 22:.1f} {end - radius:.1f} {y + 22:.1f} H {left} Z" fill="{BLUE}"/>'
        )
        body.append(_text(end + 8, y + 16, f"{value:.3f}", size=14, color=INK))
    return _canvas(width, bottom + 66, title, subtitle, body)


FIGURES = {"dumbbell": dumbbell, "bars": bars}


def main() -> int:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)
    for spec in json.loads(source.read_text(encoding="utf-8")):
        kind, name = spec.pop("kind"), spec.pop("file")
        rows = spec.pop("rows")
        (target / f"{name}.svg").write_text(FIGURES[kind](rows, **spec) + "\n", encoding="utf-8")
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
