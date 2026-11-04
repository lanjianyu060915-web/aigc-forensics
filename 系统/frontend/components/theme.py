"""界面样式。颜色只在这里定义，页面代码只用语义 class。"""

from __future__ import annotations

from html import escape

import streamlit as st


THEME_CSS = """
<style>
:root {
    --fsts-ink: #1B2733;
    --fsts-muted: #5B6B7B;
    --fsts-line: #DCE3EA;
    --fsts-surface: #FFFFFF;
    --fsts-primary: #1F4E79;
    --fsts-primary-soft: #E8F0F7;
    --fsts-ok: #1E7B4F;
    --fsts-ok-soft: #E4F4EC;
    --fsts-warn: #A8650B;
    --fsts-warn-soft: #FBF1E0;
    --fsts-bad: #B23B3B;
    --fsts-bad-soft: #FAE9E9;
}
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Segoe UI", sans-serif;
}
[data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent; height: 2.2rem; }
.block-container { max-width: 1480px; padding-top: 1.4rem; padding-bottom: 3rem; }

/* 标题层级：页面信息密度高，标题不需要很大 */
h1 { font-size: 1.55rem !important; font-weight: 700 !important; letter-spacing: 0; padding: 0 !important; }
h2 { font-size: 1.25rem !important; font-weight: 700 !important; letter-spacing: 0; }
h3 { font-size: 1.08rem !important; font-weight: 700 !important; letter-spacing: 0; }
h4 { font-size: 0.98rem !important; font-weight: 700 !important; letter-spacing: 0; }

/* 侧栏 */
[data-testid="stSidebar"] { border-right: 1px solid var(--fsts-line); }
[data-testid="stSidebar"] [data-testid="stRadio"] { width: 100%; }
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] {
    display: flex; flex-direction: column; width: 100%; gap: 0.15rem;
}
[data-testid="stSidebar"] [role="radiogroup"] > label {
    width: 100%; max-width: none; box-sizing: border-box; margin: 0; padding: 0.55rem 0.75rem;
    border-radius: 0.5rem; cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover { background: var(--fsts-primary-soft); }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background: var(--fsts-primary-soft);
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {
    color: var(--fsts-primary); font-weight: 700;
}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { display: none; }
.fsts-brand { padding: 0.1rem 0.2rem 0.9rem; }
.fsts-brand-name { font-size: 1.2rem; font-weight: 700; color: var(--fsts-ink); }
.fsts-brand-sub { font-size: 0.8rem; color: var(--fsts-muted); margin-top: 0.15rem; line-height: 1.45; }
.fsts-side-title { font-size: 0.78rem; font-weight: 700; color: var(--fsts-muted); margin: 0.2rem 0 0.45rem; }
.fsts-side-row { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;
    font-size: 0.85rem; color: var(--fsts-ink); padding: 0.22rem 0; }

/* 指标卡：中文结论较长，字号要能放下 */
[data-testid="stMetric"] {
    background: var(--fsts-surface); border: 1px solid var(--fsts-line);
    border-radius: 0.6rem; padding: 0.65rem 0.85rem;
}
[data-testid="stMetricLabel"] p { font-size: 0.8rem !important; color: var(--fsts-muted); }
[data-testid="stMetricValue"] { font-size: 1.12rem !important; font-weight: 700; line-height: 1.35; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
    overflow-wrap: anywhere;
}

/* 结果页五个页签：把单选按钮画成页签 */
[data-testid="stMain"] [role="radiogroup"] {
    display: flex; flex-wrap: wrap; gap: 0.3rem; background: #E9EEF3; padding: 0.3rem;
    border-radius: 0.6rem; width: fit-content;
}
[data-testid="stMain"] [role="radiogroup"] > label {
    margin: 0; padding: 0.42rem 0.95rem; border-radius: 0.45rem; cursor: pointer;
}
[data-testid="stMain"] [role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stMain"] [role="radiogroup"] > label:has(input:checked) {
    background: var(--fsts-surface); box-shadow: 0 1px 2px rgba(27, 39, 51, 0.12);
}
[data-testid="stMain"] [role="radiogroup"] > label:has(input:checked) p {
    color: var(--fsts-primary); font-weight: 700;
}

/* 卡片容器 */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--fsts-surface); border-color: var(--fsts-line); border-radius: 0.7rem;
}

/* 按钮 */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
    border-radius: 0.5rem; font-weight: 600; min-height: 2.5rem;
}

/* 上传区：把内置英文提示换成中文 */
[data-testid="stFileUploaderDropzone"] {
    background: var(--fsts-primary-soft); border: 1.5px dashed #9DB6CC; border-radius: 0.7rem;
    padding: 1.1rem 1rem; display: flex; align-items: center; justify-content: space-between; gap: 1rem;
}
[data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
[data-testid="stFileUploaderDropzone"]::before {
    content: "把文件拖到这里，或点右侧按钮选择"; font-size: 0.92rem; color: var(--fsts-ink);
    white-space: nowrap; flex-shrink: 0;
}
[data-testid="stFileUploaderDropzone"] > span > button [data-testid="stMarkdownContainer"],
[data-testid="stFileUploaderDropzone"] > span > button [data-testid="stIconMaterial"] { display: none; }
[data-testid="stFileUploaderDropzone"] > span > button::after {
    content: "选择文件"; font-size: 0.875rem; white-space: nowrap;
}

/* 头部 */
.fsts-header { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; margin: 0.15rem 0 0.2rem; }
.fsts-header-note { color: var(--fsts-muted); font-size: 0.86rem; }

/* 状态标签 */
.fsts-pill { display: inline-flex; align-items: center; gap: 0.32rem; padding: 0.14rem 0.6rem;
    border-radius: 999px; font-size: 0.78rem; font-weight: 600; white-space: nowrap;
    background: var(--fsts-primary-soft); color: var(--fsts-primary); }
.fsts-pill.ok { background: var(--fsts-ok-soft); color: var(--fsts-ok); }
.fsts-pill.warn { background: var(--fsts-warn-soft); color: var(--fsts-warn); }
.fsts-pill.bad { background: var(--fsts-bad-soft); color: var(--fsts-bad); }
.fsts-pill.muted { background: #EEF1F4; color: var(--fsts-muted); }

/* 步骤标题 */
.fsts-step { display: flex; align-items: center; gap: 0.5rem; font-weight: 700; font-size: 0.98rem;
    color: var(--fsts-ink); margin-bottom: 0.45rem; }
.fsts-step-no { display: inline-flex; width: 1.45rem; height: 1.45rem; align-items: center;
    justify-content: center; border-radius: 50%; background: var(--fsts-primary); color: #FFF;
    font-size: 0.8rem; font-weight: 700; }

/* 键值表 */
.fsts-kv { display: grid; grid-template-columns: 6.5rem 1fr; gap: 0.3rem 0.8rem; font-size: 0.87rem; }
.fsts-kv dt { color: var(--fsts-muted); }
.fsts-kv dd { margin: 0; color: var(--fsts-ink); overflow-wrap: anywhere; }

/* 结论条 */
.fsts-verdict { border: 1px solid var(--fsts-line); border-left: 4px solid var(--fsts-primary);
    background: var(--fsts-surface); border-radius: 0.55rem; padding: 0.7rem 0.95rem; margin-bottom: 0.5rem; }
.fsts-verdict.ok { border-left-color: var(--fsts-ok); }
.fsts-verdict.warn { border-left-color: var(--fsts-warn); }
.fsts-verdict.bad { border-left-color: var(--fsts-bad); }
.fsts-verdict.muted { border-left-color: #AEB9C4; }
.fsts-verdict-title { font-size: 0.8rem; color: var(--fsts-muted); }
.fsts-verdict-value { font-size: 1.02rem; font-weight: 700; margin: 0.12rem 0; color: var(--fsts-ink); }
.fsts-verdict-note { font-size: 0.84rem; color: var(--fsts-muted); line-height: 1.5; }
</style>
"""


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def pill(text: str, tone: str = "") -> str:
    return f'<span class="fsts-pill {escape(tone)}">{escape(text)}</span>'


def render_html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def step_title(number: int, text: str) -> None:
    render_html(
        f'<div class="fsts-step"><span class="fsts-step-no">{number}</span>{escape(text)}</div>'
    )


def key_values(rows: list[tuple[str, str]]) -> None:
    body = "".join(
        f"<dt>{escape(str(key))}</dt><dd>{escape(str(value))}</dd>" for key, value in rows
    )
    render_html(f'<dl class="fsts-kv">{body}</dl>')


def verdict_card(title: str, value: str, note: str = "", tone: str = "") -> None:
    render_html(
        f'<div class="fsts-verdict {escape(tone)}">'
        f'<div class="fsts-verdict-title">{escape(title)}</div>'
        f'<div class="fsts-verdict-value">{escape(value)}</div>'
        f'<div class="fsts-verdict-note">{escape(note)}</div></div>'
    )


__all__ = [
    "inject_theme",
    "key_values",
    "pill",
    "render_html",
    "step_title",
    "verdict_card",
]
