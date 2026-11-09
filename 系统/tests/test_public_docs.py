"""公开文档的生成完整性及证据边界。"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def test_public_documents_have_no_unfilled_tokens():
    for path in [ROOT / "README.md", *sorted((ROOT / "文档").glob("*.md"))]:
        assert not re.search(r"\{\{[A-Z_]+\}\}|@@[A-Z0-9_]+@@", path.read_text(encoding="utf-8")), path.name


def test_dashboard_does_not_treat_background_as_direct_swap_evidence():
    text = (ROOT / "系统/frontend/components/evidence_dashboard.py").read_text(encoding="utf-8")
    assert "是换脸的直接线索" not in text
    assert "也可能是同一地点的不同人" in text
