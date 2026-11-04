"""Frontend import bridge for the pure reporting explainability layer."""

from __future__ import annotations

from pathlib import Path
import sys


PRODUCT_ROOT = Path(__file__).resolve().parents[2]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))

from reporting.explainability import (  # noqa: E402
    build_deepfake_explanation,
    build_explanations,
    build_identity_explanation,
    build_source_explanation,
    display_name,
    explain_deepfake,
    explain_identity,
    explain_source,
    redact_text,
)

__all__ = [
    "build_deepfake_explanation",
    "build_explanations",
    "build_identity_explanation",
    "build_source_explanation",
    "display_name",
    "explain_deepfake",
    "explain_identity",
    "explain_source",
    "redact_text",
]
