"""Deterministic explainability and report generation for task artifacts."""

from .explainability import (
    build_deepfake_explanation,
    build_explanations,
    build_identity_explanation,
    build_source_explanation,
    display_name,
    redact_text,
)
from .report_builder import (
    DEFAULT_DISCLAIMER,
    DEFAULT_PRODUCT_NAME,
    build_final_report,
    generate_task_reports,
    report_file_names,
    render_html_report,
    render_text_report,
)

__all__ = [
    "DEFAULT_DISCLAIMER",
    "DEFAULT_PRODUCT_NAME",
    "build_deepfake_explanation",
    "build_explanations",
    "build_final_report",
    "build_identity_explanation",
    "build_source_explanation",
    "display_name",
    "generate_task_reports",
    "redact_text",
    "report_file_names",
    "render_html_report",
    "render_text_report",
]
