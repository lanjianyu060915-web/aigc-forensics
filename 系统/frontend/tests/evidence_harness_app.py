from __future__ import annotations

import os
from pathlib import Path
import sys


FRONTEND_ROOT = Path(__file__).resolve().parents[1]
if str(FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(FRONTEND_ROOT))

import streamlit as st

from components.data_loader import load_task_results
from components.evidence_dashboard import render_evidence_workspace


task_dir = Path(os.environ["FSTS_TEST_TASK_DIR"])
identity_root = Path(os.environ["FSTS_TEST_IDENTITY_ROOT"])
source_root = Path(os.environ["FSTS_TEST_SOURCE_ROOT"])
sample_root_value = os.environ.get("FSTS_TEST_SAMPLE_ROOT", "").strip()
sample_roots_value = os.environ.get("FSTS_TEST_SAMPLE_ROOTS", "").strip()
sample_roots = tuple(
    Path(value)
    for value in (
        ([sample_root_value] if sample_root_value else [])
        + ([item for item in sample_roots_value.split(os.pathsep) if item] if sample_roots_value else [])
    )
)
results = load_task_results(
    task_dir,
    identity_roots=(identity_root,),
    source_roots=(source_root,),
    sample_roots=sample_roots,
)

st.set_page_config(page_title="证据可视化测试", layout="wide")
render_evidence_workspace(
    deepfake=results.deepfake_result,
    identity=results.identity_result,
    source=results.source_result,
    policy=results.path_policy,
    task_dir=results.task_dir,
    task_status=results.task_status,
    final_report=results.final_report,
    task_artifact_manifest=results.artifact_manifest,
    product_version="0.4.0-test",
    key_prefix="harness",
)
