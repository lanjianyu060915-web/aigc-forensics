from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest


FRONTEND_ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path(__file__).with_name("evidence_harness_app.py")
RESULT_NAMES = (
    "detection_result.json",
    "identity_result.json",
    "source_result.json",
    "task_status.json",
    "unified_result.json",
    "final_report.json",
)
PAGES = (
    "研判总览",
    "AIGC鉴伪证据",
    "人脸身份检索",
    "原始底图溯源",
    "证据链与完整报告",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(task_dir: Path) -> dict[str, dict[str, int | str]]:
    return {
        name: {
            "sha256": _sha256(task_dir / name),
            "size_bytes": (task_dir / name).stat().st_size,
            "mtime_ns": (task_dir / name).stat().st_mtime_ns,
        }
        for name in RESULT_NAMES
        if (task_dir / name).is_file()
    }


def _assert_clean(app: AppTest) -> None:
    if app.exception:
        raise AssertionError([item.message for item in app.exception])
    if app.error:
        raise AssertionError([item.value for item in app.error])


def main() -> int:
    task_dir = Path(os.environ["FSTS_TEST_TASK_DIR"]).resolve()
    if str(FRONTEND_ROOT) not in sys.path:
        sys.path.insert(0, str(FRONTEND_ROOT))
    before = _snapshot(task_dir)
    sibling_names_before = sorted(path.name for path in task_dir.parent.iterdir())

    app = AppTest.from_file(str(HARNESS), default_timeout=90).run(timeout=90)
    _assert_clean(app)
    if len(app.radio) != 1:
        raise AssertionError(f"expected one page selector, got {len(app.radio)}")
    if tuple(app.radio[0].options) != PAGES:
        raise AssertionError(f"unexpected pages: {app.radio[0].options}")

    page_results = []
    for page in PAGES:
        app.radio[0].set_value(page).run(timeout=90)
        _assert_clean(app)
        after_page = _snapshot(task_dir)
        if after_page != before:
            raise AssertionError(f"switching to {page} changed authoritative result files")
        page_results.append(
            {
                "page": page,
                "exception_count": len(app.exception),
                "error_count": len(app.error),
            }
        )

    sibling_names_after_tabs = sorted(path.name for path in task_dir.parent.iterdir())
    if sibling_names_after_tabs != sibling_names_before:
        raise AssertionError("switching pages created or removed a task directory")

    # A fresh AppTest instance is the closest deterministic equivalent of a
    # browser refresh: no previous Streamlit session state is retained.
    refreshed = AppTest.from_file(str(HARNESS), default_timeout=90).run(timeout=90)
    _assert_clean(refreshed)
    if len(refreshed.radio) != 1 or tuple(refreshed.radio[0].options) != PAGES:
        raise AssertionError("existing task did not reload into the evidence workspace")
    if _snapshot(task_dir) != before:
        raise AssertionError("fresh-session reload changed authoritative result files")

    payload = {
        "status": "passed",
        "task_dir": str(task_dir),
        "task_result_snapshot": before,
        "page_switches": page_results,
        "page_switch_did_not_create_task": True,
        "fresh_session_reload": {
            "passed": True,
            "exception_count": len(refreshed.exception),
            "error_count": len(refreshed.error),
        },
    }
    output_value = os.environ.get("FSTS_TEST_OUTPUT", "").strip()
    if output_value:
        output_path = Path(output_value)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
