"""AST-only state-transition tests; no app import, project data or model load."""
from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
import ast

import pytest


class RerunRequested(BaseException):
    pass


class ResultDataError(ValueError):
    pass


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}
        self.calls = []
        self.fragment_calls = []
        self.reruns = []

    def fragment(self, *, run_every):
        def decorate(function):
            @wraps(function)
            def invoke(*args, **kwargs):
                self.fragment_calls.append((function.__name__, run_every))
                return function(*args, **kwargs)
            return invoke
        return decorate

    def rerun(self, **kwargs):
        self.reruns.append(kwargs)
        raise RerunRequested()

    def __getattr__(self, name):
        if name not in {"info", "error", "success", "caption", "progress", "dataframe", "download_button"}:
            raise AssertionError("Unexpected UI action: " + name)
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record


class NoFileReads:
    def __init__(self, value):
        self.value = value

    def is_file(self):
        return False


@pytest.fixture
def ui():
    source = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    names = {"_job_needs_polling", "_stop_job_polling_once",
             "_render_warm_status", "_poll_warm_status", "_render_warm_status_content",
             "_render_active_job", "_poll_active_job", "_render_active_job_content",
             "_render_batch_job", "_poll_batch_job", "_render_batch_job_content"}
    nodes = [node for node in ast.parse(source.read_text("utf-8")).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    st = FakeStreamlit()
    jobs, loads = {}, []
    def load(*args, **kwargs):
        loads.append((args, kwargs))
        if state["load_error"]:
            raise ResultDataError("synthetic invalid result")
        st.session_state["loaded_results"] = {"mode": "Live Analysis", "fixture": "new synthetic result"}
    state = {"st": st, "jobs": jobs, "loads": loads, "load_error": False}
    namespace = {"st": st, "Mapping": Mapping, "Any": object, "Path": NoFileReads,
                 "time": SimpleNamespace(monotonic=lambda: 10), "MODULE_KEYS": ("deepfake",),
                 "ResultDataError": ResultDataError, "_job_snapshot": lambda job_id: jobs.get(job_id),
                 "_load_task": load, "render_live_progress": lambda **kwargs: st.calls.append(("live_progress", (), kwargs)),
                 "status_label": str, "display_cell": lambda key, value: value,
                 "summaries_to_csv": lambda summaries: b"synthetic,csv\n"}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(source), "exec"), namespace)
    state["functions"] = namespace
    return state


PAIRS = [
    ("warm", "_render_warm_status", "_poll_warm_status", 2.0),
    ("live", "_render_active_job", "_poll_active_job", 0.5),
    ("batch", "_render_batch_job", "_poll_batch_job", 1.0),
]


def set_job(ui, kind, status="queued", **values):
    identifier = kind + "_synthetic_1"
    ui["st"].session_state[kind + "_job_id"] = identifier
    ui["jobs"][identifier] = {"job_id": identifier, "status": status, **values}
    return ui["jobs"][identifier]


@pytest.mark.parametrize("kind,wrapper,poll,interval", PAIRS)
def test_no_job_never_registers_polling(ui, kind, wrapper, poll, interval):
    ui["functions"][wrapper]()
    assert ui["st"].fragment_calls == [] and ui["st"].reruns == [] and ui["loads"] == []


@pytest.mark.parametrize("kind,wrapper,poll,interval", PAIRS)
def test_queued_running_terminal_refreshes_app_once_then_static(ui, kind, wrapper, poll, interval):
    job = set_job(ui, kind)
    if kind == "live":
        job.update(task_dir="synthetic/task", result={"status": "success"})
    ui["functions"][wrapper]()
    assert ui["st"].fragment_calls[-1] == (poll, interval)
    job["status"] = "running"
    ui["functions"][poll]()
    assert not ui["st"].reruns
    job["status"] = "success"
    with pytest.raises(RerunRequested):
        ui["functions"][poll]()
    assert ui["st"].reruns == [{"scope": "app"}]
    fragments = len(ui["st"].fragment_calls)
    for _ in range(3):
        ui["functions"][wrapper]()
    assert len(ui["st"].fragment_calls) == fragments
    assert len(ui["st"].reruns) == 1 and len(ui["loads"]) == (1 if kind == "live" else 0)
    # A late timer invocation after the completed transition cannot rerun again.
    ui["functions"][poll]()
    assert len(ui["st"].reruns) == 1


@pytest.mark.parametrize("kind,wrapper,poll,interval", PAIRS)
def test_terminal_error_stops_polling_without_result_load(ui, kind, wrapper, poll, interval):
    job = set_job(ui, kind)
    ui["functions"][wrapper]()
    job.update(status="error", error="synthetic failure")
    with pytest.raises(RerunRequested):
        ui["functions"][poll]()
    fragments = len(ui["st"].fragment_calls)
    ui["functions"][wrapper]()
    ui["functions"][wrapper]()
    assert len(ui["st"].fragment_calls) == fragments and len(ui["st"].reruns) == 1
    assert not ui["loads"]


@pytest.mark.parametrize("kind,wrapper,poll,interval", PAIRS)
def test_disappearing_job_stops_existing_timer_once(ui, kind, wrapper, poll, interval):
    set_job(ui, kind)
    ui["functions"][wrapper]()
    ui["jobs"].clear()
    with pytest.raises(RerunRequested):
        ui["functions"][poll]()
    ui["functions"][wrapper]()
    ui["functions"][poll]()
    assert len(ui["st"].reruns) == 1


def test_bad_live_result_load_is_cached_per_job_and_new_job_can_load(ui):
    job = set_job(ui, "live", task_dir="synthetic/task", result={"status": "partial_success"})
    ui["load_error"] = True
    ui["functions"]["_render_active_job"]()
    job["status"] = "partial_success"
    with pytest.raises(RerunRequested):
        ui["functions"]["_poll_active_job"]()
    for _ in range(3):
        ui["functions"]["_render_active_job"]()
    assert len(ui["loads"]) == 1 and len(ui["st"].reruns) == 1
    assert any(call[0] == "error" and "synthetic invalid result" in call[1][0] for call in ui["st"].calls)
    ui["load_error"] = False
    identifier = "live_synthetic_2"
    ui["st"].session_state["live_job_id"] = identifier
    ui["jobs"][identifier] = {**job, "job_id": identifier, "status": "success"}
    with pytest.raises(RerunRequested):
        ui["functions"]["_render_active_job"]()
    ui["functions"]["_render_active_job"]()
    assert len(ui["loads"]) == 2 and ui["st"].session_state["loaded_live_job_id"] == identifier


def test_already_completed_live_result_loads_once_without_poll(ui):
    set_job(ui, "live", status="success", task_dir="synthetic/task", result={"status": "success"})
    with pytest.raises(RerunRequested):
        ui["functions"]["_render_active_job"]()
    ui["functions"]["_render_active_job"]()
    assert len(ui["loads"]) == len(ui["st"].reruns) == 1
    assert ui["st"].fragment_calls == []


@pytest.mark.parametrize("status", ["success", "partial_success", "error"])
def test_terminal_batch_preserves_table_and_download_control(ui, status):
    set_job(ui, "batch", status=status, items=[{"name": "synthetic.png", "status": "success",
        "summary": {"deepfake_label": "synthetic", "deepfake_score": 12}}])
    ui["functions"]["_render_batch_job"]()
    assert ui["st"].fragment_calls == [] and not ui["st"].reruns
    assert any(call[0] == "dataframe" for call in ui["st"].calls)
    download = next(call for call in ui["st"].calls if call[0] == "download_button")
    assert download[2]["key"] == "batch_csv" and download[2]["data"] == b"synthetic,csv\n"


def test_unknown_live_status_is_static_error_not_endless_queued_progress(ui):
    set_job(ui, "live", status=None)
    ui["functions"]["_render_active_job"]()
    assert any(call[0] == "error" for call in ui["st"].calls)
    assert not ui["st"].fragment_calls and not ui["loads"]
