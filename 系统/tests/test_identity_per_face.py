"""逐脸附加证据：候选隔离、原图坐标、主结果不变与报告边界。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "modules/identity_search", ROOT / "modules/identity_search_v2", ROOT / "frontend"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from identity_search.errors import MultipleFacesError, NoFaceError
from scale_fallback import attempt_scale_chain
from decision.open_set import decide_identity_open_set
from per_face import MAX_FACES
from reporting.report_builder import build_final_report, render_html_report, render_text_report
from reporting.pdf_report import render_pdf_report
from reporting.review_records import append_review_record
from orchestrator import _task_artifact_manifest


class ColorSearcher:
    """把拼图内的彩色方块作为脸的替身，实际读取遮挡后的查询图。"""
    def __init__(self, query, faces, fail_index=None):
        self.query, self.faces, self.fail_index = query, faces, fail_index
        self.calls = []
        self.extractor = None

    def search(self, query, top_k, progress_reporter=None, **kwargs):
        self.calls.append((str(query), top_k, kwargs))
        if Path(query) == self.query and len(self.faces) > 1:
            raise MultipleFacesError("多人脸", diagnostics={"detected_faces": self.faces})
        image = cv2.imdecode(np.fromfile(str(query), dtype=np.uint8), cv2.IMREAD_COLOR)
        surviving = []
        for index, face in enumerate(self.faces):
            x1, y1, x2, y2 = face["bbox"]
            if tuple(image[(y1 + y2) // 2, (x1 + x2) // 2]) == (20 + index * 20, 10, 200):
                surviving.append(index)
        assert len(surviving) == 1
        index = surviving[0]
        if index == self.fail_index:
            raise NoFaceError("测试脸失败，路径不应进入错误结果")
        return {
            "status": "success", "query_image": str(query), "method": {}, "warnings": [],
            "metrics": {"query_face_count": 1, "top1_top2_margin": 0.3},
            "diagnostics": {"detected_faces": [self.faces[index]]},
            "candidates": [{"rank": 1, "identity_id": f"测试{index}", "identity_name": f"候选{index}", "score": 0.75, "mean_similarity": 0.75}],
        }


def scene(tmp_path, count):
    image = np.full((180, max(250, count * 110), 3), 250, dtype=np.uint8)
    faces = []
    for index in range(count):
        # 序号 0 的脸最小，确保面积排序和检测顺序不同。
        side = 30 + index * 8
        x1, y1 = 15 + index * 110, 15
        faces.append({"bbox": [x1, y1, x1 + side, y1 + side], "det_score": 0.99 - index * 0.01})
        image[y1:y1 + side, x1:x1 + side] = (20 + index * 20, 10, 200)
    query = tmp_path / "查询.png"
    query.write_bytes(cv2.imencode(".png", image)[1].tobytes())
    return query, faces


def run_faces(query, faces, output, *, include=True, fail_index=None):
    searcher = ColorSearcher(query, faces, fail_index)
    result, _ = attempt_scale_chain(searcher, str(query), (1280,), lambda scale: None,
                                    top_k=5, multi_face_policy="largest_face", main_face_dir=output,
                                    include_secondary_faces=include, artifact_root=output)
    return result, searcher


def test_small_face_has_own_candidate_and_main_values_are_identical(tmp_path):
    query, faces = scene(tmp_path, 2)
    old, _ = run_faces(query, faces, tmp_path / "旧", include=False)
    current, searcher = run_faces(query, faces, tmp_path / "新")
    for key in old:
        assert current[key] == old[key], key
    assert [face["face_index"] for face in current["faces"]] == [1, 0]
    assert current["faces"][1]["candidates"][0]["identity_id"] == "测试0"
    assert current["faces"][1]["query_bbox_original"] == faces[0]["bbox"]
    assert searcher.calls[-1][1] == 3 and "artifact_root" not in searcher.calls[-1][2]
    assert current["faces"][0]["open_set"] == decide_identity_open_set(old)
    assert "identity_binding" not in json.dumps(current)
    for face in current["faces"]:
        relative = Path(face["face_thumbnail"])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert (tmp_path / "新" / relative).is_file()


def test_seven_faces_are_capped_and_individual_failure_is_isolated(tmp_path):
    query, faces = scene(tmp_path, 7)
    result, searcher = run_faces(query, faces, tmp_path / "task", fail_index=5)
    assert result["status"] == "success"
    assert len(result["faces"]) == MAX_FACES == 3
    assert len(searcher.calls) == MAX_FACES + 1  # 原图诊断、主脸、其余两张。
    assert result["faces"][0]["face_index"] == 6
    failed = next(row for row in result["faces"] if row["face_index"] == 5)
    assert failed["status"] == "error" and failed["open_set"]["state"] == "unavailable"
    assert "路径" not in failed["error_message"]
    assert sum(row["status"] == "success" for row in result["faces"]) == MAX_FACES - 1
    assert "7 张脸" in result["face_search"]["limitations"][-1]


def test_single_face_does_not_search_again(tmp_path):
    query, faces = scene(tmp_path, 1)
    result, searcher = run_faces(query, faces, tmp_path / "task")
    assert len(searcher.calls) == 1 and len(result["faces"]) == 1


def test_thumbnail_failure_keeps_candidates_and_module_status(tmp_path, monkeypatch):
    import per_face
    query, faces = scene(tmp_path, 2)
    monkeypatch.setattr(per_face, "_thumbnail", lambda *args: (_ for _ in ()).throw(OSError("失败")))
    result, _ = run_faces(query, faces, tmp_path / "task")
    assert result["status"] == "success"
    assert all(row["status"] == "success" and row["candidates"] and row["thumbnail_error"] for row in result["faces"])


def test_optional_stage_failure_does_not_invalidate_main_result(tmp_path, monkeypatch):
    import per_face
    query, faces = scene(tmp_path, 2)
    monkeypatch.setattr(per_face, "_append_face_candidates", lambda *args: (_ for _ in ()).throw(RuntimeError("失败")))
    result, _ = run_faces(query, faces, tmp_path / "task")
    assert result["status"] == "success" and result["candidates"]
    assert result["face_search"]["status"] == "error"


def test_reports_and_manifest_include_faces_and_review_can_select_small_face(tmp_path):
    query, faces = scene(tmp_path, 2)
    result, _ = run_faces(query, faces, tmp_path)
    result["faces"][1]["candidates"][0]["identity_name"] = "候选<b>&"
    report = build_final_report("逐脸报告测试", identity_result=result)
    text = render_text_report(report)
    html = render_html_report(report)
    assert "逐脸身份候选" in text and "候选<b>&" in text
    assert "候选&lt;b&gt;&amp;" in html
    assert "系统不判断哪张脸被改动" in text
    assert "尚未在逐脸场景专门验证" in text
    assert render_pdf_report(report, tmp_path).startswith(b"%PDF")
    manifest = _task_artifact_manifest(tmp_path, "逐脸报告测试", {"identity": result})
    thumbnails = [item for item in manifest["artifacts"] if item["kind"] == "identity_face_thumbnail"]
    assert len(thumbnails) == 2
    for item in thumbnails:
        assert hashlib.sha256((tmp_path / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    (tmp_path / "final_report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    record = append_review_record(tmp_path, "复核", "无法判断", "需结合来源", target_face_index=0)
    assert record["target_face_index"] == 0
    assert report.get("forgery_type") is None
    assert "identity_binding" not in json.dumps(report)


def test_overview_adds_hint_without_changing_original_results(tmp_path):
    from components.evidence_dashboard import build_key_findings
    query, faces = scene(tmp_path, 2)
    result, _ = run_faces(query, faces, tmp_path)
    result["faces"][0]["open_set"]["state"] = "insufficient_evidence"
    result["faces"][1]["open_set"]["state"] = "high_confidence"
    before = deepcopy(result)
    _, steps = build_key_findings({}, result, {})
    assert any("主脸本次未检出高可信身份候选" in step for step in steps)
    assert result == before


def test_video_only_primary_frame_extends_faces_and_aggregate_is_unchanged(tmp_path):
    from test_video_orchestration import VideoServices, _write_config
    from orchestrator import run_task

    class Services(VideoServices):
        def __init__(self):
            super().__init__()
            self.options = []

        def run_identity(self, **kwargs):
            payload = super().run_identity(**kwargs)
            enabled = kwargs["include_secondary_faces"]
            self.options.append(enabled)
            if enabled:
                root = Path(kwargs["output_path"]).parent
                root.mkdir(parents=True, exist_ok=True)
                query, faces = scene(root, 2)
                extra, _ = run_faces(query, faces, root)
                payload.update({key: extra[key] for key in ("faces", "face_search", "artifact_manifest")})
            return payload

    config = _write_config(tmp_path)
    video = tmp_path / "query.mp4"
    video.write_bytes("测试视频替身".encode("utf-8"))
    services = Services()
    status = run_task(video, config_path=config, services=services, selected_modules=["identity"])
    task = Path(status["task_dir"])
    identity = json.loads((task / "identity_result.json").read_text(encoding="utf-8"))
    assert services.options == [True, False, False]
    assert identity["face_search"]["scope"] == "video_primary_frame"
    assert identity["candidates"][0]["mean_similarity"] == 0.5
    assert identity["candidates"][0]["frame_support"] == 3
    assert all((task / face["face_thumbnail"]).is_file() for face in identity["faces"])
    manifest = json.loads((task / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert sum(row["kind"] == "identity_face_thumbnail" for row in manifest["artifacts"]) == 2


def test_face_view_renders_candidates_and_rejects_thumbnail_escape(tmp_path):
    from streamlit.testing.v1 import AppTest
    query, faces = scene(tmp_path, 2)
    result, _ = run_faces(query, faces, tmp_path)
    result["faces"][0]["face_thumbnail"] = "../不在任务内.png"
    data_path = tmp_path / "view.json"
    data_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    app = AppTest.from_string(f'''
import json
from pathlib import Path
from components.result_views import _render_identity_faces
from components.path_security import ResourcePathPolicy
root = Path({str(tmp_path)!r})
data = json.loads((root / "view.json").read_text(encoding="utf-8"))
_render_identity_faces(data, ResourcePathPolicy(task_roots=(root,)))
''').run(timeout=20)
    assert not app.exception
    assert any("脸 1（主脸）：缩略图不可用" in item.value for item in app.caption)
    assert any("候选0" in item.value for item in app.text)
    assert any("候选1" in item.value for item in app.text)
