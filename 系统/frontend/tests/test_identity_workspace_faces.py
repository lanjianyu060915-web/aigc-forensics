"""真实工作区路由展示逐脸候选，保留原主身份结果。"""

from copy import deepcopy
from pathlib import Path
import sys

for root in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1]):
    sys.path.insert(0, str(root))

from PIL import Image
from streamlit.testing.v1 import AppTest


def identity_result():
    main = {"rank": 1, "identity_id": "main_candidate", "score": 0.7,
            "mean_similarity": 0.7, "best_reference_score": 0.8,
            "valid_reference_count": 1}
    return {
        "status": "success", "candidates": [main],
        "metrics": {"query_face_count": 1, "face_detection_score": 0.9,
                    "embedding_dimension": 512, "top1_top2_margin": 0.2},
        "faces": [
            {"face_index": 0, "is_main_face": True, "area_rank": 1,
             "status": "success", "face_thumbnail": "face_0.png",
             "open_set": {"state_label": "证据不足"},
             "candidates": [dict(main, rank=i, identity_id=f"main_{i}") for i in range(1, 4)]},
            {"face_index": 1, "is_main_face": False, "area_rank": 2,
             "status": "success", "face_thumbnail": "face_1.png",
             "open_set": {"state_label": "当前库中未发现可靠候选"},
             "candidates": [{"rank": i, "identity_id": f"other_{i}", "score": 0.1 / i}
                            for i in range(1, 4)]},
        ],
        "face_search": {"limit": 3, "scope": "query_image"},
    }


def workspace_app(tmp_path, identity):
    for i in range(2):
        Image.new("RGB", (24, 24), (100 + i * 50, 80, 60)).save(tmp_path / f"face_{i}.png")
    script = (
        "from copy import deepcopy\n"
        "from pathlib import Path\n"
        "import streamlit as st\n"
        "from components.evidence_dashboard import render_evidence_workspace\n"
        "from components.path_security import ResourcePathPolicy\n"
        "st.session_state['actual_page'] = '人脸身份比对'\n"
        f"data = {identity!r}\n"
        "before = deepcopy(data)\n"
        f"root = Path({str(tmp_path)!r})\n"
        "selected = render_evidence_workspace(deepfake={}, identity=data, source={}, "
        "policy=ResourcePathPolicy(task_roots=(root,)), task_dir=root, key_prefix='actual')\n"
        "assert selected == '人脸身份比对'\n"
        "assert data == before, '展示不得修改主结果或逐脸数据'\n"
    )
    return AppTest.from_string(script).run(timeout=20)


def test_active_identity_workspace_shows_two_faces_before_main_evidence(tmp_path):
    app = workspace_app(tmp_path, identity_result())
    assert not app.exception
    assert app.radio(key="actual_page").value == "人脸身份比对"
    headings = [item.value for item in app.markdown]
    assert headings.index("### 逐脸身份候选") < headings.index("#### 1. 人脸检测、关键点与对齐")
    text = "\n".join(item.value for item in app.text)
    for label in ("main_1", "main_2", "main_3", "other_1", "other_2", "other_3"):
        assert label in text
    assert "脸 0（主脸）" in headings and "脸 1" in headings
    assert next(item.value for item in app.metric if item.label == "检测人脸数") == "1"
    assert any("系统不判断哪张脸被改动" in item.value for item in app.caption)
    assert any("尚未在逐脸场景专门验证" in item.value for item in app.caption)


def test_active_identity_workspace_shows_failed_face_without_hiding_main_result(tmp_path):
    result = identity_result()
    result["faces"][1].update(status="error", error_message="该脸未能提取有效特征", candidates=[], open_set={})
    app = workspace_app(tmp_path, result)
    assert not app.exception
    assert any("该脸未能提取有效特征" in item.value for item in app.caption)
    text = "\n".join(item.value for item in app.text)
    assert "main_1" in text and "other_1" not in text
    assert any("main_candidate" in item.value for item in app.markdown)
    assert any(item.label == "人员级平均" and item.value == "0.700000" for item in app.metric)


def test_active_identity_workspace_single_face_keeps_existing_main_layout(tmp_path):
    result = deepcopy(identity_result())
    result["faces"] = result["faces"][:1]
    app = workspace_app(tmp_path, result)
    assert not app.exception
    assert "### 逐脸身份候选" not in [item.value for item in app.markdown]
    assert any("Top-5 当前库内候选" in item.value for item in app.markdown)
    assert next(item.value for item in app.metric if item.label == "检测人脸数") == "1"
