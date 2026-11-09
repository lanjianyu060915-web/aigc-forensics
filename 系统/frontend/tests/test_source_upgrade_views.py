"""实际前端组件的滑动对比与案件库检查交互。"""

from pathlib import Path
import sys

for root in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1]):
    sys.path.insert(0, str(root))

from PIL import Image
from streamlit.testing.v1 import AppTest


def test_pair_slider_and_frame_notice(tmp_path):
    for name, color in (("query", "red"), ("reference", "blue"), ("aligned", "green")):
        Image.new("RGB", (32, 32), color).save(tmp_path / f"{name}.png")
    pair = {"scope": "frame", "timestamp_seconds": 1.5, "conclusion_enabled": False,
            "candidates": [{"rank": 1, "status": "success", "candidate_name": "候选图",
                            "comparison": {"code": "face_replaced", "alignment": {"inliers": 12}, "faces": []},
                            "artifacts": {name: f"{name}.png" for name in ("query", "reference", "aligned")}}]}
    script = (
        "from pathlib import Path\n"
        "from components.pair_views import render_pair_comparison\n"
        "from components.path_security import ResourcePathPolicy\n"
        f"render_pair_comparison({pair!r}, ResourcePathPolicy(task_roots=(Path({str(tmp_path)!r}),)), '验收')\n"
    )
    app = AppTest.from_string(script).run(timeout=20)
    assert not app.exception
    assert len(app.slider) == 1
    assert "首要帧" in app.warning[0].value
    app.slider[0].set_value(70).run(timeout=20)
    assert not app.exception
    assert "face_replaced" not in "\n".join(item.value for item in app.markdown)


def test_legacy_pair_omission_does_not_break_page():
    app = AppTest.from_string(
        "from components.pair_views import render_pair_comparison\n"
        "from components.path_security import ResourcePathPolicy\n"
        "render_pair_comparison(None, ResourcePathPolicy(), '旧任务')\n"
    ).run(timeout=20)
    assert not app.exception
    assert not app.subheader


def test_case_library_check_button_shows_counts(tmp_path):
    config = {"identity": {"model_root": "模型", "det_size_chain": [1280]},
              "source": {}, "runtime": {}}
    item = {"file_count": 3, "invalid_count": 1, "duplicate_count": 1,
            "no_face_count": 0, "usable_count": 2, "files": [], "duplicates": []}
    script = (
        "from pathlib import Path\n"
        "from components import case_library_views as views\n"
        "from unittest.mock import patch\n"
        f"with patch.object(views, 'check_case_libraries', return_value={{'source': {item!r}, 'identity': {item!r}}}):\n"
        f"    views.render_case_library_settings('身份库', '素材库', {config!r}, Path({str(tmp_path / 'config.json')!r}))\n"
    )
    app = AppTest.from_string(script).run(timeout=20)
    assert not app.exception
    app.button(key="case_library_check").click().run(timeout=20)
    assert not app.exception
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value.iloc[0]["无法解码"] == 1


def test_case_library_check_failure_is_visible(tmp_path):
    config = {"identity": {"model_root": "模型"}, "source": {}, "runtime": {}}
    script = (
        "from pathlib import Path\n"
        "from components import case_library_views as views\n"
        "from unittest.mock import patch\n"
        "with patch.object(views, 'check_case_libraries', side_effect=ValueError('资源引用越出库根目录')):\n"
        f"    views.render_case_library_settings('身份库', '素材库', {config!r}, Path({str(tmp_path / 'config.json')!r}))\n"
    )
    app = AppTest.from_string(script).run(timeout=20)
    app.button(key="case_library_check").click().run(timeout=20)
    assert not app.exception
    assert "越出" in app.error[0].value
