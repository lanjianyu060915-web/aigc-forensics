"""真实来源页展示配对全局失败，同时保留素材候选。"""

import os
from pathlib import Path
import sys

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

# 复制到系统前端测试目录后可直接运行；外部审查通过环境指定只读源码根。
系统目录 = Path(os.environ.get('AIGC_SYSTEM_ROOT', Path(__file__).resolve().parents[2]))
for 目录 in (系统目录, 系统目录 / 'frontend'):
    sys.path.insert(0, str(目录))


def 来源界面(tmp_path, 配对结果):
    图像 = tmp_path / '候选底图.png'
    Image.new('RGB', (24, 24), (90, 120, 150)).save(图像)
    素材结果 = {'status': 'success', 'query_image': str(图像),
              'candidates': [{'rank': 1, 'file_path': str(图像), 'similarity': 0.8}]}
    脚本 = (
        'from pathlib import Path\nimport streamlit as st\n'
        'from components.evidence_dashboard import render_evidence_workspace\n'
        'from components.path_security import ResourcePathPolicy\n'
        f'root = Path({str(tmp_path)!r})\n'
        "st.session_state['pair_status_page'] = '原始素材溯源'\n"
        f'data = {素材结果!r}\n'
        f'final_report = {{"pair_compare": {配对结果!r}}}\n'
        'render_evidence_workspace(deepfake={}, identity={}, source=data, '
        'policy=ResourcePathPolicy(task_roots=(root,)), task_dir=root, '
        "final_report=final_report, key_prefix='pair_status')\n"
    )
    应用 = AppTest.from_string(脚本).run(timeout=20)
    assert not 应用.exception
    assert any('候选底图.png' in 项.value for 项 in 应用.markdown)
    return 应用


@pytest.mark.parametrize('状态', ['error', 'failed', 'partial_success'])
def test_pair_global_failure_visible_without_hiding_source(tmp_path, 状态):
    私有路径 = tmp_path / '私有目录' / '模型文件.onnx'
    应用 = 来源界面(tmp_path, {'status': 状态, 'candidates': [],
                         'error_message': '初始化失败：' + str(私有路径)})
    提示 = '\n'.join(项.value for 项 in 应用.warning)
    assert '配对比对未完成' in 提示 and '初始化失败' in 提示
    assert str(tmp_path) not in 提示 and '私有目录' not in 提示
    assert any('其他模块的结果不受影响' in 项.value for 项 in 应用.caption)


def test_pair_skipped_is_distinct_from_failure(tmp_path):
    应用 = 来源界面(tmp_path, {'status': 'skipped', 'candidates': [], 'notice': '没有可用素材候选'})
    assert any('配对比对未运行' in 项.value for 项 in 应用.info)
    assert not any('配对比对未完成' in 项.value for 项 in 应用.warning)


def test_pair_missing_status_keeps_legacy_empty_layout(tmp_path):
    应用 = 来源界面(tmp_path, {'candidates': []})
    提示 = '\n'.join(项.value for 项 in [*应用.info, *应用.warning, *应用.error])
    assert '配对比对未完成' not in 提示 and '配对比对未运行' not in 提示
    assert any('本次没有可展示的候选配对' in 项.value for 项 in 应用.caption)
