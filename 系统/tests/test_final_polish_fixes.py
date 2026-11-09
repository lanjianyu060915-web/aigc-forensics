"""验收问题的输出、异常降级和复核依据回归。"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))
from reporting import pdf_report
from reporting.report_builder import generate_task_reports, render_text_report, render_html_report
from reporting.review_records import append_review_record, review_report_mismatch
from pair_integration import pair_report_lines
from test_pdf_report import sample_report


@pytest.mark.parametrize('status,label', [('error', '配对比对未完成'), ('skipped', '配对比对未运行')])
def test_pair_failure_and_skip_reach_all_formats(tmp_path, monkeypatch, status, label):
    pair = {'status': status, 'error_message': r'规则失败 E:\private\user\rule.json',
            'notice': '没有素材候选', 'candidates': []}
    report = sample_report(tmp_path)
    report['pair_compare'] = pair
    for rendered in (render_text_report(report), render_html_report(report)):
        assert label in rendered
        assert r'E:\private' not in rendered
    captured = []
    old_p = pdf_report._p
    def capture(value, *args, **kwargs):
        captured.append(str(value))
        return old_p(value, *args, **kwargs)
    monkeypatch.setattr(pdf_report, '_p', capture)
    assert pdf_report.render_pdf_report(report, tmp_path).startswith(b'%PDF-')
    assert label in '\n'.join(captured)
    assert pair_report_lines({'status':'error','candidates':[]}) != pair_report_lines({'status':'skipped','candidates':[]})


def test_empty_pdf_cells_and_missing_values_are_different():
    assert pdf_report._text('') == ''
    assert pdf_report._text(None) == '未记录'
    table = pdf_report._table(['说明'], [[''], [None]], [200], pdf_report._styles())
    assert table._cellvalues[1][0].getPlainText() == ''
    assert table._cellvalues[2][0].getPlainText() == '未记录'


def test_punctuation_in_windows_path_does_not_leak_directories():
    assert 'private' not in pdf_report._text(r'见 E:\团队,2026\private\b.jpg 结束')
    assert pdf_report._text('https://example.org/a?x=1&y=2') == 'https://example.org/a?x=1&y=2'


def test_oversized_image_becomes_placeholder(tmp_path, monkeypatch):
    Image.new('RGB', (20, 20)).save(tmp_path / 'query.jpg')
    def bomb(*args, **kwargs):
        raise Image.DecompressionBombError('测试超大图片')
    monkeypatch.setattr(Image, 'open', bomb)
    result = pdf_report._image(tmp_path, 'query.jpg', 100, 100, pdf_report._styles())
    assert '图片无法读取' in result.getPlainText()


def test_review_hash_mismatch_is_visible_in_pdf(tmp_path, monkeypatch):
    result = generate_task_reports(tmp_path, task_id='复核依据')
    record = append_review_record(tmp_path, '复核员', '无法判断', '原意见')
    assert not review_report_mismatch(tmp_path, record)
    (tmp_path / 'final_report.json').write_text('{}', encoding='utf-8')
    assert review_report_mismatch(tmp_path, record)
    captured = []
    old_p = pdf_report._p
    def capture(value, *args, **kwargs):
        captured.append(str(value))
        return old_p(value, *args, **kwargs)
    monkeypatch.setattr(pdf_report, '_p', capture)
    pdf_report.render_pdf_report(result['report'], tmp_path, [record])
    assert '该条复核所依据的报告与当前报告不一致' in captured


def test_evidence_is_not_duplicated_in_text_or_html(tmp_path):
    report = sample_report(tmp_path)
    for text in (render_text_report(report), render_html_report(report)):
        assert text.count('档位证据强度') == 1


def test_source_method_and_limitations_are_readable(tmp_path):
    report = sample_report(tmp_path)
    report['modules']['source']['result']['method'] = {'global_feature': '特征', 'local': '几何'}
    report['modules']['source']['result']['limitations'] = ['重复局限。', '重复局限']
    blocks = pdf_report._appendix(report, {}, {}, {}, pdf_report._styles())
    table = blocks[1]
    method = next(row[1].getPlainText() for row in table._cellvalues if row[0].getPlainText() == '素材检索方法')
    assert '具体参数见结构化报告' in method
    assert sum('重复局限' in x.getPlainText() for x in blocks if hasattr(x, 'getPlainText')) == 1


def test_damaged_reviews_keep_original_pdf_download(tmp_path):
    from streamlit.testing.v1 import AppTest
    generate_task_reports(tmp_path, task_id='损坏记录')
    (tmp_path / 'review_records.jsonl').write_text('{损坏', encoding='utf-8')
    app = AppTest.from_string('from components.report_views import render_task_reports\nrender_task_reports(' + repr(str(tmp_path)) + ')').run(timeout=30)
    assert not app.exception
    assert '导出技术明细报告（PDF）' in {x.proto.label for x in app.get('download_button')}
    assert app.button[1].disabled


def test_review_validation_preserves_input_and_success_clears(tmp_path):
    from streamlit.testing.v1 import AppTest
    generate_task_reports(tmp_path, task_id='表单保留')
    app = AppTest.from_string('from components.report_views import render_task_reports\nrender_task_reports(' + repr(str(tmp_path)) + ')').run(timeout=30)
    app.text_input[0].set_value('复核员')
    next(x for x in app.button if x.label == '追加保存复核意见').click().run()
    assert app.text_input[0].value == '复核员'
    app.text_area[0].set_value('补充意见')
    next(x for x in app.button if x.label == '追加保存复核意见').click().run()
    assert not app.exception
    assert app.text_input[0].value == '' and app.text_area[0].value == ''


@pytest.mark.parametrize("method", [
    {"name": "face-masked global retrieval plus SIFT geometric verification",
     "global_features": ["HSV", "grayscale", "edge", "border"],
     "local_features": "SIFT (nfeatures=2500)"},
    "face-masked global retrieval plus SIFT geometric verification",
])
def test_real_source_method_name_is_chinese_in_pdf_appendix(tmp_path, method):
    report = sample_report(tmp_path)
    report["modules"]["source"]["result"]["method"] = method
    table = pdf_report._appendix(report, {}, {}, {}, pdf_report._styles())[1]
    rendered = next(row[1].getPlainText() for row in table._cellvalues
                    if row[0].getPlainText() == "素材检索方法")
    assert rendered == "遮蔽人脸后的全局特征检索与 SIFT 局部几何复核"
    assert "{" not in rendered and "face-masked" not in rendered
    assert report["modules"]["source"]["result"]["method"] == method
