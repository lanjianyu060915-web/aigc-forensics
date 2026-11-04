"""一键自检不能放过“证据图没写全”这类结果正确、证据残缺的情况。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def self_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # self_check 导入时会 setdefault 数据目录，先占位，避免影响其他测试。
    monkeypatch.setenv("FSTS_DATA_ROOT", str(tmp_path / "data"))
    spec = importlib.util.spec_from_file_location("self_check_under_test", ROOT / "tools" / "self_check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_path_limit_leaves_room_for_longest_file(self_check) -> None:
    longest_relative_path = 189
    assert self_check.MAX_ROOT_PATH_LENGTH + longest_relative_path < 260


def test_root_path_problem_only_above_limit(self_check) -> None:
    limit = self_check.MAX_ROOT_PATH_LENGTH
    assert self_check.root_path_problem(Path("D:\\" + "a" * (limit - 3))) == ""
    assert "短路径" in self_check.root_path_problem(Path("D:\\" + "a" * (limit - 2)))


def test_identity_evidence_warnings_are_collected(self_check, tmp_path: Path) -> None:
    assert self_check.identity_evidence_warnings(tmp_path) == []
    evidence = tmp_path / "visualizations" / "identity" / "run_1" / "identity_evidence.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(json.dumps({"warnings": []}), encoding="utf-8")
    assert self_check.identity_evidence_warnings(tmp_path) == []
    evidence.write_text(json.dumps({"warnings": ["候选卡片最佳参考图生成失败"]}), encoding="utf-8")
    assert self_check.identity_evidence_warnings(tmp_path) == ["候选卡片最佳参考图生成失败"]
