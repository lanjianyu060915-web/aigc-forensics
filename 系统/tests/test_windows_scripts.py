"""双击入口的文件格式：写错了 cmd 和 PowerShell 不一定报错，只会解析出乱七八糟的命令。"""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BATCH_FILES = [
    ROOT / "start.bat",
    ROOT / "run_app.bat",
    ROOT / "run_case.bat",
    ROOT.parent / "启动系统.bat",
    ROOT.parent / "运行自检.bat",
    ROOT.parent / "搭建环境.bat",
]


@pytest.mark.parametrize("path", BATCH_FILES, ids=lambda item: item.name)
def test_batch_files_use_crlf(path: Path) -> None:
    if not path.exists():
        pytest.skip("不在完整交付目录中")
    data = path.read_bytes()
    # 只有 LF 时，cmd 切到 65001 代码页后会读错位，chcp 都会被解析成 cp。
    assert data.count(b"\n") == data.count(b"\r\n"), f"{path.name} 必须全部使用 CRLF"
    assert not data.startswith(b"\xef\xbb\xbf"), f"{path.name} 不能带 BOM，否则第一行命令失效"


@pytest.mark.parametrize("name", ["start.ps1", "tools/setup_runtime.ps1"])
def test_powershell_launcher_has_bom(name: str) -> None:
    # Windows PowerShell 5.1 把无 BOM 的脚本按系统代码页读，中文会乱码甚至语法错误。
    assert (ROOT / name).read_bytes().startswith(b"\xef\xbb\xbf")
