"""双击入口的文件格式：写错了 cmd 和 PowerShell 不一定报错，只会解析出乱七八糟的命令。"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

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


@pytest.mark.parametrize("name", ["start.ps1", "tools/setup_runtime.ps1"])
def test_powershell_scripts_parse(name: str) -> None:
    # PowerShell 把中文引号当字符串定界符，写在双引号字符串里会直接解析失败。
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("没有 PowerShell")
    script = (
        "$errors = $null; [void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{ROOT / name}', [ref]$null, [ref]$errors); $errors | ForEach-Object {{ $_.Message }}; exit $errors.Count"
    )
    completed = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_setup_script_finds_its_own_directory_when_started_with_file() -> None:
    # 搭建环境.bat 就是这样启动的。无效组件会在开始下载之前报错，走不到这一步说明默认路径取不到。
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("没有 PowerShell")
    completed = subprocess.run(
        [
            powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(ROOT / "tools" / "setup_runtime.ps1"), "-Components", "nothing",
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=60,
    )
    output = completed.stdout + completed.stderr
    assert "Split-Path" not in output, output
    assert "nothing" in output, output
    assert completed.returncode != 0

