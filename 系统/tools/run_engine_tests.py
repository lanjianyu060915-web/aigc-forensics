# -*- coding: utf-8 -*-
"""用引擎环境跑 engine/tests。

引擎环境按锁定清单安装，不含 pytest，这里借用界面环境里的 pytest。界面环境的路径排在最后，
numpy、torch 这些包仍然用引擎环境自己的版本。用法：
    engine_runtime\\python.exe -B tools\\run_engine_tests.py [pytest 参数]
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.append(str(ROOT / "python_runtime" / "Lib" / "site-packages"))
    import pytest

    return pytest.main([str(ROOT / "engine" / "tests"), "-q", "-p", "no:cacheprovider", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
