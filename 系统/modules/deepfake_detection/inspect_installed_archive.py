from __future__ import annotations

import argparse
from pathlib import Path

from pyinstaller_pyz_runtime import CArchiveReader


def main() -> int:
    parser = argparse.ArgumentParser(description="只读列出已安装鉴伪 EXE 的 PyInstaller 条目。")
    parser.add_argument("--executable", required=True)
    parser.add_argument("--filter", default="")
    args = parser.parse_args()
    archive = CArchiveReader(Path(args.executable))
    pattern = args.filter.casefold()
    for name, entry in sorted(archive.toc.items()):
        if not pattern or pattern in name.casefold():
            print(f"C\t{entry[-1]}\t{name}")
    pyz = archive.open_pyz()
    for name, entry in sorted(pyz.toc.items()):
        if not pattern or pattern in name.casefold():
            print(f"P\t{entry[0]}\t{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
