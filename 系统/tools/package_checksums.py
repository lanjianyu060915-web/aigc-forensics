# -*- coding: utf-8 -*-
"""生成或校验交付目录的 SHA-256 清单。

  python package_checksums.py          重新生成 SHA256SUMS.csv
  python package_checksums.py --verify 按清单检查文件是否缺失或被改动

“用户数据”会随使用变化，不进清单；检查时也不报告清单外新增的文件。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "SHA256SUMS.csv"
EXCLUDED_TOP_LEVEL = {"用户数据", MANIFEST.name}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def iter_package_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] in EXCLUDED_TOP_LEVEL or "__pycache__" in relative.parts and "python_runtime" not in relative.parts:
            continue
        yield path, relative.as_posix()


def write_manifest(root: Path, manifest: Path) -> int:
    count = 0
    with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "size_bytes", "sha256"])
        for path, relative in iter_package_files(root):
            writer.writerow([relative, path.stat().st_size, sha256_of(path)])
            count += 1
    return count


def verify_manifest(root: Path, manifest: Path) -> list[str]:
    problems: list[str] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            path = root / row["path"]
            if not path.is_file():
                problems.append(f"缺失：{row['path']}")
            elif sha256_of(path) != row["sha256"]:
                problems.append(f"被改动：{row['path']}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify:
        print(f"已生成 {MANIFEST.name}：{write_manifest(ROOT, MANIFEST)} 个文件")
        return 0
    problems = verify_manifest(ROOT, MANIFEST)
    for problem in problems[:50]:
        print(problem)
    print(f"校验结束：{len(problems)} 个问题")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
