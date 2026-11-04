# -*- coding: utf-8 -*-
"""确认 GitHub 上有全部 LFS 文件：不下载内容，用批量接口逐个查 oid 和大小。

可以再抽几个小文件实际下载，核对 SHA-256。网络慢时用它代替整包下载。
    python tools/research/check_remote_lfs.py 仓库目录 [--sample N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

OWNER_REPO = "lanjianyu060915-web/aigc-forensics"
BATCH_URL = f"https://github.com/{OWNER_REPO}.git/info/lfs/objects/batch"
HEADERS = {"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"}


def pointers(repo: Path) -> dict[str, tuple[int, str]]:
    listed = subprocess.run(
        ["git", "-C", str(repo), "-c", "core.quotepath=false", "lfs", "ls-files", "--long", "--size"],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    objects: dict[str, tuple[int, str]] = {}
    for line in listed.splitlines():
        oid, _, rest = line.partition(" ")
        name = rest.split(" ", 1)[1].rsplit(" (", 1)[0]
        path = repo / name
        if path.is_file() and path.stat().st_size < 1024 and path.read_bytes().startswith(b"version https://git-lfs"):
            size = int(path.read_text(encoding="utf-8").split("size ")[1].split()[0])
        else:
            size = path.stat().st_size
        objects[oid] = (size, name)
    return objects


def batch(chunk: list[dict]) -> dict:
    body = json.dumps({"operation": "download", "transfers": ["basic"], "objects": chunk}).encode()
    request = urllib.request.Request(BATCH_URL, data=body, headers=HEADERS, method="POST")
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except OSError as exc:
            print(f"批量接口重试 {attempt + 1}：{exc}", flush=True)
            time.sleep(10)
    raise SystemExit("批量接口连不上")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("--sample", type=int, default=5, help="实际下载校验几个最小的文件")
    args = parser.parse_args()

    objects = pointers(args.repo)
    print(f"LFS 文件 {len(objects)} 个，共 {sum(size for size, _ in objects.values()) / 1e9:.2f} GB", flush=True)
    items = [{"oid": oid, "size": size} for oid, (size, _) in objects.items()]
    missing: list[str] = []
    hrefs: dict[str, dict] = {}
    for start in range(0, len(items), 100):
        for item in batch(items[start : start + 100])["objects"]:
            action = (item.get("actions") or {}).get("download")
            if item.get("error") or not action:
                missing.append(f"{objects[item['oid']][1]}：{item.get('error')}")
            else:
                hrefs[item["oid"]] = action
    print(f"GitHub 上有 {len(hrefs)} 个，缺 {len(missing)} 个", flush=True)
    for line in missing[:20]:
        print(" 缺：", line)
    checked = 0
    for oid in sorted(hrefs, key=lambda key: objects[key][0])[: args.sample]:
        action = hrefs[oid]
        request = urllib.request.Request(action["href"], headers=action.get("header") or {})
        with urllib.request.urlopen(request, timeout=120) as response:
            digest = hashlib.sha256(response.read()).hexdigest()
        if digest != oid:
            missing.append(f"{objects[oid][1]}：内容哈希不一致")
        checked += 1
    print(f"实际下载校验 {checked} 个，哈希{'有不一致' if missing else '全部一致'}", flush=True)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
