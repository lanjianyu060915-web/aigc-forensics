"""Local, inference-free query/reference separation for test-only tasks."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Collection, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from case_libraries import library_files


class QueryReferenceGuardError(ValueError):
    """The test-only separation check cannot safely allow this task."""


def _stamp(stat: Any) -> tuple[int, ...]:
    # Windows can expose a different creation time through stat/fstat after copy2
    # (NTFS tunnelling). It is not a content-change clock on that platform.
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _read_binding(path: Path, *, keep_bytes: bool = False) -> tuple[dict[str, Any], bytes]:
    """Detect replacement or modification while hashing; never decode media."""
    before = _stamp(path.stat())
    digest = hashlib.sha256()
    chunks = []
    with path.open("rb") as stream:
        if _stamp(os.fstat(stream.fileno())) != before:
            raise QueryReferenceGuardError(f"检查期间文件发生变化：{path.name}")
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            if keep_bytes:
                chunks.append(chunk)
        if _stamp(os.fstat(stream.fileno())) != before:
            raise QueryReferenceGuardError(f"检查期间文件发生变化：{path.name}")
    if _stamp(path.stat()) != before:
        raise QueryReferenceGuardError(f"检查期间文件发生变化：{path.name}")
    return {"sha256": digest.hexdigest(), "bytes": before[2]}, b"".join(chunks)


def _query_versions(query_sha: str, manifest: Path | None) -> tuple[set[str], dict | None, list[str]]:
    if manifest is None:
        return {query_sha}, None, []
    binding, raw = _read_binding(manifest, keep_bytes=True)
    value = json.loads(raw.decode("utf-8-sig"))
    rows = value.get("rows") if isinstance(value, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise QueryReferenceGuardError("查询清单必须包含非空 rows 列表。")
    versions = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise QueryReferenceGuardError("查询清单含非法条目。")
        sha, pair = row.get("sha256"), row.get("pair_id")
        if (not isinstance(sha, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha) is None
                or not isinstance(pair, str) or not pair.strip()
                or ("role" in row and row["role"] != "query")):
            raise QueryReferenceGuardError("查询清单条目的 sha256、pair_id 或 query 角色无效。")
        versions.append((sha.lower(), pair))
    pairs = {pair for sha, pair in versions if sha == query_sha}
    # Only explicitly registered versions of this query, never ancestor/source images.
    excluded = {query_sha} | {sha for sha, pair in versions if pair in pairs}
    return excluded, {"path": str(manifest), **binding}, sorted(pairs)


def check_query_reference_guard(
    *, query_path: Path, config: Mapping[str, Any], config_path: Path,
    selected_modules: Collection[str], resolve_path: Callable[[Any, Path], Path],
    source_library: str | Path | None = None, identity_library: str | Path | None = None,
) -> dict[str, Any] | None:
    """Fail closed on overlap or unverifiable data, before any model/cloud work."""
    policy = config.get("test_only_policy", {})
    if not isinstance(policy, Mapping):
        raise QueryReferenceGuardError("test_only_policy 必须是对象。")
    if not policy:
        return None
    try:
        query, _ = _read_binding(query_path)
        manifest = None
        if "query_manifest" in policy:
            value = policy["query_manifest"]
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise QueryReferenceGuardError("已配置的查询清单路径无效。")
            manifest = resolve_path(value, config_path)
        excluded, manifest_binding, pairs = _query_versions(query["sha256"], manifest)
        libraries = {}
        for kind, override in (("source", source_library), ("identity", identity_library)):
            if kind not in selected_modules:
                continue
            value = override if override is not None else config.get(kind, {}).get("library")
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise QueryReferenceGuardError(f"{kind} 参考库目录未配置。")
            root = resolve_path(value, config_path).resolve(strict=True)
            paths = library_files(root, kind=kind)
            records = []
            for path in paths:
                binding, _ = _read_binding(path)
                relative = path.relative_to(root).as_posix()
                if binding["sha256"] in excluded:
                    reason = "查询自身的同字节副本" if binding["sha256"] == query["sha256"] else "同一查询的已登记版本"
                    label = "素材库" if kind == "source" else "身份库"
                    raise QueryReferenceGuardError(
                        f"{label}包含{reason}（{relative}）。已停止整项检测；请改用排除该查询及其版本的参考库。"
                    )
                records.append({"path": relative, **binding})
            if paths != library_files(root, kind=kind):
                raise QueryReferenceGuardError(f"检查期间 {kind} 参考库文件集合发生变化。")
            libraries[kind] = {"root": str(root), "file_count": len(records), "files": records}
        return {
            "schema_version": 1, "status": "passed", "usage": "test_only", "training_allowed": False,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "query": query, "query_manifest": manifest_binding,
            "matched_pair_ids": pairs, "excluded_sha256": sorted(excluded), "libraries": libraries,
            "scope": "exact_bytes_and_registered_query_pair_versions_only",
        }
    except QueryReferenceGuardError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise QueryReferenceGuardError(f"无法完成查询与参考库隔离核验：{exc}") from exc


def verify_guard_snapshot(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    """Bind the checked upload and references to the copied task input."""
    for key in ("query", "query_manifest", "matched_pair_ids", "excluded_sha256", "libraries"):
        if before[key] != after[key]:
            raise QueryReferenceGuardError("复制输入期间查询、查询清单或参考库发生变化，已停止整项检测。")
