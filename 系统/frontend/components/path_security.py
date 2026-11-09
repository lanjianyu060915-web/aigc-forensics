from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Literal


PathKind = Literal["query", "identity", "source", "visualization"]


class PathContainmentError(ValueError):
    """A result resource path is malformed or leaves every approved root."""


def _deduplicate_roots(roots: Iterable[Path]) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for root in roots:
        try:
            normalized = Path(root).expanduser().resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise PathContainmentError(f"无法解析批准的资源根：{root}") from exc
        if normalized not in resolved:
            resolved.append(normalized)
    return tuple(resolved)


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _parse_path_value(path_value: str, label: str) -> tuple[Path, bool]:
    if not isinstance(path_value, str) or not path_value.strip():
        raise PathContainmentError(f"{label}必须是非空路径字符串。")
    if "\x00" in path_value:
        raise PathContainmentError(f"{label}包含空字符。")

    raw = path_value.strip()
    portable = raw.replace("\\", "/")
    posix_path = PurePosixPath(portable)
    windows_path = PureWindowsPath(raw)

    if ".." in posix_path.parts:
        raise PathContainmentError(f"{label}禁止包含 '..' 路径穿越。")

    is_absolute = posix_path.is_absolute() or windows_path.is_absolute()
    if windows_path.drive and not windows_path.is_absolute():
        raise PathContainmentError(f"{label}禁止使用驱动器相对路径。")

    if is_absolute:
        if windows_path.is_absolute() and os.name != "nt":
            raise PathContainmentError(f"{label}使用了当前平台无法解析的 Windows 绝对路径。")
        return Path(raw), True

    clean_parts = tuple(part for part in posix_path.parts if part not in ("", "."))
    if not clean_parts:
        raise PathContainmentError(f"{label}路径不能为空。")
    if any(":" in part for part in clean_parts):
        raise PathContainmentError(f"{label}的相对路径禁止包含冒号。")
    return Path(*clean_parts), False


def resolve_contained_path(
    path_value: str,
    allowed_roots: Iterable[Path],
    *,
    label: str = "资源路径",
) -> Path:
    """Resolve a JSON path while enforcing containment in explicit roots.

    Relative paths are tried against approved roots in order. An existing file
    wins; if no candidate exists, the first safe candidate is returned so the
    UI can show a non-fatal "missing file" warning. Absolute paths are accepted
    only when they already fall under an approved root.
    """

    roots = _deduplicate_roots(allowed_roots)
    if not roots:
        raise PathContainmentError(f"{label}没有配置任何批准的资源根。")

    parsed, is_absolute = _parse_path_value(path_value, label)
    if is_absolute:
        try:
            candidate = parsed.expanduser().resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise PathContainmentError(f"无法解析{label}：{path_value}") from exc
        if not any(_is_within(candidate, root) for root in roots):
            raise PathContainmentError(f"{label}越过批准目录，已拒绝：{path_value}")
        return candidate

    safe_candidates: list[Path] = []
    escaped_via_link = False
    for root in roots:
        try:
            candidate = (root / parsed).resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        if not _is_within(candidate, root):
            escaped_via_link = True
            continue
        safe_candidates.append(candidate)
        if candidate.is_file():
            return candidate

    if safe_candidates:
        return safe_candidates[0]
    if escaped_via_link:
        raise PathContainmentError(f"{label}经符号链接越过批准目录，已拒绝。")
    raise PathContainmentError(f"无法在批准目录中解析{label}：{path_value}")


@dataclass(frozen=True)
class ResourcePathPolicy:
    """Approved roots used for each result resource class."""

    task_roots: tuple[Path, ...] = ()
    identity_roots: tuple[Path, ...] = ()
    source_roots: tuple[Path, ...] = ()
    sample_roots: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_roots", _deduplicate_roots(self.task_roots))
        object.__setattr__(self, "identity_roots", _deduplicate_roots(self.identity_roots))
        object.__setattr__(self, "source_roots", _deduplicate_roots(self.source_roots))
        object.__setattr__(self, "sample_roots", _deduplicate_roots(self.sample_roots))

    def roots_for(self, kind: PathKind) -> tuple[Path, ...]:
        if kind == "query":
            return _deduplicate_roots((*self.task_roots, *self.sample_roots))
        if kind == "identity":
            return _deduplicate_roots(
                (*self.task_roots, *self.identity_roots, *self.sample_roots)
            )
        if kind == "source":
            return _deduplicate_roots(
                (*self.task_roots, *self.source_roots, *self.sample_roots)
            )
        if kind == "visualization":
            return _deduplicate_roots((*self.task_roots, *self.sample_roots))
        raise PathContainmentError(f"未知资源路径类型：{kind}")

    def resolve(self, path_value: str, kind: PathKind, *, label: str) -> Path:
        return resolve_contained_path(
            path_value,
            self.roots_for(kind),
            label=label,
        )

    def display_roots(self) -> list[str]:
        return [
            str(root)
            for root in _deduplicate_roots(
                (
                    *self.task_roots,
                    *self.identity_roots,
                    *self.source_roots,
                    *self.sample_roots,
                )
            )
        ]

