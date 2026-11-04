from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import marshal
import os
import struct
import sys
import zlib
from pathlib import Path
from types import ModuleType
from typing import BinaryIO


class PyInstallerArchiveError(RuntimeError):
    """Raised when the installed detector's PyInstaller archive is invalid."""


class CArchiveReader:
    """Small, read-only reader for the PKG archive appended to a PyInstaller EXE."""

    COOKIE_MAGIC = b"MEI\x0c\x0b\x0a\x0b\x0e"
    COOKIE_FORMAT = "!8sIIII64s"
    COOKIE_LENGTH = struct.calcsize(COOKIE_FORMAT)
    TOC_ENTRY_FORMAT = "!IIIIBc"
    TOC_ENTRY_LENGTH = struct.calcsize(TOC_ENTRY_FORMAT)

    def __init__(self, executable: str | Path) -> None:
        self.executable = Path(executable).resolve()
        self.start_offset = 0
        self.toc: dict[str, tuple[int, int, int, int, str]] = {}
        self._load()

    @classmethod
    def _find_magic(cls, stream: BinaryIO) -> int:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        overlap = len(cls.COOKIE_MAGIC) - 1
        while end >= len(cls.COOKIE_MAGIC):
            start = max(0, end - 8192)
            stream.seek(start)
            block = stream.read(end - start)
            index = block.rfind(cls.COOKIE_MAGIC)
            if index >= 0:
                return start + index
            if start == 0:
                break
            end = start + overlap
        return -1

    def _load(self) -> None:
        if not self.executable.is_file():
            raise PyInstallerArchiveError(f"PyInstaller EXE 不存在：{self.executable}")
        with self.executable.open("rb") as stream:
            cookie_offset = self._find_magic(stream)
            if cookie_offset < 0:
                raise PyInstallerArchiveError("未找到 PyInstaller CArchive cookie。")
            stream.seek(cookie_offset)
            cookie = stream.read(self.COOKIE_LENGTH)
            if len(cookie) != self.COOKIE_LENGTH:
                raise PyInstallerArchiveError("PyInstaller CArchive cookie 不完整。")
            magic, archive_length, toc_offset, toc_length, _pyvers, library = struct.unpack(
                self.COOKIE_FORMAT, cookie
            )
            if magic != self.COOKIE_MAGIC or not library.rstrip(b"\0"):
                raise PyInstallerArchiveError("PyInstaller CArchive cookie 无效。")
            archive_end = cookie_offset + self.COOKIE_LENGTH
            self.start_offset = archive_end - archive_length
            stream.seek(self.start_offset + toc_offset)
            toc_data = stream.read(toc_length)
        self.toc = self._parse_toc(toc_data)

    @classmethod
    def _parse_toc(cls, data: bytes) -> dict[str, tuple[int, int, int, int, str]]:
        toc: dict[str, tuple[int, int, int, int, str]] = {}
        position = 0
        while position < len(data):
            header = data[position : position + cls.TOC_ENTRY_LENGTH]
            if len(header) != cls.TOC_ENTRY_LENGTH:
                raise PyInstallerArchiveError("PyInstaller CArchive TOC 损坏。")
            entry_length, offset, length, unpacked_length, compressed, typecode = struct.unpack(
                cls.TOC_ENTRY_FORMAT, header
            )
            if entry_length < cls.TOC_ENTRY_LENGTH:
                raise PyInstallerArchiveError("PyInstaller CArchive TOC entry 长度无效。")
            name_start = position + cls.TOC_ENTRY_LENGTH
            name_end = position + entry_length
            raw_name = data[name_start:name_end].rstrip(b"\0")
            try:
                name = raw_name.decode("utf-8")
                kind = typecode.decode("ascii")
            except UnicodeDecodeError as exc:
                raise PyInstallerArchiveError("PyInstaller CArchive TOC 编码无效。") from exc
            if kind != "o":
                toc[name] = (offset, length, unpacked_length, compressed, kind)
            position = name_end
        return toc

    def extract(self, name: str) -> bytes:
        try:
            offset, length, _unpacked_length, compressed, _kind = self.toc[name]
        except KeyError as exc:
            raise PyInstallerArchiveError(f"CArchive 中不存在条目：{name}") from exc
        with self.executable.open("rb") as stream:
            stream.seek(self.start_offset + offset)
            data = stream.read(length)
        return zlib.decompress(data) if compressed else data

    def open_pyz(self) -> "ZlibArchiveReader":
        candidates = [
            (name, entry)
            for name, entry in self.toc.items()
            if entry[-1] == "z"
        ]
        if len(candidates) != 1:
            raise PyInstallerArchiveError(
                f"期望一个 PYZ 条目，实际找到 {len(candidates)} 个。"
            )
        _name, (offset, _length, _unpacked_length, compressed, _kind) = candidates[0]
        if compressed:
            raise PyInstallerArchiveError("不支持外层再次压缩的 PYZ 条目。")
        return ZlibArchiveReader(self.executable, self.start_offset + offset)


class ZlibArchiveReader:
    """Read code objects from a Python-version-compatible PyInstaller PYZ."""

    def __init__(self, executable: Path, start_offset: int) -> None:
        self.executable = executable
        self.start_offset = start_offset
        self.toc: dict[str, tuple[int, int, int]] = {}
        with executable.open("rb") as stream:
            stream.seek(start_offset)
            if stream.read(4) != b"PYZ\0":
                raise PyInstallerArchiveError("PYZ magic 无效。")
            python_magic = stream.read(4)
            if python_magic != importlib.util.MAGIC_NUMBER:
                raise PyInstallerArchiveError(
                    "PYZ Python 版本与当前解释器不兼容。"
                )
            (toc_offset,) = struct.unpack("!i", stream.read(4))
            stream.seek(start_offset + toc_offset)
            try:
                self.toc = dict(marshal.load(stream))
            except (EOFError, ValueError, TypeError) as exc:
                raise PyInstallerArchiveError("无法读取 PYZ TOC。") from exc

    def extract_code(self, name: str):
        try:
            typecode, offset, length = self.toc[name]
        except KeyError as exc:
            raise ImportError(f"PYZ 中不存在模块：{name}") from exc
        if typecode == 3:
            return None
        with self.executable.open("rb") as stream:
            stream.seek(self.start_offset + offset)
            compressed = stream.read(length)
        try:
            return marshal.loads(zlib.decompress(compressed))
        except (EOFError, ValueError, TypeError, zlib.error) as exc:
            raise ImportError(f"无法从 PYZ 读取模块：{name}") from exc


class PyInstallerPyzFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Expose the installed detector's bundled pure modules to CPython 3.12."""

    def __init__(self, archive: ZlibArchiveReader, resource_root: str | Path) -> None:
        self.archive = archive
        self.resource_root = Path(resource_root).resolve()

    def find_spec(self, fullname: str, path=None, target=None):
        # The repaired formal package ships an audited source overlay for the
        # detector itself.  Prefer those ``src`` files while retaining the PYZ
        # archive as a fallback for dependencies that are not materialized on
        # disk (for example typing_extensions in the legacy bundle).
        if fullname == "src" or fullname.startswith("src."):
            relative = self.resource_root.joinpath(*fullname.split("."))
            if (relative / "__init__.py").is_file() or relative.with_suffix(".py").is_file():
                return None
        entry = self.archive.toc.get(fullname)
        if entry is None:
            return None
        typecode = int(entry[0])
        if typecode == 3:
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [str(self._package_path(fullname))]
            return spec
        return importlib.machinery.ModuleSpec(
            fullname,
            self,
            origin=str(self._origin(fullname, typecode == 1)),
            is_package=typecode == 1,
        )

    def create_module(self, spec):
        return None

    def exec_module(self, module: ModuleType) -> None:
        name = module.__spec__.name
        typecode = int(self.archive.toc[name][0])
        is_package = typecode == 1
        origin = self._origin(name, is_package)
        module.__file__ = str(origin)
        module.__loader__ = self
        module.__package__ = name if is_package else name.rpartition(".")[0]
        if is_package:
            module.__path__ = [str(self._package_path(name))]
        code = self.archive.extract_code(name)
        if code is None:
            return
        exec(code, module.__dict__)

    def get_code(self, fullname: str):
        return self.archive.extract_code(fullname)

    def get_filename(self, fullname: str) -> str:
        entry = self.archive.toc[fullname]
        return str(self._origin(fullname, int(entry[0]) == 1))

    def is_package(self, fullname: str) -> bool:
        return int(self.archive.toc[fullname][0]) in {1, 3}

    def _package_path(self, fullname: str) -> Path:
        return self.resource_root.joinpath(*fullname.split("."))

    def _origin(self, fullname: str, is_package: bool) -> Path:
        base = self._package_path(fullname)
        return base / "__init__.py" if is_package else base.with_suffix(".py")


def install_importer(executable: str | Path, resource_root: str | Path) -> PyInstallerPyzFinder:
    """Install one idempotent finder and return it."""

    resolved_executable = Path(executable).resolve()
    resolved_root = Path(resource_root).resolve()
    for finder in sys.meta_path:
        if (
            isinstance(finder, PyInstallerPyzFinder)
            and finder.archive.executable == resolved_executable
            and finder.resource_root == resolved_root
        ):
            return finder
    archive = CArchiveReader(resolved_executable).open_pyz()
    finder = PyInstallerPyzFinder(archive, resolved_root)
    path_finder_index = next(
        (
            index
            for index, existing in enumerate(sys.meta_path)
            if existing is importlib.machinery.PathFinder
        ),
        len(sys.meta_path),
    )
    sys.meta_path.insert(path_finder_index, finder)
    return finder
