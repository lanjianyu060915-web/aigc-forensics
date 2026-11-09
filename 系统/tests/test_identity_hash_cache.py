"""身份库摘要的进程内复用与变化失效。"""
import hashlib
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "identity_search"))
from identity_search.errors import InputValidationError
from identity_search.searcher import IdentitySearcher


def _searcher(library):
    searcher = IdentitySearcher.__new__(IdentitySearcher)
    searcher.library_path = library
    searcher._file_hash_cache = {}
    return searcher


def test_hash_cache_reuses_unchanged_files_and_checks_new_process(tmp_path, monkeypatch):
    library = tmp_path / "身份库"
    person = library / "人员一"
    person.mkdir(parents=True)
    image = person / "参考.jpg"
    image.write_bytes(b"first")
    opened = []
    original = Path.open

    def counted(path, *args, **kwargs):
        if path == image:
            opened.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    searcher = _searcher(library)
    first = searcher._scan_library_manifest()
    assert len(opened) == 1
    assert searcher._scan_library_manifest() == first
    assert len(opened) == 1
    assert _searcher(library)._scan_library_manifest() == first
    assert len(opened) == 2


def test_size_mtime_and_inventory_changes_invalidate_hashes(tmp_path):
    library = tmp_path / "身份库"
    person = library / "人员一"
    person.mkdir(parents=True)
    image = person / "参考.jpg"
    image.write_bytes(b"first")
    searcher = _searcher(library)
    initial = searcher._scan_library_manifest()
    prior_mtime = image.stat().st_mtime_ns
    image.write_bytes(b"other")
    os.utime(image, ns=(prior_mtime + 1_000_000_000, prior_mtime + 1_000_000_000))
    changed = searcher._scan_library_manifest()
    assert changed[0]["sha256"] == hashlib.sha256(b"other").hexdigest()
    assert changed != initial
    # 大小变化即使保留修改时间也要重算。
    current_mtime = image.stat().st_mtime_ns
    image.write_bytes(b"longer contents")
    os.utime(image, ns=(current_mtime, current_mtime))
    assert searcher._scan_library_manifest()[0]["sha256"] == hashlib.sha256(b"longer contents").hexdigest()
    second = person / "另一张.jpg"
    second.write_bytes(b"second")
    assert len(searcher._scan_library_manifest()) == 2
    image.unlink()
    remaining = searcher._scan_library_manifest()
    assert len(remaining) == len(searcher._file_hash_cache) == 1
    assert remaining[0]["relative_path"].endswith("另一张.jpg")


def test_file_changed_during_hashing_is_rejected(tmp_path, monkeypatch):
    library = tmp_path / "身份库"
    person = library / "人员一"
    person.mkdir(parents=True)
    image = person / "参考.jpg"
    image.write_bytes(b"before")
    searcher = _searcher(library)
    original = hashlib.sha256

    class ChangingDigest:
        def __init__(self):
            self.digest = original()

        def update(self, data):
            self.digest.update(data)
            image.write_bytes(b"changed during scan")

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr(hashlib, "sha256", ChangingDigest)
    with pytest.raises(InputValidationError, match="校验时发生变化"):
        searcher._scan_library_manifest()
