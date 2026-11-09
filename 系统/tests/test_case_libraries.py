"""案件库内容清单、路径边界与建库记录测试。"""

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from case_libraries import (
    LibraryValidationError, _registry_path, checked_reference, case_source_cache,
    describe_case_libraries, inspect_library, library_files,
)


def make_picture(path: Path, color: int = 80) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", np.full((32, 48, 3), color, dtype=np.uint8))
    assert ok
    encoded.tofile(path)
    return path


def test_external_root_and_relative_reference(tmp_path):
    library = tmp_path / "仓库外的案件库"
    image = make_picture(library / "照片.png")
    assert checked_reference(library, "照片.png") == image.resolve()
    assert library_files(library) == [image]


def test_empty_library_does_not_mean_current_directory():
    with pytest.raises(LibraryValidationError, match="先选择"):
        inspect_library("", kind="source")


@pytest.mark.parametrize("reference", ["../外部.png", "子目录/../../外部.png"])
def test_parent_reference_rejected(tmp_path, reference):
    library = tmp_path / "库"
    library.mkdir()
    with pytest.raises(LibraryValidationError, match="父目录"):
        checked_reference(library, reference)


def test_absolute_escape_rejected(tmp_path):
    library = tmp_path / "库"
    library.mkdir()
    outside = make_picture(tmp_path / "外部.png")
    with pytest.raises(LibraryValidationError, match="越出"):
        checked_reference(library, outside)


def test_symlink_escape_rejected(tmp_path):
    library = tmp_path / "库"
    library.mkdir()
    outside = make_picture(tmp_path / "外部.png")
    link = library / "链接.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("当前系统没有文件符号链接创建权限")
    with pytest.raises(LibraryValidationError, match="越出"):
        library_files(library)


def test_directory_link_escape_rejected(tmp_path):
    library = tmp_path / "库"
    library.mkdir()
    outside = tmp_path / "外部"
    make_picture(outside / "照片.png")
    link = library / "链接"
    if os.name == "nt":
        import subprocess
        quoted_link = str(link).replace("'", "''")
        quoted_outside = str(outside).replace("'", "''")
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"New-Item -ItemType Junction -Path '{quoted_link}' -Target '{quoted_outside}' | Out-Null"],
            capture_output=True)
        if completed.returncode:
            pytest.skip("当前环境不能创建目录联接")
    else:
        link.symlink_to(outside, target_is_directory=True)
    try:
        with pytest.raises(LibraryValidationError, match="越出"):
            library_files(library)
    finally:
        if link.exists():
            if os.name == "nt":
                os.rmdir(link)
            else:
                link.unlink()


def test_corrupt_duplicate_and_stable_content_manifest(tmp_path):
    image = make_picture(tmp_path / "甲.png")
    (tmp_path / "重复.png").write_bytes(image.read_bytes())
    (tmp_path / "损坏.jpg").write_bytes(b"broken")
    progress = []
    inspected = inspect_library(tmp_path, kind="source", progress=progress.append)
    assert inspected["file_count"] == 3
    assert inspected["invalid_count"] == 1
    assert inspected["duplicate_count"] == 1
    assert progress[-1]["current"] == progress[-1]["total"] == 3
    os.utime(image, None)
    assert inspect_library(tmp_path, kind="source")["manifest_sha256"] == inspected["manifest_sha256"]
    make_picture(image, 120)
    assert inspect_library(tmp_path, kind="source")["manifest_sha256"] != inspected["manifest_sha256"]


def test_identity_missing_face_and_directory(tmp_path):
    make_picture(tmp_path / "散放.png")
    make_picture(tmp_path / "人员甲" / "没有脸.png")
    make_picture(tmp_path / "人员乙" / "多人.png")
    make_picture(tmp_path / "人员丙" / "合规.png")
    counts = {"散放.png": 1, "没有脸.png": 0, "多人.png": 2, "合规.png": 1}
    result = inspect_library(tmp_path, kind="identity", face_counter=lambda path: counts[path.name])
    assert result["no_face_count"] == 1
    assert result["multiple_faces_count"] == 1
    assert result["usable_count"] == 1
    assert result["face_check_completed"] is True


def test_face_check_not_silently_assumed(tmp_path):
    make_picture(tmp_path / "人员甲" / "照片.png")
    result = inspect_library(tmp_path, kind="identity")
    assert result["face_check_completed"] is False
    assert result["files"][0]["face_count"] is None


def test_identity_tiff_is_hashed_but_source_tiff_is_not_indexed(tmp_path):
    image = make_picture(tmp_path / "人员甲" / "参考.tiff")
    identity = inspect_library(tmp_path, kind="identity", face_counter=lambda path: 1)
    assert identity["file_count"] == identity["usable_count"] == 1
    assert inspect_library(tmp_path, kind="source")["file_count"] == 0
    make_picture(image, 180)
    changed = inspect_library(tmp_path, kind="identity", face_counter=lambda path: 1)
    assert identity["manifest_sha256"] != changed["manifest_sha256"]


def test_nested_identity_photo_is_not_reported_as_indexable(tmp_path):
    make_picture(tmp_path / "人员甲" / "子目录" / "参考.png")
    identity = inspect_library(tmp_path, kind="identity", face_counter=lambda path: 1)
    assert identity["file_count"] == 1
    assert identity["usable_count"] == 0
    assert "直接" in identity["files"][0]["issues"][0]


def test_report_detects_library_changed_after_build(tmp_path):
    source = tmp_path / "素材"
    identity = tmp_path / "身份"
    cache = tmp_path / "缓存"
    image = make_picture(source / "照片.png")
    identity_photo = make_picture(identity / "人员甲" / "照片.tiff")
    source_manifest = inspect_library(source, kind="source")
    identity_manifest = inspect_library(identity, kind="identity")
    target = _registry_path(cache, source, identity)
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"name": "案件甲", "built_at": "冻结时刻",
                                  "source": source_manifest, "identity": identity_manifest}), encoding="utf-8")
    report = describe_case_libraries(cache_root=cache, source_root=source, identity_root=identity)
    assert report["libraries"]["source"]["build_record_matches"] is True
    make_picture(image, 220)
    report = describe_case_libraries(cache_root=cache, source_root=source, identity_root=identity)
    assert report["libraries"]["source"]["build_record_matches"] is False
    assert report["libraries"]["source"]["built_at"] is None
    assert report["libraries"]["identity"]["built_at"] == "冻结时刻"
    make_picture(identity_photo, 120)
    changed_identity = describe_case_libraries(cache_root=cache, source_root=source, identity_root=identity)
    assert changed_identity["libraries"]["identity"]["build_record_matches"] is False
    assert changed_identity["libraries"]["identity"]["built_at"] is None


def test_manifest_uses_bytes_not_image_appearance(tmp_path):
    image = make_picture(tmp_path / "照片.png")
    record = inspect_library(tmp_path, kind="source")["files"][0]
    assert record["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()


def test_content_and_location_separate_case_caches(tmp_path):
    cache = tmp_path / "缓存"
    first = tmp_path / "位置甲" / "同名库"
    second = tmp_path / "位置乙" / "同名库"
    assert case_source_cache(cache, first, "甲") != case_source_cache(cache, first, "乙")
    assert case_source_cache(cache, first, "甲") != case_source_cache(cache, second, "甲")
    assert case_source_cache(cache, first, "甲") == case_source_cache(cache, first, "甲")
