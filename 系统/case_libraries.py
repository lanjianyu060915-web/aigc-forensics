"""案件库的目录边界检查、内容清单和索引准备。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
IDENTITY_SUFFIXES = IMAGE_SUFFIXES | {".tif", ".tiff"}
Progress = Callable[[dict[str, Any]], None]


class LibraryValidationError(ValueError):
    """案件库内容或目录边界不符合要求。"""


def _emit(callback: Progress | None, stage: str, current: int, total: int,
          message: str) -> None:
    if callback:
        callback({"stage": stage, "current": current, "total": total,
                  "message": message})


def checked_reference(root: str | Path, value: str | Path) -> Path:
    """允许用户选择外部库，拒绝库内引用逃逸及父目录跳转。"""
    base = Path(root).expanduser().resolve(strict=True)
    candidate = Path(value)
    if ".." in candidate.parts:
        raise LibraryValidationError("库内引用不能包含父目录跳转。")
    target = (candidate if candidate.is_absolute() else base / candidate).resolve(strict=True)
    if not target.is_relative_to(base):
        raise LibraryValidationError("资源引用越出用户选定的库根目录。")
    return target


def library_files(root: str | Path, *, kind: str = "source") -> list[Path]:
    """逐层核对解析路径，绝不先遍历越界链接的目标。"""
    if not str(root).strip():
        raise LibraryValidationError("请先选择案件库目录。")
    base = Path(root).expanduser().resolve(strict=True)
    if not base.is_dir():
        raise LibraryValidationError("案件库必须是目录。")
    output: list[Path] = []
    visited: set[Path] = set()
    suffixes = IDENTITY_SUFFIXES if kind == "identity" else IMAGE_SUFFIXES

    def walk(directory: Path) -> None:
        resolved = checked_reference(base, directory)
        if resolved in visited:
            raise LibraryValidationError("案件库含循环或重复的目录链接。")
        visited.add(resolved)
        for entry in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
            safe = checked_reference(base, entry)
            if safe.is_dir():
                walk(entry)
            elif safe.is_file() and entry.suffix.lower() in suffixes:
                output.append(entry)

    walk(base)
    return output


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_hash(records: list[dict[str, Any]]) -> str:
    payload = [{"path": row["path"], "sha256": row["sha256"]} for row in records]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def inspect_library(root: str | Path, *, kind: str,
                    face_counter: Callable[[Path], int] | None = None,
                    face_inspector: Callable[[Path], dict[str, Any]] | None = None,
                    progress: Progress | None = None) -> dict[str, Any]:
    """生成稳定内容清单；无法解码和无人脸照片仍保留在审计记录中。"""
    if kind not in {"source", "identity"}:
        raise LibraryValidationError("库类型必须是素材库或身份库。")
    if not str(root).strip():
        raise LibraryValidationError("请先选择案件库目录。")
    import cv2
    import numpy as np

    base = Path(root).expanduser().resolve(strict=True)
    paths = library_files(base, kind=kind)
    records: list[dict[str, Any]] = []
    first_by_hash: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    for index, path in enumerate(paths, 1):
        checked_reference(base, path)
        before = path.stat()
        relative = path.relative_to(base).as_posix()
        row: dict[str, Any] = {"path": relative, "sha256": _hash_file(path),
                               "size": before.st_size, "readable": False,
                               "face_count": None, "issues": []}
        try:
            raw = np.fromfile(path, dtype=np.uint8)
            picture = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
        except (OSError, ValueError, cv2.error):
            picture = None
        if picture is None:
            row["issues"].append("无法解码")
        else:
            row.update(readable=True, width=int(picture.shape[1]), height=int(picture.shape[0]))
        if kind == "identity":
            if len(Path(relative).parts) != 2:
                row["issues"].append("身份照片需直接放在人员子目录中")
            else:
                row["identity_id"] = Path(relative).parts[0]
            if picture is not None and (face_counter is not None or face_inspector is not None):
                if face_inspector is not None:
                    face_result = face_inspector(path)
                    row["face_count"] = face_result["face_count"]
                    row["reference_detection"] = face_result["reference_detection"]
                    row["issues"].extend(face_result["issues"])
                else:
                    row["face_count"] = int(face_counter(path))
                if row["face_count"] == 0:
                    row["issues"].append("未检出人脸")
                elif row["face_count"] != 1:
                    row["issues"].append("身份参考照片需要恰好一张人脸")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise LibraryValidationError("检查时库文件发生变化，请重新检查。")
        if row["sha256"] in first_by_hash:
            duplicates.append({"path": relative, "duplicate_of": first_by_hash[row["sha256"]]})
        else:
            first_by_hash[row["sha256"]] = relative
        records.append(row)
        _emit(progress, f"inspect_{kind}", index, len(paths), f"检查 {relative}")
    return {
        "kind": kind, "name": base.name, "root": str(base),
        "file_count": len(records), "readable_count": sum(row["readable"] for row in records),
        "invalid_count": sum(not row["readable"] for row in records),
        "no_face_count": sum(row["face_count"] == 0 for row in records),
        "multiple_faces_count": sum((row["face_count"] or 0) > 1 for row in records),
        "face_check_completed": kind != "identity" or face_counter is not None or face_inspector is not None,
        "duplicate_count": len(duplicates), "duplicates": duplicates,
        "usable_count": sum(not row["issues"] for row in records),
        "files": records, "manifest_sha256": _manifest_hash(records),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def _prepare_identity(model_root: Path, scale: int):
    from identity_search.extractor import FeatureExtractor
    from identity_search.model import preflight_model
    model = preflight_model(model_root, verify_hashes=True)
    extractor = FeatureExtractor(model_root=model.root, device="cpu", ctx_id=-1,
                                 det_size=(scale, scale))
    return model, extractor


def _reference_scales(identity_scale, identity_scales):
    from identity_search.reference_detection import DEFAULT_REFERENCE_SCALES, reference_policy
    scales = (tuple(identity_scales) if identity_scales is not None else
              DEFAULT_REFERENCE_SCALES if identity_scale == 1280 else (identity_scale,))
    return tuple(reference_policy(scales)["scale_chain"])


def _module_imports(product_root: Path) -> None:
    for module in ("source_search", "identity_search"):
        value = str(product_root / "modules" / module)
        if value not in sys.path:
            sys.path.insert(0, value)


def check_case_libraries(*, source_root: str | Path, identity_root: str | Path,
                         product_root: str | Path, model_root: str | Path,
                         identity_scale: int = 1280,
                         identity_scales=None,
                         progress: Progress | None = None) -> dict[str, Any]:
    """仅检查，不在用户选择的库内生成或修改文件。"""
    library_files(source_root, kind="source")
    library_files(identity_root, kind="identity")
    _module_imports(Path(product_root))
    source = inspect_library(source_root, kind="source", progress=progress)
    _emit(progress, "load_identity", 0, 1, "正在加载本地人脸模型")
    from identity_search.reference_detection import inspect_reference
    scales = _reference_scales(identity_scale, identity_scales)
    _, extractor = _prepare_identity(Path(model_root), scales[0])
    identity = inspect_library(identity_root, kind="identity", progress=progress,
                               face_inspector=lambda path: inspect_reference(extractor, path, scales))
    return {"source": source, "identity": identity}


class _IndexProgress:
    def __init__(self, callback: Progress | None):
        self.callback = callback

    def emit(self, stage: str, status: str, message: str, **values: Any) -> None:
        _emit(self.callback, stage, int(values.get("progress_current") or 0),
              int(values.get("progress_total") or 0), message)


def _registry_path(cache_root: Path, source: str | Path, identity: str | Path) -> Path:
    value = "\n".join(str(Path(item).resolve()).casefold() for item in (source, identity))
    token = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return cache_root / "case_libraries" / f"{token}.json"


def case_source_cache(cache_root: str | Path, library_root: str | Path,
                      manifest_sha256: str) -> Path:
    """按库根和内容清单隔离缓存，避免同名库或保留时间戳的改图复用旧特征。"""
    value = f"{Path(library_root).resolve()}\n{manifest_sha256}"
    token = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return Path(cache_root) / f"case_{token}"


def build_case_libraries(*, name: str, source_root: str | Path, identity_root: str | Path,
                         product_root: str | Path, model_root: str | Path,
                         cache_root: str | Path, source_cache: str | Path,
                         identity_cache: str | Path, identity_scale: int = 1280,
                         identity_scales=None,
                         prewarm_sift: bool = True,
                         progress: Progress | None = None) -> dict[str, Any]:
    """复用现有特征与身份索引算法，缓存全部留在本机用户目录。"""
    if not str(source_root).strip() or not str(identity_root).strip():
        raise LibraryValidationError("请先选择素材库和身份库目录。")
    product = Path(product_root)
    _module_imports(product)
    from identity_search.searcher import IdentitySearcher
    from identity_search.reference_detection import inspect_reference
    from source_search.face_mask import FaceMasker
    from source_search.feature_cache import SourceFeatureCache
    import cv2

    roots = [Path(source_root).expanduser().resolve(), Path(identity_root).expanduser().resolve()]
    for cache in (cache_root, source_cache, identity_cache):
        resolved_cache = Path(cache).expanduser().resolve()
        if any(resolved_cache.is_relative_to(root) for root in roots):
            raise LibraryValidationError("缓存目录不能放在案件库里面。")
    # 任何索引读取之前先检查目录边界。
    for kind, root in zip(("source", "identity"), roots):
        library_files(root, kind=kind)
    _emit(progress, "load_identity", 0, 1, "正在加载本地人脸模型")
    scales = _reference_scales(identity_scale, identity_scales)
    model, extractor = _prepare_identity(Path(model_root), scales[0])
    source = inspect_library(roots[0], kind="source", progress=progress)
    identity = inspect_library(roots[1], kind="identity", progress=progress,
                               face_inspector=lambda path: inspect_reference(extractor, path, scales))
    if not source["usable_count"] or not identity["usable_count"]:
        raise LibraryValidationError("素材库需有可解码图片，身份库需有合规的单人脸参考照片。")
    source_index = SourceFeatureCache(
        case_source_cache(source_cache, roots[0], source["manifest_sha256"]), roots[0])
    entries, skipped = source_index.load_or_build(
        FaceMasker(), progress_callback=lambda current, total, path, readable:
        _emit(progress, "source_global", current, total, f"建立素材特征 {path.name}"))
    sift = cv2.SIFT_create(nfeatures=2500) if prewarm_sift else None
    for current, entry in enumerate(entries, 1):
        checked_reference(roots[0], entry.path)
        if prewarm_sift:
            source_index.sift_features(entry, sift)
            _emit(progress, "source_local", current, len(entries), f"建立局部特征 {entry.path.name}")
    identity_index = IdentitySearcher(
        library_path=roots[1], cache_dir=identity_cache, model_root=model.root,
        model_fingerprint=model.fingerprint, device="cpu", ctx_id=-1,
        det_size=(scales[0], scales[0]), extractor=extractor, reference_scales=scales,
        progress_reporter=_IndexProgress(progress))
    # 防止建立索引期间有人改库后仍保存旧清单。
    for root, snapshot in ((roots[0], source), (roots[1], identity)):
        current = [{"path": path.relative_to(root).as_posix(), "sha256": _hash_file(path)}
                   for path in library_files(root, kind=snapshot["kind"])]
        if _manifest_hash(current) != snapshot["manifest_sha256"]:
            raise LibraryValidationError("建库时文件发生变化，未保存建库记录，请重新建立。")
    result = {"schema_version": 1, "name": name.strip() or "案件库",
              "built_at": datetime.now(timezone.utc).isoformat(),
              "source": source, "identity": identity,
              "source_indexed_count": len(entries), "source_skipped_count": len(skipped),
              "source_sift_prewarmed": bool(prewarm_sift),
              "identity_indexed_count": len(identity_index.identities), "status": "ready"}
    target = _registry_path(Path(cache_root), roots[0], roots[1])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    _emit(progress, "complete", 1, 1, "案件库检查及索引完成")
    return result


def describe_case_libraries(*, cache_root: str | Path, source_root: str | Path,
                            identity_root: str | Path) -> dict[str, Any]:
    """报告采用本次实际内容摘要；旧建库记录与现有内容不一致时明确标记。"""
    registry = _registry_path(Path(cache_root), source_root, identity_root)
    try:
        stored = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = {}
    result: dict[str, Any] = {"name": stored.get("name", "本次比对库"),
                              "built_at": stored.get("built_at"), "libraries": {}}
    for kind, root in (("source", source_root), ("identity", identity_root)):
        base = Path(root).expanduser().resolve(strict=True)
        records = [{"path": path.relative_to(base).as_posix(), "sha256": _hash_file(path)}
                   for path in library_files(base, kind=kind)]
        digest = _manifest_hash(records)
        previous = stored.get(kind, {})
        result["libraries"][kind] = {
            "name": base.name, "file_count": len(records), "manifest_sha256": digest,
            "built_at": stored.get("built_at") if previous.get("manifest_sha256") == digest else None,
            "build_record_matches": bool(stored) and previous.get("manifest_sha256") == digest,
        }
    return result
