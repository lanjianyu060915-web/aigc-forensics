from __future__ import annotations

import argparse
import ast
from array import array
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator import DEFAULT_CONFIG, load_config, resolve_path


SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
IDENTITY_INDEX_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)
BUFFALO_L_FILES = (
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
)
EXPECTED_BUFFALO_L_FILES: dict[str, tuple[int, str]] = {
    "1k3d68.onnx": (
        143_607_619,
        "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
    ),
    "2d106det.onnx": (
        5_030_888,
        "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
    ),
    "det_10g.onnx": (
        16_923_827,
        "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    ),
    "genderage.onnx": (
        1_322_532,
        "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
    ),
    "w600k_r50.onnx": (
        174_383_860,
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    ),
}
DEFAULT_DEEPFAKE_SHA256 = {
    "executable": "bc67e52b77fa0559d4d6fe04f024dcdb4a8edf43128f80f7d70947e3da490203",
    "spatial_model": "5bb4b33936eb3adc631828ea271a466d62070a2b8de1902e5d96122317534794",
    "frequency_model": "2ec6828eb2a3151784667eb0c105f07cdb41de6ec2df587d28b2766ceee063eb",
    "fusion_config": "8f76abeabb55da1119b28689e15ccd31eac7f1066e401e766987efbb4603e0b2",
    "adapter": "fc14c1209b7f46715eb252c309f23770cfab584b1c987a7e2a3e137c76ffd131",
    "adapter_runtime": "e117ef6aeef718c4ddbc5eb92e0a615f0239bebe85f7bd830a0963c245e9ef54",
    "adapter_inspector": "f2b46b0102afcca041f8a173ac9e9279548483632ba4932f25a4086eede03dba",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    verify_hash: bool = False,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": None,
        "expected_size_bytes": expected_size_bytes,
        "size_status": "not_checked",
        "expected_sha256": expected_sha256,
        "actual_sha256": None,
        "hash_status": "not_checked",
        "status": "missing",
    }
    if not record["exists"]:
        return record
    try:
        record["size_bytes"] = path.stat().st_size
        if expected_size_bytes is not None:
            record["size_status"] = (
                "match"
                if record["size_bytes"] == expected_size_bytes
                else "mismatch"
            )
        if verify_hash and expected_sha256:
            actual = _sha256(path)
            record["actual_sha256"] = actual
            record["hash_status"] = (
                "match" if actual.casefold() == expected_sha256.casefold() else "mismatch"
            )
        elif expected_sha256:
            record["hash_status"] = "known_not_verified"
        if record["size_status"] == "mismatch":
            record["status"] = "size_mismatch"
        elif record["hash_status"] == "mismatch":
            record["status"] = "hash_mismatch"
        else:
            record["status"] = "ok"
    except OSError as exc:
        record["status"] = "unreadable"
        record["error"] = str(exc)
    return record


def _path_from_section(
    section: dict[str, Any],
    key: str,
    config_path: Path,
) -> Path | None:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return resolve_path(value, config_path)


def _image_inventory(library: Path, *, identity: bool) -> dict[str, Any]:
    if not library.is_dir():
        return {
            "path": str(library),
            "status": "missing",
            "available": False,
            "person_count": 0 if identity else None,
            "image_count": 0,
            "message": f"图库不存在或不是目录：{library}",
        }
    try:
        if identity:
            person_directories = [item for item in library.iterdir() if item.is_dir()]
            people_and_images = [
                (
                    person,
                    [
                        item
                        for item in person.iterdir()
                        if item.is_file()
                        and item.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
                    ],
                )
                for person in person_directories
            ]
            people = [person for person, images in people_and_images if images]
            image_paths = [image for _person, images in people_and_images for image in images]
        else:
            person_directories = []
            people = []
            image_paths = [
                item
                for item in library.rglob("*")
                if item.is_file() and item.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
            ]
    except OSError as exc:
        return {
            "path": str(library),
            "status": "unreadable",
            "available": False,
            "person_count": 0 if identity else None,
            "image_count": 0,
            "message": f"无法读取图库：{exc}",
        }
    person_count = len(people) if identity else None
    available = bool(image_paths) and (not identity or person_count > 0)
    status = "ready" if available else "empty"
    message = (
        f"{person_count} 人、{len(image_paths)} 张图片"
        if identity and available
        else f"{len(image_paths)} 张图片"
        if available
        else "图库为空或没有受支持图片"
    )
    return {
        "path": str(library),
        "status": status,
        "available": available,
        "person_count": person_count,
        "person_directory_count": len(person_directories) if identity else None,
        "image_count": len(image_paths),
        "message": message,
    }


def _buffalo_model_fingerprint(
    records: dict[str, dict[str, Any]],
) -> str | None:
    """Reproduce identity_search.model.preflight_model's frozen fingerprint."""

    if any(record.get("status") != "ok" for record in records.values()):
        return None
    lines: list[str] = []
    for name in sorted(EXPECTED_BUFFALO_L_FILES):
        record = records[name]
        actual_hash = record.get("actual_sha256")
        actual_size = record.get("size_bytes")
        if not isinstance(actual_hash, str) or not isinstance(actual_size, int):
            return None
        lines.append(f"{name}:{actual_size}:{actual_hash.casefold()}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _identity_live_manifest(library: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    person_dirs = sorted(
        (item for item in library.iterdir() if item.is_dir()),
        key=lambda path: (path.name.casefold(), path.name),
    )
    for person_dir in person_dirs:
        images = sorted(
            (
                item
                for item in person_dir.iterdir()
                if item.is_file()
                and item.suffix.casefold() in IDENTITY_INDEX_EXTENSIONS
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
        for image in images:
            stat = image.stat()
            manifest.append(
                {
                    "identity_id": person_dir.name,
                    "relative_path": image.relative_to(library).as_posix(),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": _sha256(image),
                }
            )
    return manifest


def _identity_index_fingerprint(
    manifest: list[dict[str, Any]], model_fingerprint: str
) -> str:
    payload = {
        "cache_schema_version": 1,
        "files": manifest,
        "model_fingerprint": model_fingerprint,
        "det_size": [1280, 1280],
        "embedding_policy": "l2-normalized/person-mean-v1",
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_embedding_npz(path: Path) -> tuple[tuple[int, ...], int]:
    """Validate the safe numeric embeddings.npy member without importing NumPy."""

    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.namelist()
            if members != ["embeddings.npy"]:
                raise ValueError(f"NPZ 成员异常：{members}")
            raw = archive.read("embeddings.npy")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError(f"无法读取 NPZ：{exc}") from exc
    if len(raw) < 10 or raw[:6] != b"\x93NUMPY":
        raise ValueError("embeddings.npy magic 无效")
    major, minor = raw[6], raw[7]
    if (major, minor) == (1, 0):
        header_length = struct.unpack("<H", raw[8:10])[0]
        header_start = 10
    elif major in {2, 3}:
        if len(raw) < 12:
            raise ValueError("embeddings.npy header 截断")
        header_length = struct.unpack("<I", raw[8:12])[0]
        header_start = 12
    else:
        raise ValueError(f"不支持的 NPY 版本：{major}.{minor}")
    header_end = header_start + header_length
    if header_end > len(raw):
        raise ValueError("embeddings.npy header 长度越界")
    try:
        header = ast.literal_eval(raw[header_start:header_end].decode("latin1").strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"embeddings.npy header 无效：{exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("embeddings.npy header 不是对象")
    shape = header.get("shape")
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(not isinstance(value, int) or value < 0 for value in shape)
    ):
        raise ValueError(f"embedding shape 无效：{shape!r}")
    if header.get("fortran_order") is not False:
        raise ValueError("embedding 必须为 C 顺序")
    descriptor = header.get("descr")
    if descriptor not in {"<f4", "=f4"}:
        raise ValueError(f"embedding dtype 必须为 little-endian float32：{descriptor!r}")
    expected_bytes = math.prod(shape) * 4
    data = raw[header_end:]
    if len(data) != expected_bytes:
        raise ValueError(
            f"embedding 数据长度不符：实际 {len(data)}，预期 {expected_bytes}"
        )
    floats = array("f")
    floats.frombytes(data)
    if sys.byteorder != "little":
        floats.byteswap()
    if any(not math.isfinite(value) for value in floats):
        raise ValueError("embedding 包含 NaN 或 Infinity")
    return shape, len(floats)


def _identity_cache_status(
    library: Path,
    cache_root: Path,
    *,
    model_fingerprint: str | None,
) -> dict[str, Any]:
    library_key = hashlib.sha256(str(library).encode("utf-8")).hexdigest()[:24]
    metadata_path = cache_root / f"identity_index_{library_key}.json"
    data_path = cache_root / f"identity_index_{library_key}.npz"
    base = {
        "cache_root": str(cache_root),
        "metadata_path": str(metadata_path),
        "data_path": str(data_path),
        "embedding_count": 0,
        "identity_count": 0,
        "persistent": True,
        "validation_scope": (
            "实时图库清单+模型指纹+缓存 fingerprint、NPZ SHA-256/shape/dtype/有限值；"
            "不初始化模型"
        ),
    }
    if not metadata_path.is_file() and not data_path.is_file():
        return {
            **base,
            "status": "not_initialized",
            "index_status": "not_initialized",
            "message": "身份 embedding/index 缓存尚未建立；首次搜索将构建",
        }
    if not metadata_path.is_file() or not data_path.is_file():
        return {
            **base,
            "status": "incomplete",
            "index_status": "rebuild_required",
            "message": "身份缓存文件不完整；下次搜索将安全重建",
        }
    try:
        if model_fingerprint is None:
            raise ValueError("buffalo_l 模型未通过冻结哈希检查")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("缓存 schema_version 无效")
        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("缓存 entries 为空或格式错误")
        expected_data_hash = payload.get("data_sha256")
        if not isinstance(expected_data_hash, str) or len(expected_data_hash) != 64:
            raise ValueError("缓存缺少有效 NPZ SHA-256")
        actual_data_hash = _sha256(data_path)
        if actual_data_hash.casefold() != expected_data_hash.casefold():
            raise ValueError("缓存 NPZ SHA-256 不一致")
        manifest = _identity_live_manifest(library)
        expected_fingerprint = _identity_index_fingerprint(
            manifest, model_fingerprint
        )
        actual_fingerprint = payload.get("fingerprint")
        if actual_fingerprint != expected_fingerprint:
            raise ValueError("缓存 fingerprint 与当前图库/模型/预处理不一致")
        manifest_identity = {
            item["relative_path"]: item["identity_id"] for item in manifest
        }
        manifest_order = {
            item["relative_path"]: index for index, item in enumerate(manifest)
        }
        cached_people: set[str] = set()
        missing_references: list[str] = []
        seen_paths: set[str] = set()
        previous_order = -1
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("缓存 entry 不是对象")
            relative_path = entry.get("relative_path")
            identity_id = entry.get("identity_id")
            if not isinstance(relative_path, str) or not isinstance(identity_id, str):
                raise ValueError("缓存 entry 字段无效")
            if relative_path not in manifest_identity:
                raise ValueError("缓存 entry 不属于当前图库清单")
            if identity_id != manifest_identity[relative_path]:
                raise ValueError("缓存 entry 身份与图库目录不一致")
            if relative_path in seen_paths:
                raise ValueError("缓存 entry 路径重复")
            current_order = manifest_order[relative_path]
            if current_order <= previous_order:
                raise ValueError("缓存 entry 顺序异常")
            seen_paths.add(relative_path)
            previous_order = current_order
            cached_people.add(identity_id)
            if not (library / Path(relative_path)).is_file():
                missing_references.append(relative_path)
        if missing_references:
            return {
                **base,
                "status": "stale",
                "index_status": "rebuild_required",
                "embedding_count": len(entries),
                "identity_count": len(cached_people),
                "missing_reference_count": len(missing_references),
                "message": "身份缓存引用了已缺失图片；下次搜索将重建",
            }
        embedding_shape, embedding_value_count = _read_embedding_npz(data_path)
        if embedding_shape != (len(entries), 512):
            raise ValueError(
                f"embedding shape 应为 ({len(entries)}, 512)，实际 {embedding_shape}"
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            **base,
            "status": "corrupt",
            "index_status": "rebuild_required",
            "message": f"身份缓存不可用；下次搜索将安全重建：{exc}",
        }
    return {
        **base,
        "status": "ready",
        "index_status": "ready",
        "embedding_count": len(entries),
        "identity_count": len(cached_people),
        "npz_sha256": actual_data_hash,
        "fingerprint": actual_fingerprint,
        "embedding_shape": list(embedding_shape),
        "embedding_value_count": embedding_value_count,
        "message": f"缓存可读：{len(cached_people)} 人、{len(entries)} 个 embedding",
    }


def _nvidia_status() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {
            "cuda_available": False,
            "probe": "nvidia-smi",
            "gpu_names": [],
            "message": "未发现 nvidia-smi，按 CPU 可用性报告",
        }
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "cuda_available": False,
            "probe": "nvidia-smi",
            "gpu_names": [],
            "message": f"GPU 探测失败，按 CPU 可用性报告：{exc}",
        }
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "cuda_available": completed.returncode == 0 and bool(names),
        "probe": "nvidia-smi",
        "gpu_names": names,
        "message": (
            "检测到 NVIDIA GPU：" + ", ".join(names)
            if names
            else "未检测到可用 NVIDIA GPU，使用 CPU"
        ),
    }


def _deepfake_runtime_capability(install_root: Path) -> dict[str, Any]:
    """Inspect the installed Torch build without importing or initializing it."""

    version_path = install_root / "torch" / "version.py"
    try:
        text = version_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {
            "torch_version": None,
            "cuda_build": None,
            "cuda_capable": None,
            "message": f"无法只读识别 AIGC Torch 构建：{exc}",
        }
    version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", text)
    cuda_match = re.search(r"^cuda(?:\s*:[^=]+)?\s*=\s*(.+)$", text, re.MULTILINE)
    version = version_match.group(1) if version_match else None
    cuda_value = cuda_match.group(1).strip() if cuda_match else None
    cpu_build = bool(version and "+cpu" in version.casefold()) or cuda_value == "None"
    return {
        "torch_version": version,
        "cuda_build": None if cpu_build else cuda_value,
        "cuda_capable": not cpu_build,
        "version_file": str(version_path),
        "message": (
            f"AIGC 安装运行时为 CPU 构建（torch {version}）"
            if cpu_build
            else f"AIGC 安装运行时包含 CUDA 支持（torch {version}）"
        ),
    }


def _dist_info_component(install_root: Path, prefix: str) -> dict[str, Any]:
    candidates = sorted(install_root.glob(f"{prefix}-*.dist-info"))
    if not candidates:
        return {"status": "missing", "available": False, "version": None}
    name = candidates[-1].name
    version = name[len(prefix) + 1 : -len(".dist-info")]
    return {
        "status": "ok",
        "available": True,
        "version": version,
        "path": str(candidates[-1]),
    }


def _opencv_component(install_root: Path) -> dict[str, Any]:
    version_file = install_root / "cv2" / "version.py"
    try:
        text = version_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {"status": "missing", "available": False, "version": None}
    match = re.search(r"opencv_version\s*=\s*['\"]([^'\"]+)", text)
    return {
        "status": "ok" if match else "corrupt",
        "available": bool(match),
        "version": match.group(1) if match else None,
        "path": str(version_file),
    }


def _threshold_component(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        threshold = float(payload["frame_threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("frame_threshold越界")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {
            "status": "corrupt" if path.is_file() else "missing",
            "available": False,
            "value": None,
            "message": f"判定阈值配置不可用：{exc}",
        }
    return {"status": "ok", "available": True, "value": threshold}


def _unavailable_module(message: str) -> dict[str, Any]:
    return {"status": "unavailable", "available": False, "message": message}


def _source_feature_cache_status(
    section: dict[str, Any],
    library_path: Path,
    config_path: Path,
    *,
    image_count: int | None,
) -> dict[str, Any]:
    """只看缓存文件是否存在，不读取特征；命名规则与 source_search.feature_cache 一致。"""

    value = section.get("feature_cache")
    if not value:
        return {
            "status": "not_persistent",
            "persistent": False,
            "feature_count": 0,
            "build_policy": "realtime_scan",
            "summary": "无持久 index，检索时实时扫描",
            "message": "底图模块没有持久 index；每次任务实时扫描图库并提取特征",
        }
    resolved_library = library_path.resolve(strict=False)
    key = hashlib.sha256(
        f"{resolved_library.parent.name}/{resolved_library.name}".casefold().encode("utf-8")
    ).hexdigest()[:24]
    index_path = resolve_path(value, config_path) / f"source_index_{key}.npz"
    ready = index_path.is_file()
    return {
        "status": "ready" if ready else "not_initialized",
        "persistent": True,
        "path": str(index_path),
        "size_bytes": index_path.stat().st_size if ready else 0,
        "feature_count": (image_count or 0) if ready else 0,
        "build_policy": "incremental_on_search",
        "summary": "特征缓存已就绪" if ready else "特征缓存首次检索时自动建立",
        "message": (
            "库侧特征已缓存，检索时复用，文件变化时只重算变化的文件"
            if ready
            else "特征缓存尚未建立；首次检索时逐图建立，之后复用"
        ),
    }


def inspect_system_status(
    config_path: str | Path | None = None,
    *,
    identity_library: str | Path | None = None,
    source_library: str | Path | None = None,
) -> dict[str, Any]:
    """Return a read-only, JSON-serializable snapshot of all three modules."""

    requested_config = Path(config_path or DEFAULT_CONFIG).expanduser().resolve(strict=False)
    try:
        config, resolved_config = load_config(requested_config)
    except Exception as exc:
        message = f"无法读取系统配置：{exc}"
        return {
            "schema_version": "1.0",
            "generated_at": _now_iso(),
            "config_path": str(requested_config),
            "overall_status": "unavailable",
            "available": False,
            "device": {"effective": "unknown", "message": message},
            "deepfake": _unavailable_module(message),
            "identity": _unavailable_module(message),
            "source": _unavailable_module(message),
        }

    deepfake_section = config["deepfake"]
    known_hashes = dict(DEFAULT_DEEPFAKE_SHA256)
    configured_hashes = deepfake_section.get("known_sha256")
    if isinstance(configured_hashes, dict):
        known_hashes.update(
            {str(key): str(value) for key, value in configured_hashes.items() if value}
        )
    install_root = _path_from_section(deepfake_section, "install_root", resolved_config)
    install_root = install_root or resolved_config.parent / "__missing_deepfake_install__"
    python_path = _path_from_section(deepfake_section, "python", resolved_config)
    adapter_path = _path_from_section(deepfake_section, "runner", resolved_config)
    adapter_root = (
        adapter_path.parent
        if adapter_path is not None
        else resolved_config.parent / "__missing_deepfake_adapter__"
    )
    deepfake_files = {
        "executable": _file_record(
            install_root / "深度伪造识别系统.exe",
            expected_sha256=known_hashes.get("executable"),
            verify_hash=True,
        ),
        "spatial_model": _file_record(
            install_root
            / "models"
            / "spatial_mobilenet_finetuned_gpu"
            / "spatial_mobilenet_finetuned_gpu.pt",
            expected_sha256=known_hashes.get("spatial_model"),
            verify_hash=True,
        ),
        "frequency_model": _file_record(
            install_root
            / "models"
            / "frequency_stat_formal"
            / "frequency_stat_detector.joblib",
            expected_sha256=known_hashes.get("frequency_model"),
            verify_hash=True,
        ),
        "fusion_config": _file_record(
            install_root / "models" / "fusion_formal" / "fusion_config.json",
            expected_sha256=known_hashes.get("fusion_config"),
            verify_hash=True,
        ),
    }
    deepfake_python = _file_record(python_path) if python_path else {"status": "missing", "path": None, "exists": False}
    deepfake_adapter = (
        _file_record(
            adapter_path,
            expected_sha256=known_hashes.get("adapter"),
            verify_hash=True,
        )
        if adapter_path
        else {"status": "missing", "path": None, "exists": False}
    )
    deepfake_adapter_support = {
        "pyinstaller_runtime": _file_record(
            adapter_root / "pyinstaller_pyz_runtime.py",
            expected_sha256=known_hashes.get("adapter_runtime"),
            verify_hash=True,
        ),
        "archive_inspector": _file_record(
            adapter_root / "inspect_installed_archive.py",
            expected_sha256=known_hashes.get("adapter_inspector"),
            verify_hash=True,
        ),
    }
    read_only_policy = bool(deepfake_section.get("install_root_read_only", True))
    deepfake_runtime = _deepfake_runtime_capability(install_root)
    deepfake_components = {
        "spatial_model": deepfake_files["spatial_model"],
        "frequency_model": deepfake_files["frequency_model"],
        "fusion_config": deepfake_files["fusion_config"],
        "decision_threshold": _threshold_component(
            install_root / "models" / "fusion_formal" / "fusion_config.json"
        ),
        "python": deepfake_python,
        "torch": {
            "status": "ok" if deepfake_runtime.get("torch_version") else "missing",
            "available": bool(deepfake_runtime.get("torch_version")),
            "version": deepfake_runtime.get("torch_version"),
            "cuda_build": deepfake_runtime.get("cuda_build"),
        },
        "torchvision": _dist_info_component(install_root, "torchvision"),
        "sklearn": _dist_info_component(install_root, "scikit_learn"),
        "opencv": _opencv_component(install_root),
    }
    deepfake_available = (
        bool(deepfake_section.get("enabled", True))
        and read_only_policy
        and deepfake_python.get("status") == "ok"
        and deepfake_adapter.get("status") == "ok"
        and all(
            record.get("status") == "ok"
            for record in deepfake_adapter_support.values()
        )
        and all(record.get("status") == "ok" for record in deepfake_files.values())
        and all(
            record.get("status") == "ok"
            for record in deepfake_components.values()
        )
    )
    missing_resources = [
        key for key, record in deepfake_components.items() if record.get("status") != "ok"
    ]
    deepfake_status = {
        "status": "available" if deepfake_available else "unavailable",
        "available": deepfake_available,
        "enabled": deepfake_section.get("enabled", True),
        "message": (
            "AIGC EXE、适配器、配置与模型 SHA-256 均通过只读检查"
            if deepfake_available
            else "AIGC 运行文件、模型、配置或已知 SHA-256 检查未通过"
        ),
        "python": deepfake_python,
        "adapter": deepfake_adapter,
        "adapter_support": deepfake_adapter_support,
        "install_root": {
            "path": str(install_root),
            "exists": install_root.is_dir(),
            "read_only_policy": read_only_policy,
        },
        "executable": deepfake_files["executable"],
        "models": {
            "spatial": deepfake_files["spatial_model"],
            "frequency": deepfake_files["frequency_model"],
        },
        "config": deepfake_files["fusion_config"],
        "components": deepfake_components,
        "readiness": {
            "status": "ready" if deepfake_available else "unavailable",
            "missing_resources": missing_resources,
        },
        "official_base_weight_required": False,
        "spatial_initialization": "mobilenet_v3_small(weights=None) + 项目完整checkpoint(strict=True)",
    }

    identity_section = config["identity"]
    identity_library_path = resolve_path(
        identity_library or identity_section.get("library", "__missing_identity_library__"),
        resolved_config,
    )
    identity_inventory = _image_inventory(identity_library_path, identity=True)
    identity_python_value = identity_section.get("executable") or identity_section.get("python")
    identity_python_path = (
        resolve_path(identity_python_value, resolved_config)
        if isinstance(identity_python_value, str) and identity_python_value
        else resolved_config.parent / "__missing_identity_worker__"
    )
    identity_runner = _path_from_section(identity_section, "runner", resolved_config)
    identity_worker = _file_record(identity_python_path)
    identity_runner_record = (
        _file_record(identity_runner)
        if identity_runner is not None
        else {"status": "not_required" if identity_section.get("executable") else "missing", "path": None, "exists": False}
    )
    model_root = _path_from_section(identity_section, "model_root", resolved_config)
    model_root = model_root or resolved_config.parent / "__missing_identity_model__"
    # Keep the exact same contract as identity_search.model.model_directory():
    # model_root is always the InsightFace root above models/buffalo_l.
    model_dir = model_root / "models" / "buffalo_l"
    identity_models = {
        name: _file_record(
            model_dir / name,
            expected_size_bytes=EXPECTED_BUFFALO_L_FILES[name][0],
            expected_sha256=EXPECTED_BUFFALO_L_FILES[name][1],
            verify_hash=True,
        )
        for name in BUFFALO_L_FILES
    }
    model_fingerprint = _buffalo_model_fingerprint(identity_models)
    identity_cache_root = resolve_path(
        config["runtime"].get("identity_cache", "runtime/cache/identity"),
        resolved_config,
    )
    cache_status = _identity_cache_status(
        identity_library_path,
        identity_cache_root,
        model_fingerprint=model_fingerprint,
    )
    identity_available = (
        identity_inventory["available"]
        and identity_worker["status"] == "ok"
        and identity_runner_record["status"] in {"ok", "not_required"}
        and all(record["status"] == "ok" for record in identity_models.values())
    )
    identity_status = {
        "status": "available" if identity_available else "unavailable",
        "available": identity_available,
        "message": (
            f"身份库可用：{identity_inventory['person_count']} 人、{identity_inventory['image_count']} 张图片"
            if identity_available
            else identity_inventory["message"]
            if not identity_inventory["available"]
            else "身份 worker、runner 或 buffalo_l 模型不完整"
        ),
        "person_count": identity_inventory["person_count"],
        "image_count": identity_inventory["image_count"],
        "embedding_count": cache_status["embedding_count"],
        "index_status": cache_status["index_status"],
        "library": identity_inventory,
        "worker": identity_worker,
        "runner": identity_runner_record,
        "model": {
            "name": "buffalo_l",
            "path": str(model_dir),
            "files": identity_models,
            "fingerprint": model_fingerprint,
            "complete": all(record["status"] == "ok" for record in identity_models.values()),
        },
        "cache": cache_status,
    }

    source_section = config["source"]
    source_library_path = resolve_path(
        source_library or source_section.get("library", "__missing_source_library__"),
        resolved_config,
    )
    source_inventory = _image_inventory(source_library_path, identity=False)
    source_python_value = source_section.get("executable") or source_section.get("python")
    source_python_path = (
        resolve_path(source_python_value, resolved_config)
        if isinstance(source_python_value, str) and source_python_value
        else resolved_config.parent / "__missing_source_worker__"
    )
    source_runner = _path_from_section(source_section, "runner", resolved_config)
    source_worker = _file_record(source_python_path)
    source_runner_record = (
        _file_record(source_runner)
        if source_runner is not None
        else {"status": "not_required" if source_section.get("executable") else "missing", "path": None, "exists": False}
    )
    source_available = (
        source_inventory["available"]
        and source_worker["status"] == "ok"
        and source_runner_record["status"] in {"ok", "not_required"}
    )
    source_index = _source_feature_cache_status(
        source_section,
        source_library_path,
        resolved_config,
        image_count=source_inventory["image_count"],
    )
    source_status = {
        "status": "available" if source_available else "unavailable",
        "available": source_available,
        "message": (
            f"底图库可用：{source_inventory['image_count']} 张图片；{source_index['summary']}"
            if source_available
            else source_inventory["message"]
            if not source_inventory["available"]
            else "底图 worker 或 runner 不完整"
        ),
        "image_count": source_inventory["image_count"],
        "feature_count": source_index["feature_count"],
        "index_status": source_index["status"],
        "library": source_inventory,
        "worker": source_worker,
        "runner": source_runner_record,
        "index": source_index,
    }

    gpu = _nvidia_status()
    requested_device = str(deepfake_section.get("device", "auto"))
    effective_device = (
        "cuda"
        if requested_device in {"auto", "cuda"}
        and gpu["cuda_available"]
        and deepfake_runtime["cuda_capable"] is True
        else "cpu"
    )
    identity_ctx = identity_section.get("ctx", -1)
    try:
        identity_ctx_number = int(identity_ctx)
    except (TypeError, ValueError):
        identity_effective = "unknown"
    else:
        identity_effective = "cpu" if identity_ctx_number < 0 else f"cuda:{identity_ctx_number}"
    device_status = {
        **gpu,
        "deepfake_requested": requested_device,
        "deepfake_effective": effective_device,
        "deepfake_runtime": deepfake_runtime,
        "deepfake_cuda_available": bool(
            gpu["cuda_available"] and deepfake_runtime["cuda_capable"] is True
        ),
        "identity_ctx": identity_ctx,
        "identity_effective": identity_effective,
        "effective": effective_device,
        "model_status": "loaded_or_ready" if deepfake_available else "resource_check_failed",
    }
    deepfake_status["components"]["cuda"] = {
        "status": "ok",
        "available": bool(
            gpu["cuda_available"] and deepfake_runtime["cuda_capable"] is True
        ),
        "effective_device": effective_device,
        "message": (
            "CUDA可用" if effective_device == "cuda"
            else "CUDA未启用/不可用，已自动使用CPU"
        ),
    }
    module_availability = [
        deepfake_status["available"],
        identity_status["available"],
        source_status["available"],
    ]
    overall_status = (
        "available"
        if all(module_availability)
        else "degraded"
        if any(module_availability)
        else "unavailable"
    )
    return {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "config_path": str(resolved_config),
        "overall_status": overall_status,
        "available": all(module_availability),
        "device": device_status,
        "deepfake": deepfake_status,
        "identity": identity_status,
        "source": source_status,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只读检查三模块、图库、缓存与设备状态。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置 JSON 路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status = inspect_system_status(args.config)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["overall_status"] == "available" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_DEEPFAKE_SHA256", "inspect_system_status"]
