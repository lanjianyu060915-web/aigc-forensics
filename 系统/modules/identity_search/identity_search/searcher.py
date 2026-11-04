"""Deterministic person-level identity index with a safe JSON/NPZ cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable

from .errors import (
    EmptyLibraryError,
    ImageDecodeError,
    InputValidationError,
    InvalidEmbeddingError,
    MultipleFacesError,
    NoFaceError,
)
from .extractor import FeatureExtractor, face_diagnostics
from .model import MODEL_NAME


SUPPORTED_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)
CACHE_SCHEMA_VERSION = 1
CACHE_DIAGNOSTICS_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = "3.0"
EMBEDDING_DIMENSION = 512


def _emit_progress(reporter, stage: str, status: str, message: str, **kwargs: Any) -> None:
    """Keep display/logging failures outside the frozen identity algorithm."""

    if reporter is None:
        return
    try:
        reporter.emit(stage, status, message, **kwargs)
    except Exception:
        return


def _sort_key(path: Path) -> tuple[str, str]:
    return path.name.casefold(), path.name


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def default_cache_dir() -> Path:
    """Choose a user cache root that can never default to the source directory."""

    configured = os.environ.get("IDENTITY_SEARCH_CACHE_ROOT")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "FaceSourceSearch" / "identity_cache"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "face_source_search" / "identity_cache"


class IdentitySearcher:
    """Build, cache, and query a deterministic person-level identity index."""

    def __init__(
        self,
        library_path: os.PathLike[str] | str,
        *,
        cache_dir: os.PathLike[str] | str | None = None,
        model_root: os.PathLike[str] | str | None = None,
        model_fingerprint: str = "buffalo_l-unverified",
        device: str = "cpu",
        ctx_id: int | None = None,
        det_size: tuple[int, int] = (1280, 1280),
        extractor=None,
        progress_reporter=None,
    ) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:  # pragma: no cover - caught by CLI
            from .errors import DependencyError

            raise DependencyError("缺少数值依赖: numpy") from exc

        self.np = np
        self.library_argument = str(library_path)
        self.library_path = Path(library_path).expanduser().resolve(strict=False)
        if not self.library_path.is_dir():
            raise InputValidationError(f"身份库目录不存在: {library_path}")

        self.cache_dir = Path(cache_dir or default_cache_dir()).expanduser().resolve(
            strict=False
        )
        if _is_relative_to(self.cache_dir, self.library_path):
            raise InputValidationError(
                f"缓存目录不得位于身份库内部: {self.cache_dir}"
            )

        self.model_fingerprint = str(model_fingerprint)
        self.device = device
        self.ctx_id = ctx_id
        self.det_size = (int(det_size[0]), int(det_size[1]))
        self.progress_reporter = progress_reporter
        self.extractor = extractor or FeatureExtractor(
            model_root=model_root,
            device=device,
            ctx_id=ctx_id,
            det_size=self.det_size,
        )
        self.identities: dict[str, dict[str, Any]] = {}
        self.diagnostics: dict[str, Any] = {
            "cache_status": "not_checked",
            "skipped_images": [],
            "valid_identities": 0,
        }
        _emit_progress(
            self.progress_reporter,
            "check_identity_library",
            "running",
            "正在检查身份库。",
            progress_percent=None,
            details={"library": str(self.library_path)},
            force=True,
        )
        self._manifest = self._scan_library_manifest()
        _emit_progress(
            self.progress_reporter,
            "identity_library_checked",
            "success",
            "身份库文件清单检查完成。",
            progress_percent=None,
            details={
                "library_identity_count": len(
                    {item["identity_id"] for item in self._manifest}
                ),
                "library_image_count": len(self._manifest),
            },
            force=True,
        )
        self._fingerprint = self._index_fingerprint(self._manifest)
        self._load_or_build_index()

    def _scan_library_manifest(self) -> list[dict[str, Any]]:
        manifest: list[dict[str, Any]] = []
        person_dirs = sorted(
            (item for item in self.library_path.iterdir() if item.is_dir()),
            key=_sort_key,
        )
        for person_dir in person_dirs:
            image_paths = sorted(
                (
                    item
                    for item in person_dir.iterdir()
                    if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
                ),
                key=_sort_key,
            )
            for image_path in image_paths:
                stat = image_path.stat()
                digest = hashlib.sha256()
                with image_path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                manifest.append(
                    {
                        "identity_id": person_dir.name,
                        "relative_path": image_path.relative_to(
                            self.library_path
                        ).as_posix(),
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": digest.hexdigest(),
                    }
                )
        return manifest

    def _index_fingerprint(self, manifest: list[dict[str, Any]]) -> str:
        payload = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "files": manifest,
            "model_fingerprint": self.model_fingerprint,
            "det_size": list(self.det_size),
            "embedding_policy": "l2-normalized/person-mean-v1",
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get_cache_paths(self) -> tuple[Path, Path]:
        """Return the stable JSON metadata and NPZ data paths for this library."""

        library_key = hashlib.sha256(
            str(self.library_path).encode("utf-8")
        ).hexdigest()[:24]
        base = f"identity_index_{library_key}"
        return self.cache_dir / f"{base}.json", self.cache_dir / f"{base}.npz"

    def _load_or_build_index(self) -> None:
        metadata_path, data_path = self.get_cache_paths()
        _emit_progress(
            self.progress_reporter,
            "load_identity_cache",
            "running",
            "正在载入或校验身份库特征缓存。",
            progress_percent=None,
            details={"manifest_image_count": len(self._manifest)},
            force=True,
        )
        if not (metadata_path.is_file() and data_path.is_file()):
            self._adopt_relocated_cache(metadata_path, data_path)
        if metadata_path.is_file() and data_path.is_file():
            try:
                if self._load_cache(metadata_path, data_path):
                    self.diagnostics["cache_status"] = "hit"
                    self.diagnostics["valid_identities"] = len(self.identities)
                    _emit_progress(
                        self.progress_reporter,
                        "identity_cache_ready",
                        "success",
                        "身份库特征缓存校验完成，可复用。",
                        progress_percent=None,
                        details={
                            "cache_status": "hit",
                            "valid_identity_count": len(self.identities),
                            "valid_embedding_count": sum(
                                len(data["features"]) for data in self.identities.values()
                            ),
                        },
                        force=True,
                    )
                    return
                self.diagnostics["cache_status"] = "stale_rebuilt"
            except Exception as exc:
                self.diagnostics["cache_status"] = "corrupt_rebuilt"
                self.diagnostics["cache_error"] = f"{type(exc).__name__}: {exc}"
                print(
                    f"缓存损坏，将安全重建: {metadata_path}",
                    file=sys.stderr,
                )
        else:
            self.diagnostics["cache_status"] = "miss_built"

        self._build_index()
        self._write_cache(metadata_path, data_path)
        self.diagnostics["valid_identities"] = len(self.identities)
        _emit_progress(
            self.progress_reporter,
            "identity_cache_ready",
            "success",
            "身份库特征缓存已建立。",
            progress_percent=None,
            details={
                "cache_status": self.diagnostics["cache_status"],
                "valid_identity_count": len(self.identities),
                "valid_embedding_count": sum(
                    len(data["features"]) for data in self.identities.values()
                ),
                "skipped_library_image_count": len(
                    self.diagnostics.get("skipped_images", [])
                ),
            },
            force=True,
        )

    def _adopt_relocated_cache(self, metadata_path: Path, data_path: Path) -> bool:
        """Reuse an index built before the library directory was moved.

        Cache file names hash the absolute library path, while the fingerprint
        covers only relative paths, content hashes and the model. An equal
        fingerprint therefore means the same index; ``_load_cache`` still
        validates the copy afterwards.
        """

        if not self.cache_dir.is_dir():
            return False
        for candidate in sorted(self.cache_dir.glob("identity_index_*.json")):
            candidate_data = candidate.with_suffix(".npz")
            if candidate == metadata_path or not candidate_data.is_file():
                continue
            try:
                with candidate.open("r", encoding="utf-8") as stream:
                    metadata = json.load(stream)
            except (OSError, ValueError):
                continue
            if not isinstance(metadata, dict):
                continue
            if metadata.get("fingerprint") != self._fingerprint:
                continue
            # Copy the NPZ first: a metadata file without its data reads as a miss.
            shutil.copyfile(candidate_data, data_path)
            shutil.copyfile(candidate, metadata_path)
            return True
        return False

    def _load_cache(self, metadata_path: Path, data_path: Path) -> bool:
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            return False
        if metadata.get("fingerprint") != self._fingerprint:
            return False
        if (
            metadata.get("diagnostics_schema_version")
            != CACHE_DIAGNOSTICS_SCHEMA_VERSION
        ):
            # A legacy score cache may be valid but cannot explain which
            # references were skipped. Rebuild once so diagnostics become
            # evidence rather than guesses, while keeping the frozen embedding
            # fingerprint and score algorithm unchanged.
            return False
        entries = metadata.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("缓存 entries 为空或格式错误")
        skipped_images = metadata.get("skipped_images")
        if not isinstance(skipped_images, list):
            raise ValueError("缓存 skipped_images 格式错误")

        manifest_identity = {
            item["relative_path"]: item["identity_id"] for item in self._manifest
        }
        manifest_order = {
            item["relative_path"]: index for index, item in enumerate(self._manifest)
        }
        seen_paths: set[str] = set()
        previous_order = -1
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("缓存 entry 不是对象")
            identity_id = entry.get("identity_id")
            identity_name = entry.get("identity_name")
            relative_path = entry.get("relative_path")
            if not all(
                isinstance(value, str)
                for value in (identity_id, identity_name, relative_path)
            ):
                raise ValueError("缓存 entry 字段类型错误")
            if relative_path not in manifest_identity:
                raise ValueError("缓存 entry 不属于当前图库清单")
            if identity_id != manifest_identity[relative_path] or identity_name != identity_id:
                raise ValueError("缓存 entry 身份语义与图库目录不一致")
            if relative_path in seen_paths:
                raise ValueError("缓存 entry 路径重复")
            current_order = manifest_order[relative_path]
            if current_order <= previous_order:
                raise ValueError("缓存 entry 顺序不确定或被篡改")
            seen_paths.add(relative_path)
            previous_order = current_order

        skipped_paths: set[str] = set()
        for item in skipped_images:
            if not isinstance(item, dict):
                raise ValueError("缓存 skipped_images 项不是对象")
            relative_path = item.get("relative_path")
            reason = item.get("reason")
            message = item.get("message")
            if not all(isinstance(value, str) for value in (relative_path, reason, message)):
                raise ValueError("缓存 skipped_images 字段类型错误")
            if relative_path not in manifest_identity:
                raise ValueError("缓存 skipped_images 路径不属于当前图库清单")
            if relative_path in seen_paths or relative_path in skipped_paths:
                raise ValueError("缓存有效/跳过参考图路径重复")
            skipped_paths.add(relative_path)
        if seen_paths | skipped_paths != set(manifest_identity):
            raise ValueError("缓存未完整覆盖当前图库清单")

        expected_data_sha256 = metadata.get("data_sha256")
        if not isinstance(expected_data_sha256, str) or len(expected_data_sha256) != 64:
            raise ValueError("缓存缺少 NPZ SHA-256")
        actual_data_sha256 = self._hash_file(data_path)
        if actual_data_sha256 != expected_data_sha256:
            raise ValueError("缓存 NPZ SHA-256 不一致")

        # ``allow_pickle=False`` is the security boundary: NPZ cannot execute an
        # attacker-controlled pickle payload.
        with self.np.load(data_path, allow_pickle=False) as archive:
            embeddings = self.np.asarray(archive["embeddings"], dtype=self.np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(entries):
            raise ValueError("缓存特征矩阵与元数据数量不一致")
        if not self.np.all(self.np.isfinite(embeddings)):
            raise ValueError("缓存特征包含非有限值")

        identities: dict[str, dict[str, Any]] = {}
        for row, entry in zip(embeddings, entries):
            identity_id = entry["identity_id"]
            relative_path = entry["relative_path"]
            data = identities.setdefault(
                identity_id,
                {"name": entry.get("identity_name", identity_id), "features": [], "paths": []},
            )
            data["features"].append(row.copy())
            data["paths"].append(relative_path)
        if not identities:
            raise ValueError("缓存没有有效身份")
        self.identities = identities
        self.diagnostics["skipped_images"] = [dict(item) for item in skipped_images]
        return True

    def _build_index(self) -> None:
        identities: dict[str, dict[str, Any]] = {}
        skipped: list[dict[str, str]] = []
        total = len(self._manifest)
        for current, item in enumerate(self._manifest, start=1):
            absolute_path = self.library_path / Path(item["relative_path"])
            try:
                feature, _ = self.extractor.extract(absolute_path)
            except (
                ImageDecodeError,
                NoFaceError,
                MultipleFacesError,
                InvalidEmbeddingError,
            ) as exc:
                skipped.append(
                    {
                        "relative_path": item["relative_path"],
                        "reason": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                _emit_progress(
                    self.progress_reporter,
                    "build_identity_cache",
                    "running",
                    f"正在提取身份库参考图特征：{current} / {total}",
                    progress_current=current,
                    progress_total=total,
                    progress_percent=None,
                    details={
                        "processed_reference_images": current,
                        "library_image_count": total,
                        "valid_embeddings_so_far": sum(
                            len(data["features"]) for data in identities.values()
                        ),
                        "skipped_images_so_far": len(skipped),
                    },
                )
                continue
            identity_id = item["identity_id"]
            data = identities.setdefault(
                identity_id,
                {"name": identity_id, "features": [], "paths": []},
            )
            data["features"].append(
                self.np.asarray(feature, dtype=self.np.float32).reshape(-1)
            )
            data["paths"].append(item["relative_path"])
            _emit_progress(
                self.progress_reporter,
                "build_identity_cache",
                "running",
                f"正在提取身份库参考图特征：{current} / {total}",
                progress_current=current,
                progress_total=total,
                progress_percent=None,
                details={
                    "processed_reference_images": current,
                    "library_image_count": total,
                    "valid_embeddings_so_far": sum(
                        len(value["features"]) for value in identities.values()
                    ),
                    "skipped_images_so_far": len(skipped),
                },
            )

        self.diagnostics["skipped_images"] = skipped
        if not identities:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise EmptyLibraryError(
                "身份库没有至少一个可用身份：每个身份需有一张可解码、"
                f"且恰好包含一张人脸的参考图。支持格式: {supported}",
                diagnostics={
                    "library_identity_count": 0,
                    "library_image_count": len(self._manifest),
                    "valid_embedding_count": 0,
                    "skipped_library_image_count": len(skipped),
                    "skipped_images": skipped,
                },
            )
        dimensions = {
            int(feature.shape[0])
            for data in identities.values()
            for feature in data["features"]
        }
        if len(dimensions) != 1:
            raise ValueError(f"参考特征维度不一致: {sorted(dimensions)}")
        self.identities = identities

    def _write_cache(self, metadata_path: Path, data_path: Path) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, str]] = []
        features: list[Any] = []
        for identity_id, data in self.identities.items():
            for feature, relative_path in zip(data["features"], data["paths"]):
                entries.append(
                    {
                        "identity_id": identity_id,
                        "identity_name": data["name"],
                        "relative_path": relative_path,
                    }
                )
                features.append(feature)
        matrix = self.np.stack(features).astype(self.np.float32, copy=False)
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "diagnostics_schema_version": CACHE_DIAGNOSTICS_SCHEMA_VERSION,
            "fingerprint": self._fingerprint,
            "entries": entries,
            "skipped_images": [
                dict(item) for item in self.diagnostics.get("skipped_images", [])
            ],
        }

        temp_data = data_path.with_name(f".{data_path.name}.{os.getpid()}.tmp.npz")
        temp_metadata = metadata_path.with_name(
            f".{metadata_path.name}.{os.getpid()}.tmp"
        )
        try:
            self.np.savez_compressed(temp_data, embeddings=matrix)
            metadata["data_sha256"] = self._hash_file(temp_data)
            with temp_metadata.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temp_data, data_path)
            os.replace(temp_metadata, metadata_path)
        finally:
            for temp_path in (temp_data, temp_metadata):
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _display_reference_path(self, relative_path: str) -> str:
        return str(Path(self.library_argument) / Path(relative_path))

    def search(
        self,
        query_path: os.PathLike[str] | str,
        top_k: int = 5,
        *,
        progress_reporter=None,
        artifact_root: os.PathLike[str] | str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Search by person-level mean cosine similarity (the frozen algorithm)."""

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise InputValidationError(f"top-k 必须为正整数，实际值: {top_k}")
        start_time = time.perf_counter()
        reporter = progress_reporter or self.progress_reporter
        _emit_progress(
            reporter,
            "detect_query_face",
            "running",
            "正在检测查询图片中的人脸。",
            progress_percent=None,
            details={"strict_single_face_required": True},
            force=True,
        )
        query_feature, query_face = self.extractor.extract(query_path)
        query_face_details = face_diagnostics(query_face)
        embedding_dimension = int(query_feature.shape[0])
        embedding_l2_norm = float(self.np.linalg.norm(query_feature))
        _emit_progress(
            reporter,
            "detect_query_face",
            "success",
            "查询图人脸检测完成：检测到 1 张有效人脸。",
            progress_percent=None,
            details={
                "query_face_count": 1,
                "selected_face_bbox": query_face_details.get("bbox"),
                "face_detection_score": query_face_details.get("det_score"),
            },
            force=True,
        )
        has_five_landmarks = (
            isinstance(query_face_details.get("keypoints_5"), list)
            and len(query_face_details["keypoints_5"]) == 5
        )
        _emit_progress(
            reporter,
            "locate_query_landmarks",
            "success" if has_five_landmarks else "warning",
            (
                "查询图五点关键点定位完成。"
                if has_five_landmarks
                else "查询人脸已检测，但模型未返回可用五点关键点。"
            ),
            progress_percent=None,
            details={
                "landmark_count": query_face_details.get("landmark_count", 0),
                "keypoints_5": query_face_details.get("keypoints_5"),
            },
            force=True,
        )
        _emit_progress(
            reporter,
            "extract_identity_embedding",
            "success",
            f"已提取 {embedding_dimension} 维归一化身份特征。",
            progress_percent=None,
            details={
                "query_face_count": 1,
                "selected_face_bbox": query_face_details.get("bbox"),
                "face_detection_score": query_face_details.get("det_score"),
                "embedding_dimension": embedding_dimension,
                "embedding_normalization": "l2",
                "embedding_l2_norm": embedding_l2_norm,
            },
            force=True,
        )

        results: list[dict[str, Any]] = []
        identity_total = len(self.identities)
        valid_embedding_total = sum(
            len(data["features"]) for data in self.identities.values()
        )
        _emit_progress(
            reporter,
            "load_identity_features",
            "success",
            "身份库归一化特征已载入，准备逐参考图比较。",
            progress_percent=None,
            details={
                "library_identity_count": identity_total,
                "valid_embedding_count": valid_embedding_total,
                "cache_status": self.diagnostics.get("cache_status"),
            },
            force=True,
        )
        compared_reference_count = 0
        _emit_progress(
            reporter,
            "compare_reference_images",
            "running",
            "正在逐张计算查询特征与有效参考图的余弦相似度。",
            progress_percent=None,
            details={"valid_embedding_count": valid_embedding_total},
            force=True,
        )
        for identity_current, (identity_id, data) in enumerate(
            self.identities.items(), start=1
        ):
            similarities: list[float] = []
            for reference_index, (reference, relative_path) in enumerate(
                zip(data["features"], data["paths"], strict=True), start=1
            ):
                # Frozen scoring primitive: dot product of two L2-normalized
                # embeddings, identical to the pre-visualization list
                # comprehension and therefore still cosine similarity.
                similarities.append(float(self.np.dot(query_feature, reference)))
                compared_reference_count += 1
                _emit_progress(
                    reporter,
                    "compare_reference_images",
                    "running",
                    (
                        "正在逐张比较参考图："
                        f"{compared_reference_count} / {valid_embedding_total}"
                    ),
                    progress_current=compared_reference_count,
                    progress_total=valid_embedding_total,
                    details={
                        "identity_id": identity_id,
                        "reference_index_within_identity": reference_index,
                        "reference_count_within_identity": len(data["features"]),
                        "reference_relative_path": Path(relative_path).as_posix(),
                        "similarity": similarities[-1],
                    },
                )
            _emit_progress(
                reporter,
                "aggregate_identity_scores",
                "running",
                f"正在按人员聚合参考图相似度：{identity_current} / {identity_total}",
                progress_percent=None,
                details={
                    "identity_id": identity_id,
                    "valid_reference_count": len(similarities),
                    "aggregation_method": "mean",
                },
            )
            average_similarity = float(self.np.mean(similarities))
            best_index = int(self.np.argmax(similarities))
            skipped_references = [
                {
                    "reference_image": self._display_reference_path(
                        str(item["relative_path"])
                    ),
                    "reference_relative_path": Path(
                        str(item["relative_path"])
                    ).as_posix(),
                    "reference_file_name": Path(str(item["relative_path"])).name,
                    "reason": item.get("reason"),
                    "message": item.get("message"),
                    "status": "skipped",
                }
                for item in self.diagnostics.get("skipped_images", [])
                if Path(str(item.get("relative_path", ""))).parts[:1]
                == (identity_id,)
            ]
            results.append(
                {
                    "identity_id": identity_id,
                    "identity_name": data["name"],
                    "score": average_similarity,
                    "best_ref_image": self._display_reference_path(
                        data["paths"][best_index]
                    ),
                    "best_reference_score": similarities[best_index],
                    "reference_count": len(similarities),
                    "valid_reference_count": len(similarities),
                    "skipped_reference_count": len(skipped_references),
                    "best_reference_relative_path": Path(
                        data["paths"][best_index]
                    ).as_posix(),
                    "best_reference_file_name": Path(
                        data["paths"][best_index]
                    ).name,
                    "reference_scores": [
                        {
                            "reference_image": self._display_reference_path(relative_path),
                            "reference_relative_path": Path(relative_path).as_posix(),
                            "reference_file_name": Path(relative_path).name,
                            "similarity": similarity,
                            "status": "valid",
                        }
                        for relative_path, similarity in zip(
                            data["paths"], similarities, strict=True
                        )
                    ],
                    "skipped_reference_images": skipped_references,
                }
            )
            _emit_progress(
                reporter,
                "compare_identities",
                "running",
                f"正在比较身份候选：{identity_current} / {identity_total}",
                progress_current=identity_current,
                progress_total=identity_total,
                details={
                    "compared_identity_count": identity_current,
                    "library_identity_count": identity_total,
                    "current_identity_id": identity_id,
                },
            )

        # Identity ID is an explicit tie-breaker, so equal-score output is stable.
        _emit_progress(
            reporter,
            "compare_reference_images",
            "success",
            "逐参考图余弦相似度比较完成。",
            progress_percent=None,
            details={
                "compared_reference_count": compared_reference_count,
                "valid_embedding_count": valid_embedding_total,
            },
            force=True,
        )
        _emit_progress(
            reporter,
            "aggregate_identity_scores",
            "success",
            "人员级平均余弦相似度聚合完成。",
            progress_percent=None,
            details={
                "aggregated_identity_count": len(results),
                "aggregation_method": "mean",
            },
            force=True,
        )
        _emit_progress(
            reporter,
            "rank_identity_candidates",
            "running",
            "正在按人员级平均余弦相似度生成 Top-K 候选排序。",
            progress_percent=None,
            details={"top_k": top_k, "identity_count": len(results)},
            force=True,
        )
        results.sort(key=lambda item: (-item["score"], item["identity_id"]))
        top1_top2_margin = (
            float(results[0]["score"] - results[1]["score"])
            if len(results) >= 2
            else None
        )
        candidates = [
            {
                "rank": rank,
                "identity_id": item["identity_id"],
                "identity_name": item["identity_name"],
                "score": item["score"],
                "reference_image": item["best_ref_image"],
                "reference_count": item["reference_count"],
                "mean_similarity": item["score"],
                "best_reference_score": item["best_reference_score"],
                "best_reference_image": item["best_ref_image"],
                "best_reference_relative_path": item[
                    "best_reference_relative_path"
                ],
                "best_reference_file_name": item["best_reference_file_name"],
                "reference_scores": item["reference_scores"],
                "valid_reference_count": item["valid_reference_count"],
                "skipped_reference_count": item["skipped_reference_count"],
                "skipped_reference_images": item["skipped_reference_images"],
                "gap_from_top1": float(results[0]["score"] - item["score"]),
            }
            for rank, item in enumerate(results[:top_k], start=1)
        ]
        valid_embedding_count = sum(
            len(data["features"]) for data in self.identities.values()
        )
        library_image_count = len(self._manifest)
        warnings: list[str] = []
        if embedding_dimension != EMBEDDING_DIMENSION:
            warnings.append(
                f"模型实际返回 {embedding_dimension} 维特征；冻结基线期望 {EMBEDDING_DIMENSION} 维。"
            )
        if not has_five_landmarks:
            warnings.append("模型未返回可用五点关键点，本次无法生成标准对齐证据图。")
        top1_reason = None
        if candidates:
            top1 = candidates[0]
            if top1_top2_margin is None:
                top1_reason = (
                    f"{top1['identity_id']} 的人员级平均余弦相似度为 "
                    f"{top1['mean_similarity']:.12g}；当前库仅有一个有效身份候选。"
                )
            else:
                top1_reason = (
                    f"{top1['identity_id']} 的人员级平均余弦相似度为 "
                    f"{top1['mean_similarity']:.12g}，与第二候选差值为 "
                    f"{top1_top2_margin:.12g}，因此排在当前身份库第一位。"
                )
        runtime_seconds = time.perf_counter() - start_time
        _emit_progress(
            reporter,
            "rank_identity_candidates",
            "success",
            "Top-K 身份候选排序完成。",
            progress_percent=None,
            details={
                "returned_candidate_count": len(candidates),
                "top1_identity_id": candidates[0]["identity_id"] if candidates else None,
                "top1_top2_margin": top1_top2_margin,
            },
            force=True,
        )

        visual_evidence: dict[str, Any] = {
            "module": "identity",
            "status": "not_requested",
            "query_face_count": 1,
            "query_face": query_face_details,
            "embedding_status": "completed",
            "embedding_dimension": embedding_dimension,
            "embedding_normalization": "l2",
            "embedding_l2_norm": embedding_l2_norm,
            "top_candidates": candidates,
            "top1_top2_margin": top1_top2_margin,
            "aggregation_method": "mean",
            "per_reference_scores": [
                {
                    "rank": candidate["rank"],
                    "identity_id": candidate["identity_id"],
                    "mean_similarity": candidate["mean_similarity"],
                    "best_reference_score": candidate["best_reference_score"],
                    "valid_reference_count": candidate["valid_reference_count"],
                    "skipped_reference_count": candidate["skipped_reference_count"],
                    "reference_scores": candidate["reference_scores"],
                    "skipped_reference_images": candidate[
                        "skipped_reference_images"
                    ],
                }
                for candidate in candidates
            ],
        }
        artifact_manifest: list[dict[str, Any]] = []
        if artifact_root is not None:
            _emit_progress(
                reporter,
                "align_query_face",
                "running",
                "正在依据本次五点关键点生成查询人脸对齐证据。",
                progress_percent=None,
                details={
                    "landmark_count": query_face_details.get("landmark_count", 0),
                    "alignment_size": [112, 112],
                },
                force=True,
            )
            _emit_progress(
                reporter,
                "generate_identity_visuals",
                "running",
                "正在生成查询人脸、候选聚合和最佳参考对比证据。",
                progress_percent=None,
                details={"top_candidate_count": len(candidates)},
                force=True,
            )
            try:
                from .evidence import build_identity_visual_evidence

                generated, artifact_manifest, evidence_warnings = (
                    build_identity_visual_evidence(
                        query_path=Path(query_path),
                        query_face=query_face,
                        query_feature=query_feature,
                        candidates=candidates,
                        extractor=self.extractor,
                        task_root=Path(artifact_root),
                        task_id=task_id,
                    )
                )
                card_paths = {
                    int(item["rank"]): item.get("path")
                    for item in generated.get("candidate_card_images", [])
                    if isinstance(item, dict)
                    and isinstance(item.get("rank"), int)
                    and item.get("path")
                }
                for candidate in candidates:
                    candidate["candidate_card_image"] = card_paths.get(
                        int(candidate["rank"])
                    )
                generated.update(
                    {
                        "module": "identity",
                        "top_candidates": candidates,
                        "top1_top2_margin": top1_top2_margin,
                        "aggregation_method": "mean",
                        "per_reference_scores": visual_evidence[
                            "per_reference_scores"
                        ],
                    }
                )
                visual_evidence = generated
                warnings.extend(evidence_warnings)
                alignment_generated = bool(
                    visual_evidence.get("aligned_face_image")
                )
                _emit_progress(
                    reporter,
                    "align_query_face",
                    "success" if alignment_generated else "warning",
                    (
                        "查询人脸标准对齐证据已生成。"
                        if alignment_generated
                        else "查询人脸对齐证据未生成，原身份检索结果不受影响。"
                    ),
                    progress_percent=None,
                    details={
                        "aligned_face_image": visual_evidence.get(
                            "aligned_face_image"
                        ),
                        "alignment_size": visual_evidence.get(
                            "aligned_face_size"
                        ),
                    },
                    force=True,
                )
                _emit_progress(
                    reporter,
                    "generate_identity_visuals",
                    "success" if not evidence_warnings else "warning",
                    (
                        "身份检索任务级可视化证据已生成。"
                        if not evidence_warnings
                        else "身份排序成功，部分可视化证据降级。"
                    ),
                    progress_percent=None,
                    details={
                        "artifact_count": len(artifact_manifest),
                        "evidence_status": visual_evidence.get("status"),
                        "warning_count": len(evidence_warnings),
                    },
                    force=True,
                )
            except Exception as exc:
                evidence_error = (
                    f"身份检索可视化证据生成失败: {type(exc).__name__}: {exc}"
                )
                warnings.append(evidence_error)
                visual_evidence["status"] = "failed"
                visual_evidence["warnings"] = [evidence_error]
                _emit_progress(
                    reporter,
                    "align_query_face",
                    "warning",
                    "查询人脸对齐证据生成失败，原身份检索结果不受影响。",
                    progress_percent=None,
                    details={"error": evidence_error},
                    force=True,
                )
                _emit_progress(
                    reporter,
                    "generate_identity_visuals",
                    "warning",
                    "可视化证据生成失败，原身份候选排序已保留。",
                    progress_percent=None,
                    details={"error": evidence_error},
                    force=True,
                )
        _emit_progress(
            reporter,
            "identity_ranking_complete",
            "success",
            "身份候选排序完成。",
            progress_percent=100.0,
            details={
                "returned_candidate_count": len(candidates),
                "library_identity_count": identity_total,
                "top1_identity_id": candidates[0]["identity_id"] if candidates else None,
                "top1_top2_margin": top1_top2_margin,
            },
            force=True,
        )
        _emit_progress(
            reporter,
            "identity_complete",
            "success",
            "身份检索模块完成：候选分数、排序与任务证据已结构化保存。",
            progress_percent=100.0,
            details={
                "returned_candidate_count": len(candidates),
                "artifact_count": len(artifact_manifest),
                "visual_evidence_status": visual_evidence.get("status"),
            },
            force=True,
        )
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "success",
            "query_image": str(query_path),
            "candidates": candidates,
            "runtime_seconds": runtime_seconds,
            "error_message": None,
            "method": {
                "name": "insightface_closed_set_person_mean_cosine",
                "model_name": MODEL_NAME,
                "face_detection_size": list(self.det_size),
                "embedding_dimension": embedding_dimension,
                "embedding_normalization": "l2",
                "similarity": "cosine_similarity",
                "aggregation_method": "mean",
                "open_set_rejection_enabled": False,
            },
            "metrics": {
                "query_face_count": 1,
                "selected_face_bbox": query_face_details.get("bbox"),
                "face_detection_score": query_face_details.get("det_score"),
                "selected_face_keypoints_5": query_face_details.get(
                    "keypoints_5"
                ),
                "embedding_dimension": embedding_dimension,
                "embedding_l2_norm": embedding_l2_norm,
                "library_identity_count": identity_total,
                "library_image_count": library_image_count,
                "valid_embedding_count": valid_embedding_count,
                "skipped_library_image_count": max(
                    0, library_image_count - valid_embedding_count
                ),
                "top1_top2_margin": top1_top2_margin,
            },
            "diagnostics": {
                "query_face_count": 1,
                "detected_faces": [query_face_details],
                "selected_face_bbox": query_face_details.get("bbox"),
                "face_detection_score": query_face_details.get("det_score"),
                "selected_face_keypoints_5": query_face_details.get(
                    "keypoints_5"
                ),
                "embedding_dimension": embedding_dimension,
                "embedding_l2_norm": embedding_l2_norm,
                "library_identity_count": identity_total,
                "library_image_count": library_image_count,
                "valid_embedding_count": valid_embedding_count,
                "skipped_library_images": self.diagnostics.get(
                    "skipped_images", []
                ),
                "cache_status": self.diagnostics.get("cache_status"),
                "top1_top2_margin": top1_top2_margin,
            },
            "explanation": {
                "summary": "这是当前身份库中的候选排序，不是最终身份确认。",
                "ranking_basis": (
                    "查询图512维L2归一化特征与每张有效参考图计算余弦相似度，"
                    "同一人员的相似度取均值后按人员排序。"
                ),
                "top1_reason": top1_reason,
            },
            "warnings": warnings,
            "visual_evidence": visual_evidence,
            "artifact_manifest": artifact_manifest,
            "limitations": [
                "当前提供闭集候选排序，未启用未经独立验证的开放集拒识阈值。",
                "候选相似度不是身份概率，不能单独据此确认人物身份。",
                "查询图片必须恰好检测到一张人脸。",
            ],
            "runtime": {
                "seconds": runtime_seconds,
                "device": self.device,
                "ctx_id": self.ctx_id,
                "cache_status": self.diagnostics.get("cache_status"),
            },
            "provenance": {
                "library_path": str(self.library_path),
                "library_fingerprint": self._fingerprint,
                "model_name": MODEL_NAME,
                "model_fingerprint": self.model_fingerprint,
            },
        }
