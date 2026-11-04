"""底图库侧特征缓存。

只避免对同一张库图重复做解码、人脸检测、全局特征和 SIFT，不改变任何数值：
特征按原始精度落盘，读回后逐字节相同，分数口径仍由 global_features.py 与
local_matcher.py 决定。文件的大小或修改时间变化时只重算该文件。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .face_mask import FaceMasker
from .global_features import extract_global
from .image_io import SUPPORTED_EXTENSIONS, read_image


FEATURE_FORMAT_VERSION = 1
FEATURE_KEYS = ("hist", "thumb", "edge", "border", "border_valid")
NO_BOX = (-1, -1, -1, -1)


@dataclass
class LibraryEntry:
    path: Path
    stamp: str
    shape: tuple[int, int]
    face_box: tuple[int, int, int, int] | None
    features: dict[str, np.ndarray]


@dataclass
class CacheStats:
    index_hits: int = 0
    index_misses: int = 0
    sift_hits: int = 0
    sift_misses: int = 0
    rebuilt_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "index_hits": self.index_hits,
            "index_misses": self.index_misses,
            "sift_hits": self.sift_hits,
            "sift_misses": self.sift_misses,
            "rebuilt_reason": self.rebuilt_reason,
        }


def library_cache_key(library_path: Path) -> str:
    """只取库目录的最后两级，整个系统搬到别的盘后缓存仍可用。

    同名的不同库最多互相覆盖缓存：每个条目都按相对路径、大小和修改时间校验，不会用错特征。
    """

    resolved = Path(library_path).resolve(strict=False)
    token = f"{resolved.parent.name}/{resolved.name}".casefold()
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]


def _stamp(path: Path) -> str:
    info = path.stat()
    return f"{info.st_size}:{info.st_mtime_ns}"


def library_paths(directory: Path) -> list[Path]:
    """与 scan_library 相同的枚举与排序口径。"""

    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"library directory does not exist: {directory}")
    return [
        path
        for path in sorted(directory.rglob("*"), key=lambda p: str(p).lower())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class SourceFeatureCache:
    def __init__(self, cache_dir: Path, library_path: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.library_path = Path(library_path)
        self.library_key = library_cache_key(self.library_path)
        self.index_path = self.cache_dir / f"source_index_{self.library_key}.npz"
        self.sift_dir = self.cache_dir / f"sift_{self.library_key}"
        self.stats = CacheStats()
        self._signature = json.dumps(
            {"format": FEATURE_FORMAT_VERSION, "opencv": cv2.__version__},
            sort_keys=True,
        )

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if not self.index_path.is_file():
            self.stats.rebuilt_reason = "index_missing"
            return {}
        try:
            with np.load(self.index_path, allow_pickle=False) as data:
                if str(data["signature"]) != self._signature:
                    self.stats.rebuilt_reason = "signature_changed"
                    return {}
                records: dict[str, dict[str, Any]] = {}
                for row, relative in enumerate(data["relative_paths"].tolist()):
                    box = tuple(int(value) for value in data["boxes"][row])
                    records[str(relative)] = {
                        "stamp": str(data["stamps"][row]),
                        "shape": tuple(int(value) for value in data["shapes"][row]),
                        "box": None if box == NO_BOX else box,
                        "features": {key: data[key][row] for key in FEATURE_KEYS},
                    }
                for relative, stamp in zip(
                    data["skipped_paths"].tolist(),
                    data["skipped_stamps"].tolist(),
                    strict=True,
                ):
                    records[str(relative)] = {"stamp": str(stamp), "skipped": True}
                return records
        except (OSError, ValueError, KeyError, EOFError) as exc:
            self.stats.rebuilt_reason = f"index_unreadable:{type(exc).__name__}"
            return {}

    def _save_index(
        self, entries: list[LibraryEntry], skipped: list[tuple[str, str]]
    ) -> None:
        arrays: dict[str, Any] = {
            "signature": np.array(self._signature),
            "relative_paths": np.array(
                [entry.path.relative_to(self.library_path).as_posix() for entry in entries]
            ),
            "stamps": np.array([entry.stamp for entry in entries]),
            "shapes": np.array([entry.shape for entry in entries], dtype=np.int32),
            "boxes": np.array(
                [entry.face_box or NO_BOX for entry in entries], dtype=np.int32
            ),
            "skipped_paths": np.array([item[0] for item in skipped], dtype=str),
            "skipped_stamps": np.array([item[1] for item in skipped], dtype=str),
        }
        for key in FEATURE_KEYS:
            arrays[key] = np.stack([entry.features[key] for entry in entries])
        _atomic_savez(self.index_path, **arrays)

    def load_or_build(
        self,
        masker: FaceMasker,
        progress_callback: Callable[[int, int, Path, bool], None] | None = None,
    ) -> tuple[list[LibraryEntry], list[str]]:
        """返回与 scan_library 同序的库条目和无法解码的文件列表。"""

        paths = library_paths(self.library_path)
        records = self._load_index()
        entries: list[LibraryEntry] = []
        skipped_paths: list[str] = []
        skipped_records: list[tuple[str, str]] = []
        dirty = len(records) != len(paths)
        total = len(paths)
        for current, path in enumerate(paths, 1):
            relative = path.relative_to(self.library_path).as_posix()
            stamp = _stamp(path)
            record = records.get(relative)
            if record is not None and record["stamp"] == stamp:
                self.stats.index_hits += 1
                if record.get("skipped"):
                    skipped_paths.append(str(path))
                    skipped_records.append((relative, stamp))
                    continue
                entries.append(
                    LibraryEntry(path, stamp, record["shape"], record["box"], record["features"])
                )
                continue
            dirty = True
            self.stats.index_misses += 1
            image = read_image(path)
            if image is None:
                skipped_paths.append(str(path))
                skipped_records.append((relative, stamp))
            else:
                detected = masker.detect(image)
                box = tuple(int(value) for value in detected) if detected is not None else None
                mask = masker.mask(image.shape, box)
                entries.append(
                    LibraryEntry(
                        path,
                        stamp,
                        (int(image.shape[0]), int(image.shape[1])),
                        box,
                        extract_global(image, mask),
                    )
                )
            if progress_callback is not None:
                progress_callback(current, total, path, image is not None)
        if not entries:
            raise ValueError("library contains no decodable supported images")
        if dirty:
            self._save_index(entries, skipped_records)
        return entries, skipped_paths

    def _sift_path(self, entry: LibraryEntry) -> Path:
        token = f"{entry.path.relative_to(self.library_path).as_posix()}|{entry.stamp}|{self._signature}"
        return self.sift_dir / f"{hashlib.sha256(token.encode('utf-8')).hexdigest()[:32]}.npz"

    def _load_sift(self, path: Path) -> tuple[list[cv2.KeyPoint], np.ndarray | None] | None:
        try:
            with np.load(path, allow_pickle=False) as data:
                floats = data["keypoint_floats"]
                ints = data["keypoint_ints"]
                descriptors = data["descriptors"] if bool(data["has_descriptors"]) else None
        except (OSError, ValueError, KeyError, EOFError):
            return None
        keypoints = [
            cv2.KeyPoint(
                x=float(row[0]),
                y=float(row[1]),
                size=float(row[2]),
                angle=float(row[3]),
                response=float(row[4]),
                octave=int(extra[0]),
                class_id=int(extra[1]),
            )
            for row, extra in zip(floats, ints, strict=True)
        ]
        return keypoints, descriptors

    def sift_features(
        self, entry: LibraryEntry, sift: Any
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        path = self._sift_path(entry)
        if path.is_file():
            loaded = self._load_sift(path)
            if loaded is not None:
                self.stats.sift_hits += 1
                return loaded
        self.stats.sift_misses += 1
        image = self.read_image(entry)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(
            gray, FaceMasker.mask(image.shape, entry.face_box)
        )
        keypoints = list(keypoints or [])
        _atomic_savez(
            path,
            keypoint_floats=np.array(
                [[k.pt[0], k.pt[1], k.size, k.angle, k.response] for k in keypoints],
                dtype=np.float32,
            ).reshape(-1, 5),
            keypoint_ints=np.array(
                [[k.octave, k.class_id] for k in keypoints], dtype=np.int32
            ).reshape(-1, 2),
            has_descriptors=np.array(descriptors is not None),
            descriptors=(
                descriptors if descriptors is not None else np.zeros((0, 128), dtype=np.float32)
            ),
        )
        return keypoints, descriptors

    @staticmethod
    def read_image(entry: LibraryEntry) -> np.ndarray:
        image = read_image(entry.path)
        if image is None:
            raise ValueError(f"library image changed during search: {entry.path}")
        return image
