"""Offline ``buffalo_l`` model location and integrity policy.

Only Python's standard library is imported here so model preflight can produce a
structured CLI error even when the inference stack is not installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Mapping

from .errors import ModelPreflightError


MODEL_NAME = "buffalo_l"

# Frozen from the machine cache audited on 2026-08-19. The root passed to
# InsightFace is the directory *above* ``models`` (normally ~/.insightface).
EXPECTED_MODEL_FILES: Mapping[str, tuple[int, str]] = {
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


@dataclass(frozen=True)
class ModelFile:
    """A verified local model file."""

    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ModelPreflight:
    """Result of a successful local-only model preflight."""

    root: Path
    model_dir: Path
    files: tuple[ModelFile, ...]
    fingerprint: str


def default_model_root() -> Path:
    """Return the InsightFace root, not ``models/buffalo_l`` itself."""

    configured = os.environ.get("INSIGHTFACE_ROOT")
    return Path(configured).expanduser() if configured else Path.home() / ".insightface"


def model_directory(model_root: os.PathLike[str] | str) -> Path:
    """Resolve ``<InsightFace root>/models/buffalo_l``."""

    return Path(model_root).expanduser().resolve(strict=False) / "models" / MODEL_NAME


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading a model into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_model(
    model_root: os.PathLike[str] | str,
    *,
    verify_hashes: bool = True,
    expected_files: Mapping[str, tuple[int, str]] = EXPECTED_MODEL_FILES,
) -> ModelPreflight:
    """Verify that the frozen model package is complete and local.

    This function never downloads anything. A missing, truncated, or unexpected
    model is reported before InsightFace is imported.
    """

    root = Path(model_root).expanduser().resolve(strict=False)
    directory = model_directory(root)
    problems: list[str] = []
    verified: list[ModelFile] = []

    if not directory.is_dir():
        raise ModelPreflightError(
            "buffalo_l 模型目录不存在: "
            f"{directory}。请准备本地模型；程序不会自动联网下载。"
        )

    actual_inventory = sorted(item.name for item in directory.iterdir() if item.is_file())
    unexpected = sorted(set(actual_inventory) - set(expected_files))
    if unexpected:
        problems.append("存在清单外文件: " + ", ".join(unexpected))

    for name in sorted(expected_files):
        expected_size, expected_sha256 = expected_files[name]
        path = directory / name
        if not path.is_file():
            problems.append(f"缺少 {name}")
            continue
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            problems.append(f"无法读取 {name}: {exc}")
            continue
        if actual_size != expected_size:
            problems.append(
                f"{name} 大小不符（实际 {actual_size}，预期 {expected_size}）"
            )
            continue
        try:
            actual_sha256 = sha256_file(path) if verify_hashes else expected_sha256
        except OSError as exc:
            problems.append(f"无法读取 {name}: {exc}")
            continue
        if verify_hashes and actual_sha256.lower() != expected_sha256.lower():
            problems.append(f"{name} SHA-256 不符")
            continue
        verified.append(ModelFile(name, actual_size, actual_sha256.lower()))

    if problems:
        raise ModelPreflightError(
            "buffalo_l 模型预检失败（不会自动联网下载）: " + "；".join(problems)
        )

    fingerprint_source = "\n".join(
        f"{item.name}:{item.size_bytes}:{item.sha256}" for item in verified
    )
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
    return ModelPreflight(root, directory, tuple(verified), fingerprint)
