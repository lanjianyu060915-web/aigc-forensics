"""Versioned tensor-only checkpoints with fixed constructors and file digests."""

import hashlib
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from .clip_adapt import ClipAdaptConfig, ClipAdaptVideoClassifier
from .clip_community_fusion import (
    ClipCommunityFusionClassifier,
    ClipCommunityFusionConfig,
)
from .community_forensics import (
    CommunityForensicsConfig,
    CommunityForensicsVideoClassifier,
)
from .ufd import UFDConfig, UFDVideoClassifier


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected_sha256: str) -> None:
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")


def create_model(architecture: str, config: dict[str, Any]) -> nn.Module:
    """Instantiate only built-in architectures; never resolve arbitrary classes."""
    if architecture == "ufd_video_v1":
        return UFDVideoClassifier(UFDConfig(**config))
    if architecture == "clip_adapt_video_v1":
        return ClipAdaptVideoClassifier(ClipAdaptConfig(**config))
    if architecture == "community_forensics_video_v1":
        return CommunityForensicsVideoClassifier(CommunityForensicsConfig(**config))
    if architecture == "video_clip_community_fusion_v1":
        return ClipCommunityFusionClassifier(ClipCommunityFusionConfig(**config))
    raise ValueError(f"unsupported architecture/config: {architecture}")


def load_checkpoint(
    path: Path, *, expected_sha256: str
) -> tuple[nn.Module, dict[str, Any]]:
    """Verify a trusted hash before safe deserialization and strict state loading."""
    verify_sha256(path, expected_sha256)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema_version",
        "architecture",
        "config",
        "score_direction",
        "state_dict",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("invalid checkpoint schema fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema version")
    if payload["score_direction"] != "higher_is_fake":
        raise ValueError("unsupported checkpoint score direction")
    if not isinstance(payload["architecture"], str) or not isinstance(
        payload["config"], dict
    ):
        raise TypeError("invalid checkpoint architecture/config")
    state = payload["state_dict"]
    if (
        not isinstance(state, dict)
        or not state
        or any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in state.items()
        )
    ):
        raise ValueError("state_dict must be a nonempty string-to-tensor mapping")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
        raise ValueError("checkpoint contains non-finite tensors")
    model = create_model(payload["architecture"], payload["config"])
    model.load_state_dict(state, strict=True)
    model.eval()
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    metadata["sha256"] = expected_sha256
    return model, cast(dict[str, Any], metadata)
