"""Fixed weighted logit fusion of adapted CLIP heads and Community Forensics."""

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from .clip_adapt import ClipAdaptConfig, ClipAdaptVideoClassifier
from .community_forensics import (
    CommunityForensicsConfig,
    CommunityForensicsVideoClassifier,
)
from .core import FeatureHead, HeadConfig


@dataclass(frozen=True)
class ClipCommunityFusionConfig:
    clip: dict[str, Any]
    community: dict[str, Any]
    clip_weight: float = 0.8
    clip_frames: int = 10
    clip_crops: int = 5
    community_frames: int = 4
    aggregation: str = "mean"
    refit_heads: int = 0
    refit_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.aggregation != "mean":
            raise ValueError("clip/community fusion requires mean logits")
        if not math.isfinite(self.clip_weight) or not 0 < self.clip_weight < 1:
            raise ValueError("clip/community fusion weight must be inside (0, 1)")
        if not math.isfinite(self.refit_weight) or not 0 < self.refit_weight < 1:
            raise ValueError("clip/community fusion refit weight must be inside (0, 1)")
        for name in ("clip_frames", "clip_crops", "community_frames"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"clip/community fusion {name} must be positive")
        if type(self.refit_heads) is not int or self.refit_heads < 0:
            raise ValueError("clip/community fusion refit_heads must be nonnegative")


class ClipCommunityFusionClassifier(nn.Module):
    """Combine raw video logits with fixed weights; nothing is fitted at inference.

    The CLIP score is the original head, optionally averaged with refit heads that
    share one backbone pass. Each branch keeps its own spatial view and normalization,
    so the scorer passes separately decoded frames. All weights are in one checkpoint.
    """

    def __init__(self, config: ClipCommunityFusionConfig) -> None:
        super().__init__()
        self.config = config
        self.clip = ClipAdaptVideoClassifier(ClipAdaptConfig(**config.clip))
        self.community = CommunityForensicsVideoClassifier(
            CommunityForensicsConfig(**config.community)
        )
        head = HeadConfig(
            self.clip.backbone.config.projection_dim, self.clip.config.hidden_dim, 0.0
        )
        self.refit_heads = nn.ModuleList(
            FeatureHead(head) for _ in range(config.refit_heads)
        )

    def _clip_score(self, frames: Tensor, frame_batch_size: int) -> Tensor:
        if not self.config.refit_heads:
            return cast(Tensor, self.clip(frames, frame_batch_size=frame_batch_size))
        batch, count = frames.shape[:2]
        device = next(self.parameters()).device
        features = torch.cat(
            [
                self.clip.backbone(pixel_values=chunk.to(device)).image_embeds
                for chunk in torch.split(frames.flatten(0, 1), frame_batch_size)
            ]
        ).reshape(batch, count, -1)
        original = self.clip.head(features).mean(dim=1)
        refit = torch.stack([head(features).mean(dim=1) for head in self.refit_heads])
        weight = self.config.refit_weight
        return cast(Tensor, (1 - weight) * original + weight * refit.mean(dim=0))

    def forward(
        self, clip_frames: Tensor, community_frames: Tensor, *, frame_batch_size: int
    ) -> Tensor:
        clip_score = self._clip_score(clip_frames, frame_batch_size)
        community_score = self.community(
            community_frames, frame_batch_size=frame_batch_size
        )
        weight = self.config.clip_weight
        return cast(Tensor, weight * clip_score + (1 - weight) * community_score)
