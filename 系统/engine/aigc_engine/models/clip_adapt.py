"""Local CLIP last-block adaptation with reusable frozen prefix tokens."""

from dataclasses import dataclass
from math import isqrt
from typing import Self, cast

import torch
from torch import Tensor, nn

from .core import FeatureHead, HeadConfig
from .ufd import UFDConfig, UFDVideoClassifier


@dataclass(frozen=True)
class ClipAdaptConfig(UFDConfig):
    hidden_dim: int = 64
    dropout: float = 0.1
    train_last_block: bool = True
    temporal_adapter_dim: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.train_last_block, bool):
            raise TypeError("train_last_block must be a bool")
        if type(self.temporal_adapter_dim) is not int or self.temporal_adapter_dim < 0:
            raise ValueError("temporal_adapter_dim must be a nonnegative integer")


class _PatchTemporalAdapter(nn.Module):
    """Identity-initialized temporal residual; CLS tokens bypass the adapter."""

    def __init__(self, hidden_size: int, adapter_dim: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim)
        self.conv = nn.Conv3d(
            adapter_dim,
            adapter_dim,
            (3, 1, 1),
            padding=(1, 0, 0),
            groups=adapter_dim,
        )
        self.up = nn.Linear(adapter_dim, hidden_size)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(cast(Tensor, self.conv.bias))
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: Tensor) -> Tensor:
        patches = tokens[:, :, 1:, :]
        batch, frames, count, _ = patches.shape
        side = isqrt(count)
        hidden = self.down(patches).reshape(batch, frames, side, side, -1)
        hidden = self.conv(hidden.permute(0, 4, 1, 2, 3))
        hidden = hidden.permute(0, 2, 3, 4, 1).reshape(batch, frames, count, -1)
        patches = patches + self.up(hidden)
        return torch.cat((tokens[:, :, :1, :], patches), dim=2)


class ClipAdaptVideoClassifier(UFDVideoClassifier):
    """Train the final visual block and MLP; keep the prefix/projection frozen.

    Raw input is [batch, frames, 3, height, width]. Cached prefix input is
    [batch, frames, tokens, hidden_size], preserving every CLS/patch token.
    Projected features remain unnormalized, as in the UFD inference path.
    """

    config: ClipAdaptConfig

    def __init__(self, config: ClipAdaptConfig) -> None:
        super().__init__(config)
        self.head = FeatureHead(
            HeadConfig(
                self.backbone.config.projection_dim, config.hidden_dim, config.dropout
            )
        )
        self.temporal_adapter: _PatchTemporalAdapter | None = (
            _PatchTemporalAdapter(
                self.backbone.config.hidden_size, config.temporal_adapter_dim
            )
            if config.temporal_adapter_dim
            else None
        )
        self.backbone.requires_grad_(False)
        if config.train_last_block:
            vision = self.backbone.vision_model
            vision.encoder.layers[-1].requires_grad_(True)
            vision.post_layernorm.requires_grad_(True)
        self.train()

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        cast(nn.Module, self.backbone).eval()
        vision = self.backbone.vision_model
        vision.encoder.layers[-1].train(mode and self.config.train_last_block)
        vision.post_layernorm.train(mode and self.config.train_last_block)
        return self

    @torch.no_grad()
    def encode_prefix(self, images: Tensor) -> Tensor:
        """Encode frames through the frozen prefix; FP32 inputs yield FP32 tokens."""
        vision = self.backbone.vision_model
        hidden = vision.pre_layrnorm(vision.embeddings(images))
        for layer in vision.encoder.layers[:-1]:
            hidden = layer(hidden, attention_mask=None)
        return cast(Tensor, hidden)

    def forward_tokens(
        self, tokens: Tensor, *, frame_batch_size: int | None = None
    ) -> Tensor:
        batch, frames = tokens.shape[:2]
        vision = self.backbone.vision_model
        parameter = next(self.backbone.parameters())
        if self.temporal_adapter is not None:
            tokens = tokens.to(
                device=parameter.device, dtype=parameter.dtype, non_blocking=True
            )
            tokens = self.temporal_adapter(tokens)
        flattened = tokens.flatten(0, 1)
        chunk_size = len(flattened) if frame_batch_size is None else frame_batch_size
        features: list[Tensor] = []
        for chunk in torch.split(flattened, chunk_size):
            hidden = chunk.to(
                device=parameter.device, dtype=parameter.dtype, non_blocking=True
            )
            hidden = vision.encoder.layers[-1](hidden, attention_mask=None)
            pooled = vision.post_layernorm(hidden[:, 0, :])
            features.append(self.backbone.visual_projection(pooled))
        logits = self.head(torch.cat(features)).reshape(batch, frames)
        return cast(Tensor, logits.mean(dim=1))

    def forward(self, images: Tensor, *, frame_batch_size: int | None = None) -> Tensor:
        if images.ndim == 4:
            return self.forward_tokens(images, frame_batch_size=frame_batch_size)
        if self.temporal_adapter is None:
            return super().forward(images, frame_batch_size=frame_batch_size)
        batch, frames = images.shape[:2]
        flattened = images.flatten(0, 1)
        device = next(self.backbone.parameters()).device
        chunk_size = len(flattened) if frame_batch_size is None else frame_batch_size
        chunks = torch.split(flattened, chunk_size)
        with torch.autocast(device.type, enabled=False):
            tokens = torch.cat(
                [
                    self.encode_prefix(chunk.to(device=device, dtype=torch.float32))
                    for chunk in chunks
                ]
            )
        return self.forward_tokens(
            tokens.reshape(batch, frames, *tokens.shape[1:]),
            frame_batch_size=frame_batch_size,
        )
