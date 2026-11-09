"""MLP head shared by the CLIP video heads."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class HeadConfig:
    input_dim: int
    hidden_dim: int = 64
    dropout: float = 0.1

    def __post_init__(self) -> None:
        _positive_integer(self.input_dim, "input_dim")
        _positive_integer(self.hidden_dim, "hidden_dim")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")


class FeatureHead(nn.Module):
    """MLP shared by cached audio embeddings and individual video frames."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim < 2 or features.shape[-1] != self.config.input_dim:
            raise ValueError("features must have shape [..., input_dim]")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features must be finite")
        result: Tensor = self.layers(features).squeeze(-1)
        return result
