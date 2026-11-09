"""Offline Community Forensics ViT-S/16 384 frame detector."""

from dataclasses import dataclass
from typing import Any, cast

from torch import Tensor, cat, nn, split


@dataclass(frozen=True)
class CommunityForensicsConfig:
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        if self.aggregation != "mean":
            raise ValueError(
                "Community Forensics requires mean frame-logit aggregation"
            )


class CommunityForensicsVideoClassifier(nn.Module):
    """Run the complete public detector, then average raw logits over frames.

    Input is [batch, frames, 3, 384, 384]. Frame batching bounds backbone calls
    and device transfers. All parameters, including the head, come from a local
    checkpoint; construction never requests pretrained weights.
    """

    def __init__(self, config: CommunityForensicsConfig) -> None:
        super().__init__()
        import timm  # noqa: PLC0415

        self.config = config
        self.backbone = timm.create_model(
            "vit_small_patch16_384.augreg_in21k_ft_in1k",
            pretrained=False,
            num_classes=1,
        )
        self.preprocessing: dict[str, Any] = {
            "input_size": (3, 384, 384),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "resize_mode": "pil_center_crop",
            "resize_short_edge": 440,
        }

    def forward(self, images: Tensor, *, frame_batch_size: int | None = None) -> Tensor:
        batch, frames = images.shape[:2]
        flattened = images.reshape(batch * frames, *images.shape[2:])
        if frame_batch_size is None:
            outputs = self.backbone(flattened)
        else:
            device = next(self.parameters()).device
            outputs = cat(
                [
                    self.backbone(chunk.to(device))
                    for chunk in split(flattened, frame_batch_size)
                ]
            )
        logits = outputs[:, -1].reshape(batch, frames)
        return cast(Tensor, logits.mean(dim=1))
