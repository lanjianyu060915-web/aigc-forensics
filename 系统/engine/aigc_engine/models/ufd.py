"""Self-contained UniversalFakeDetect CLIP visual encoder and linear head."""

from dataclasses import dataclass, field
from typing import Any, cast

from torch import Tensor, cat, nn, split


def _vit_l14_config() -> dict[str, Any]:
    return {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "image_size": 224,
        "patch_size": 14,
        "projection_dim": 768,
        "hidden_act": "quick_gelu",
        "layer_norm_eps": 1e-5,
        "attention_dropout": 0.0,
    }


@dataclass(frozen=True)
class UFDConfig:
    backbone_config: dict[str, Any] = field(default_factory=_vit_l14_config)
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        if self.aggregation != "mean":
            raise ValueError("UFD requires mean frame-logit aggregation")


class UFDVideoClassifier(nn.Module):
    """Apply the author's unnormalized CLIP projection and linear fake head.

    Input is [batch, frames, 3, height, width]; output is one raw logit per video.
    Construction uses only the embedded configuration and never downloads weights.
    frame_batch_size bounds inference backbone calls and device transfers;
    the head and video mean still use all frame features in their original order.
    """

    def __init__(self, config: UFDConfig) -> None:
        super().__init__()
        from transformers import (  # noqa: PLC0415
            CLIPVisionConfig,
            CLIPVisionModelWithProjection,
        )

        self.config = config
        vision_config = CLIPVisionConfig(**config.backbone_config)
        vision_config._attn_implementation = "eager"
        self.backbone = CLIPVisionModelWithProjection(vision_config)
        self.head: nn.Module = nn.Linear(vision_config.projection_dim, 1)
        self.preprocessing: dict[str, Any] = {
            "input_size": (3, vision_config.image_size, vision_config.image_size),
            "mean": (0.48145466, 0.4578275, 0.40821073),
            "std": (0.26862954, 0.26130258, 0.27577711),
            "resize_mode": "native_center_crop",
        }

    def forward(self, images: Tensor, *, frame_batch_size: int | None = None) -> Tensor:
        batch, frames = images.shape[:2]
        flattened = images.reshape(batch * frames, *images.shape[2:])
        if frame_batch_size is None:
            features = self.backbone(pixel_values=flattened).image_embeds
        else:
            device = next(self.parameters()).device
            features = cat(
                [
                    self.backbone(pixel_values=chunk.to(device)).image_embeds
                    for chunk in split(flattened, frame_batch_size)
                ]
            )
        logits = self.head(features).reshape(batch, frames)
        return cast(Tensor, logits.mean(dim=1))
