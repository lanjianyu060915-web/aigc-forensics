"""一次加载模型，对单张图片或单条视频打分，并保留各分支的明细。

分数公式与 video_clip_community_fusion_v1 的 forward 一致：
0.8 × [0.5 × CLIP 原头 + 0.5 × 重训头均值] + 0.2 × CommunityForensics，都是原始 logit，高分更倾向生成。
图片按“一帧视频”处理：CLIP 分支取 5 个原尺寸裁剪，CommunityForensics 取一帧。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .media import (
    MediaError,
    VideoConfig,
    _check_diagnostics,
    _native_crops,
    _pil_resize_crop,
    _run_output,
    decode_video,
)
from .models import ClipCommunityFusionClassifier, load_checkpoint


IMAGE_MAX_PIXELS = 40_000_000
VIDEO_TIMEOUT_SECONDS = 180.0
DEFAULT_FRAME_BATCH_SIZE = 16


@dataclass(frozen=True)
class ScoreResult:
    kind: str
    logit: float
    clip_logit: float
    clip_original_logit: float
    clip_refit_logit: float | None
    community_logit: float
    frame_logits: tuple[tuple[float, ...], ...]
    frame_timestamps: tuple[float, ...]
    community_frame_logits: tuple[float, ...]
    community_timestamps: tuple[float, ...]
    width: int
    height: int
    duration_seconds: float | None
    decode_mode: str
    decode_seconds: float
    inference_seconds: float
    frame_batch_size: int
    warnings: tuple[str, ...] = ()


def resolve_device(requested: str) -> torch.device:
    """auto 优先用显卡；显式要 cuda 而不可用时报错，不静默退回 CPU。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境不可用")
    return device


def normalize_rgb(rgb: NDArray[np.uint8], preprocessing: dict[str, Any]) -> Tensor:
    """Normalize already decoded RGB frames, preserving their order and bytes."""
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("decoded RGB must be uint8 [frames, height, width, 3]")
    frames = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255)
    mean = frames.new_tensor(preprocessing["mean"])[None, :, None, None]
    std = frames.new_tensor(preprocessing["std"])[None, :, None, None]
    result: Tensor = (frames - mean) / std
    if not bool(torch.isfinite(result).all()):
        raise ValueError("decoded frames must be finite [frames, 3, height, width]")
    return result.contiguous()


def select_video_stream(path: Path, timeout: float = 30.0) -> int:
    """取第一条非封面视频流；没有就报错，不把音频或封面当视频。"""
    raw, diagnostic = _run_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        timeout,
        1_048_576,
    )
    streams = sorted(
        cast(list[dict[str, Any]], json.loads(raw).get("streams", [])),
        key=lambda stream: int(stream["index"]),
    )
    for stream in streams:
        if (
            stream.get("codec_type") == "video"
            and not stream.get("disposition", {}).get("attached_pic", 0)
            and int(stream.get("width", 0)) > 0
            and int(stream.get("height", 0)) > 0
        ):
            return int(stream["index"])
    _check_diagnostics(diagnostic)
    raise MediaError("missing_stream", "文件里没有可用的视频流")


def load_image_rgb(path: Path) -> tuple[NDArray[np.uint8], list[str]]:
    """读成 [高, 宽, 3]，按 EXIF 方向摆正；超大图等比缩小并记录。"""
    from PIL import Image, ImageOps  # noqa: PLC0415

    warnings: list[str] = []
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle)
        if getattr(handle, "n_frames", 1) > 1:
            warnings.append("文件含多帧，只检测第一帧。")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise MediaError("invalid_metadata", "图片尺寸无效")
        if width * height > IMAGE_MAX_PIXELS:
            ratio = math.sqrt(IMAGE_MAX_PIXELS / (width * height))
            size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
            image = image.resize(size, resample=Image.Resampling.BILINEAR)
            warnings.append(f"图片过大，从 {width}×{height} 缩小到 {size[0]}×{size[1]} 后检测。")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return rgb, warnings


def pad_to_minimum(rgb: NDArray[np.uint8], height: int, width: int) -> NDArray[np.uint8]:
    """小于裁剪尺寸时居中补黑，与视频路径的 FFmpeg pad 表达式一致。"""
    source_height, source_width = rgb.shape[:2]
    target_height, target_width = max(source_height, height), max(source_width, width)
    if (target_height, target_width) == (source_height, source_width):
        return rgb
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    top = (target_height - source_height) // 2
    left = (target_width - source_width) // 2
    canvas[top : top + source_height, left : left + source_width] = rgb
    return canvas


class GeneralAigcScorer:
    """Hold one verified model; media errors surface to the caller per input."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        sha256: str,
        device: str = "auto",
        frame_batch_size: int = DEFAULT_FRAME_BATCH_SIZE,
    ) -> None:
        self.device = resolve_device(device)
        started = time.perf_counter()
        model, self.metadata = load_checkpoint(Path(checkpoint), expected_sha256=sha256)
        if not isinstance(model, ClipCommunityFusionClassifier):
            raise ValueError("通用 AIGC 引擎需要 video_clip_community_fusion_v1 权重")
        self.model = model.to(self.device).eval()
        self.frame_batch_size = frame_batch_size
        self.load_seconds = time.perf_counter() - started
        config = self.model.config
        clip_input = self.model.clip.preprocessing["input_size"]
        community = self.model.community.preprocessing
        self.clip_view = VideoConfig(
            n_frames=config.clip_frames,
            height=int(clip_input[1]),
            width=int(clip_input[2]),
            resize_mode="native_center_crop",
            native_crops=config.clip_crops,
            timeout_seconds=VIDEO_TIMEOUT_SECONDS,
        )
        self.community_view = replace(
            self.clip_view,
            n_frames=config.community_frames,
            height=int(community["input_size"][1]),
            width=int(community["input_size"][2]),
            resize_mode=community["resize_mode"],
            resize_short_edge=community["resize_short_edge"],
            native_crops=1,
        )

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            return str(torch.cuda.get_device_name(self.device))
        return "CPU"

    def score_image(self, path: str | Path) -> ScoreResult:
        started = time.perf_counter()
        rgb, warnings = load_image_rgb(Path(path))
        height, width = rgb.shape[:2]
        crops = max(1, self.clip_view.native_crops)
        single = replace(self.clip_view, n_frames=1)
        padded = pad_to_minimum(rgb, single.height, single.width)
        clip_rgb = _native_crops(padded[None, ...], single)
        community_rgb = _pil_resize_crop(rgb[None, ...], replace(self.community_view, n_frames=1))
        decode_seconds = time.perf_counter() - started
        if min(height, width) < single.height:
            warnings.append(f"图片短边 {min(height, width)}px 小于 {single.height}px，已补黑边后检测，结果可靠性下降。")
        return self._score(
            kind="image",
            clip_rgb=clip_rgb,
            community_rgb=community_rgb,
            crops=crops,
            frame_timestamps=(0.0,),
            community_timestamps=(0.0,),
            width=width,
            height=height,
            duration=None,
            decode_mode="image",
            decode_seconds=decode_seconds,
            warnings=warnings,
        )

    def score_video(self, path: str | Path) -> ScoreResult:
        started = time.perf_counter()
        source = Path(path)
        stream = select_video_stream(source)
        warnings: list[str] = []
        try:
            clip_video = decode_video(source, replace(self.clip_view, stream_index=stream))
            community_video = decode_video(
                source, replace(self.community_view, stream_index=stream)
            )
            decode_mode = "strict"
        except MediaError as exc:
            # 严格模式只要有一处解码告警就失败；实战文件常有小损坏，改用容错解码并如实记录。
            if exc.code not in {"decode_error", "invalid_timestamps", "frame_count_mismatch"}:
                raise
            clip_video = decode_video(
                source, replace(self.clip_view, stream_index=stream, tolerant=True)
            )
            community_video = decode_video(
                source, replace(self.community_view, stream_index=stream, tolerant=True)
            )
            decode_mode = "tolerant"
            warnings.append(f"文件存在解码问题，已改用容错解码（{exc.code}），结果需结合原文件复核。")
        decode_seconds = time.perf_counter() - started
        return self._score(
            kind="video",
            clip_rgb=clip_video.frames,
            community_rgb=community_video.frames,
            crops=self.clip_view.native_crops,
            frame_timestamps=clip_video.timestamps_seconds,
            community_timestamps=community_video.timestamps_seconds,
            width=clip_video.display_width or clip_video.original_width,
            height=clip_video.display_height or clip_video.original_height,
            duration=clip_video.original_duration_seconds,
            decode_mode=decode_mode,
            decode_seconds=decode_seconds,
            warnings=warnings,
        )

    def _score(
        self,
        *,
        kind: str,
        clip_rgb: NDArray[np.uint8],
        community_rgb: NDArray[np.uint8],
        crops: int,
        frame_timestamps: tuple[float, ...],
        community_timestamps: tuple[float, ...],
        width: int,
        height: int,
        duration: float | None,
        decode_mode: str,
        decode_seconds: float,
        warnings: list[str],
    ) -> ScoreResult:
        clip_frames = normalize_rgb(clip_rgb, self.model.clip.preprocessing)
        community_frames = normalize_rgb(community_rgb, self.model.community.preprocessing)
        started = time.perf_counter()
        batch = self.frame_batch_size
        while True:
            try:
                details = self._forward(clip_frames, community_frames, batch)
                break
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda" or batch <= 1:
                    raise
                torch.cuda.empty_cache()
                batch = max(1, batch // 2)
        per_crop, community_logits, original, refit = details
        config = self.model.config
        clip_logit = float(np.mean(per_crop))
        community_logit = float(np.mean(community_logits))
        logit = config.clip_weight * clip_logit + (1 - config.clip_weight) * community_logit
        frame_logits = per_crop.reshape(-1, crops)
        return ScoreResult(
            kind=kind,
            logit=float(logit),
            clip_logit=clip_logit,
            clip_original_logit=original,
            clip_refit_logit=refit,
            community_logit=community_logit,
            frame_logits=tuple(tuple(float(v) for v in row) for row in frame_logits),
            frame_timestamps=tuple(float(v) for v in frame_timestamps),
            community_frame_logits=tuple(float(v) for v in community_logits),
            community_timestamps=tuple(float(v) for v in community_timestamps),
            width=int(width),
            height=int(height),
            duration_seconds=float(duration) if duration is not None else None,
            decode_mode=decode_mode,
            decode_seconds=decode_seconds,
            inference_seconds=time.perf_counter() - started,
            frame_batch_size=batch,
            warnings=tuple(warnings),
        )

    @torch.inference_mode()
    def _forward(
        self, clip_frames: Tensor, community_frames: Tensor, batch: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float | None]:
        """拆开模型 forward，多保留逐裁剪和逐帧分数；数值与 forward 相同。"""
        model = self.model
        features = torch.cat(
            [
                model.clip.backbone(pixel_values=chunk.to(self.device)).image_embeds
                for chunk in torch.split(clip_frames, batch)
            ]
        )
        original = model.clip.head(features)
        if model.config.refit_heads:
            refit = torch.stack([head(features) for head in model.refit_heads]).mean(dim=0)
            weight = model.config.refit_weight
            combined = (1 - weight) * original + weight * refit
            refit_logit: float | None = float(refit.mean().float().cpu())
        else:
            combined = original
            refit_logit = None
        community = torch.cat(
            [
                model.community.backbone(chunk.to(self.device))[:, -1]
                for chunk in torch.split(community_frames, batch)
            ]
        )
        return (
            combined.float().cpu().numpy().astype(np.float64),
            community.float().cpu().numpy().astype(np.float64),
            float(original.mean().float().cpu()),
            refit_logit,
        )


def logit_to_score(logit: float) -> float:
    """把 logit 压到 0–1 便于展示；这不是校准过的概率。"""
    return 1.0 / (1.0 + math.exp(-max(min(logit, 50.0), -50.0)))
