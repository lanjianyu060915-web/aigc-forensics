"""Deterministic, bounded local media decoding with explicit failures.

Video sampling uses decoded presentation timestamps, including VFR sources.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .media_tools import media_tool


class MediaError(ValueError):
    """A sample-level failure whose stable code can be recorded in a manifest."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


# Crop centers as fractions of width/height; the first is the standard center crop.
NATIVE_CROP_CENTERS = (
    (0.5, 0.5),
    (0.25, 0.25),
    (0.75, 0.25),
    (0.25, 0.75),
    (0.75, 0.75),
)


@dataclass(frozen=True)
class VideoConfig:
    n_frames: int = 16
    height: int = 224
    width: int = 224
    resize_mode: Literal[
        "pad", "center_crop", "native_center_crop", "pil_center_crop"
    ] = "pad"
    stream_index: int | None = None
    max_duration_seconds: float = 600.0
    timeout_seconds: float = 60.0
    max_source_pixels: int = 40_000_000
    max_source_frames: int = 100_000
    max_output_bytes: int = 512 * 1024 * 1024
    resize_short_edge: int | None = None
    native_crops: int = 1
    tolerant: bool = False

    def __post_init__(self) -> None:
        if self.stream_index is not None and (
            type(self.stream_index) is not int or self.stream_index < 0
        ):
            raise ValueError("stream_index must be a nonnegative integer or null")
        for name in (
            "n_frames",
            "height",
            "width",
            "max_source_pixels",
            "max_source_frames",
            "max_output_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _positive(self.max_duration_seconds, "max_duration_seconds")
        _positive(self.timeout_seconds, "timeout_seconds")
        if self.resize_mode not in (
            "pad",
            "center_crop",
            "native_center_crop",
            "pil_center_crop",
        ):
            raise ValueError("unsupported video resize_mode")
        if self.resize_mode == "pil_center_crop":
            if type(self.resize_short_edge) is not int or self.resize_short_edge < max(
                self.height, self.width
            ):
                raise ValueError("pil_center_crop requires a short edge >= crop size")
        elif self.resize_short_edge is not None:
            raise ValueError("resize_short_edge requires pil_center_crop")
        if self.native_crops not in (1, len(NATIVE_CROP_CENTERS)):
            raise ValueError(f"native_crops must be 1 or {len(NATIVE_CROP_CENTERS)}")
        if self.native_crops != 1 and self.resize_mode != "native_center_crop":
            raise ValueError("native_crops requires native_center_crop")
        if self.n_frames > 1024:
            raise ValueError("n_frames cannot exceed 1024")
        if (
            self.n_frames * self.native_crops * self.height * self.width * 3
            > self.max_output_bytes
        ):
            raise ValueError("requested video tensor exceeds max_output_bytes")


@dataclass(frozen=True)
class VideoData:
    frames: NDArray[np.uint8]
    timestamps_seconds: tuple[float, ...]
    frame_indices: tuple[int, ...]
    requested_timestamps_seconds: tuple[float, ...]
    original_width: int
    original_height: int
    original_duration_seconds: float
    display_width: int | None = None
    display_height: int | None = None


def _run_output(
    command: list[str], timeout: float, max_bytes: int
) -> tuple[bytes, str]:
    """Capture one bounded invocation; classify diagnostics after stream choice."""
    command = [media_tool(command[0]), *command[1:]]
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen(  # noqa: S603 - fixed executable, shell=False
                command, stdin=subprocess.DEVNULL, stdout=output, stderr=errors
            )
        except FileNotFoundError as exc:
            raise MediaError("missing_dependency", f"{command[0]} unavailable") from exc
        except OSError as exc:
            raise MediaError("process_error", str(exc)) from exc
        deadline = time.monotonic() + timeout
        try:
            while True:
                if os.fstat(output.fileno()).st_size > max_bytes:
                    raise MediaError("output_limit", "decoded output exceeds limit")
                if os.fstat(errors.fileno()).st_size > 1_048_576:
                    raise MediaError("decode_error", "decoder emitted excessive errors")
                if process.poll() is not None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MediaError("timeout", "media subprocess exceeded timeout")
                time.sleep(min(0.02, remaining))
            errors.seek(0)
            diagnostic_bytes = errors.read(1_048_577)
            if len(diagnostic_bytes) > 1_048_576:
                raise MediaError("decode_error", "decoder emitted excessive errors")
            diagnostic = diagnostic_bytes.decode("utf-8", errors="replace").strip()
            if process.returncode != 0:
                raise MediaError("decode_error", diagnostic[:4096] or "decoder failed")
            output.seek(0)
            result = output.read(max_bytes + 1)
            if len(result) > max_bytes:
                raise MediaError("output_limit", "decoded output exceeds limit")
            return result, diagnostic
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def _check_diagnostics(diagnostic: str, *, video_only: bool = False) -> None:
    if video_only:
        # FFmpeg probes audio before -an/-select_streams takes effect. This exact
        # MP2 audio-header diagnostic cannot describe the selected video stream.
        diagnostic = "\n".join(
            line
            for line in diagnostic.splitlines()
            if not re.fullmatch(r"\[mp2 @ 0x[0-9a-fA-F]+\] Header missing", line)
        )
    if diagnostic:
        raise MediaError("decode_error", diagnostic[:4096])


def _run(
    command: list[str],
    timeout: float,
    max_bytes: int,
    *,
    video_only: bool = False,
    tolerant: bool = False,
) -> bytes:
    raw, diagnostic = _run_output(command, timeout, max_bytes)
    if not tolerant:
        _check_diagnostics(diagnostic, video_only=video_only)
    return raw


def _local_file(path: str | Path) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise MediaError("missing_file", f"media file does not exist: {path}")
    return str(resolved)


def _probe(
    path: str,
    stream: str,
    timeout: float,
    *,
    video_only: bool = False,
    tolerant: bool = False,
) -> dict[str, Any]:
    raw = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-select_streams",
            stream,
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            path,
        ],
        timeout,
        1_048_576,
        video_only=video_only,
        tolerant=tolerant,
    )
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or not data.get("streams"):
            raise MediaError("missing_stream", f"no {stream} stream")
        return dict(data)
    except (ValueError, TypeError) as exc:
        if isinstance(exc, MediaError):
            raise
        raise MediaError("invalid_metadata", "invalid probe metadata") from exc


def _duration(probe: dict[str, Any], maximum: float) -> float:
    try:
        duration = float(
            probe["streams"][0].get("duration", probe.get("format", {}).get("duration"))
        )
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise MediaError("invalid_metadata", "missing media duration") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise MediaError("invalid_metadata", "invalid media duration")
    if duration > maximum:
        raise MediaError("duration_limit", "media duration exceeds configured limit")
    return duration


def _tolerant_timestamps(frames: list[dict[str, Any]]) -> NDArray[np.float64]:
    """Fill missing or backward times so a damaged file can still be sampled."""
    values: list[float] = []
    for frame in frames:
        try:
            value = float(frame["best_effort_timestamp_time"])
        except (KeyError, ValueError, TypeError):
            value = math.nan
        if not math.isfinite(value):
            value = values[-1] if values else 0.0
        values.append(max(value, values[-1]) if values else value)
    return np.asarray(values, dtype=np.float64)


def _frame_timestamps(source: str, config: VideoConfig) -> NDArray[np.float64]:
    raw = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-threads",
            "1",
            "-protocol_whitelist",
            "file,pipe",
            "-select_streams",
            str(config.stream_index) if config.stream_index is not None else "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            source,
        ],
        config.timeout_seconds,
        config.max_source_frames * 256,
        video_only=True,
        tolerant=config.tolerant,
    )
    try:
        frames = json.loads(raw)["frames"]
        if not frames:
            raise MediaError("no_frames", "video has no decoded frames")
        if len(frames) > config.max_source_frames:
            raise MediaError("frame_limit", "source frame count exceeds limit")
        if config.tolerant:
            return _tolerant_timestamps(frames)
        timestamps = np.asarray(
            [float(frame["best_effort_timestamp_time"]) for frame in frames],
            dtype=np.float64,
        )
    except (KeyError, ValueError, TypeError) as exc:
        if isinstance(exc, MediaError):
            raise
        raise MediaError(
            "invalid_timestamps", "frame presentation time missing"
        ) from exc
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise MediaError("invalid_timestamps", "frame timestamps invalid or unsorted")
    if timestamps[-1] - timestamps[0] > config.max_duration_seconds:
        raise MediaError("duration_limit", "decoded video timeline exceeds limit")
    return timestamps


def _display_dimensions(
    stream: dict[str, Any], *, allow_arbitrary_rotation: bool = False
) -> tuple[int, int]:
    width, height = int(stream["width"]), int(stream["height"])
    rotation = float(stream.get("tags", {}).get("rotate", 0))
    for item in stream.get("side_data_list", []):
        if "rotation" in item:
            rotation = float(item["rotation"])
    angle = rotation % 180
    if math.isclose(angle, 90, abs_tol=1e-3):
        return height, width
    if not allow_arbitrary_rotation and not math.isclose(angle, 0, abs_tol=1e-3):
        raise MediaError("unsupported_rotation", "PIL input needs right-angle rotation")
    # FFmpeg's non-right-angle rotate filter keeps the source canvas dimensions.
    return width, height


def _pil_resize_crop(
    frames: NDArray[np.uint8], config: VideoConfig
) -> NDArray[np.uint8]:
    from PIL import Image  # noqa: PLC0415

    short_edge = cast(int, config.resize_short_edge)
    height, width = frames.shape[1:3]
    resized = (
        (short_edge, int(short_edge * height / width))
        if width <= height
        else (int(short_edge * width / height), short_edge)
    )
    if len(frames) * resized[0] * resized[1] * 3 > config.max_output_bytes:
        raise MediaError("output_limit", "PIL resized frames exceed output limit")
    left = round((resized[0] - config.width) / 2)
    top = round((resized[1] - config.height) / 2)
    return np.stack(
        [
            np.asarray(
                Image.fromarray(frame)
                .resize(resized, resample=Image.Resampling.BILINEAR)
                .crop((left, top, left + config.width, top + config.height))
            )
            for frame in frames
        ]
    )


def _crop_offset(total: int, size: int, center: float) -> int:
    if center == 0.5:
        # The same tie rounding as the FFmpeg native_center_crop expression.
        base = (total - size) // 2
        return base + (base % 2) * ((total - size) % 2)
    return int(np.clip(round(total * center - size / 2), 0, total - size))


def _native_crops(frames: NDArray[np.uint8], config: VideoConfig) -> NDArray[np.uint8]:
    """Return [frames * crops, height, width, 3], grouped by frame in center order."""
    _, height, width, _ = frames.shape
    crops = []
    for fraction_x, fraction_y in NATIVE_CROP_CENTERS:
        left = _crop_offset(width, config.width, fraction_x)
        top = _crop_offset(height, config.height, fraction_y)
        crops.append(frames[:, top : top + config.height, left : left + config.width])
    return np.stack(crops, axis=1).reshape(-1, config.height, config.width, 3)


def decode_video(path: str | Path, config: VideoConfig | None = None) -> VideoData:
    config = config or VideoConfig()
    source = _local_file(path)
    metadata = _probe(
        source,
        str(config.stream_index) if config.stream_index is not None else "v:0",
        config.timeout_seconds,
        video_only=True,
        tolerant=config.tolerant,
    )
    duration = _duration(metadata, config.max_duration_seconds)
    try:
        width = int(metadata["streams"][0]["width"])
        height = int(metadata["streams"][0]["height"])
    except (ValueError, TypeError, KeyError) as exc:
        raise MediaError("invalid_metadata", "source dimensions missing") from exc
    if width <= 0 or height <= 0 or width * height > config.max_source_pixels:
        raise MediaError("dimension_limit", "source dimensions exceed limits")
    display_width, display_height = _display_dimensions(
        metadata["streams"][0],
        allow_arbitrary_rotation=config.resize_mode != "pil_center_crop",
    )
    timestamps = _frame_timestamps(source, config)
    targets = np.linspace(timestamps[0], timestamps[-1], config.n_frames)
    right = np.searchsorted(timestamps, targets).clip(0, len(timestamps) - 1)
    left = (right - 1).clip(0, len(timestamps) - 1)
    indices = np.where(
        targets - timestamps[left] <= timestamps[right] - targets, left, right
    )
    unique, inverse = np.unique(indices, return_inverse=True)
    selection = "+".join(f"eq(n\\,{int(index)})" for index in unique)
    mode = "decrease" if config.resize_mode == "pad" else "increase"
    scale = (
        f"scale={config.width}:{config.height}:force_original_aspect_ratio={mode}"
        ":flags=bicubic"
    )
    finish = (
        f"pad={config.width}:{config.height}:(ow-iw)/2:(oh-ih)/2:black"
        if config.resize_mode == "pad"
        else f"crop={config.width}:{config.height}"
    )
    spatial_filter = f"{scale},{finish}"
    decoded_width, decoded_height = config.width, config.height
    if config.resize_mode == "pil_center_crop":
        decoded_width, decoded_height = display_width, display_height
        spatial_filter = "format=rgb24"
    if config.resize_mode == "native_center_crop":
        # Work in displayed dimensions after FFmpeg autorotation. PIL CenterCrop
        # pads small images, then rounds odd center offsets with ties to even.
        # RGB avoids chroma-subsampling rounding of crop/pad offsets.
        crop_x = (
            f"floor((iw-{config.width})/2)"
            f"+mod(floor((iw-{config.width})/2),2)*mod(iw-{config.width},2)"
        )
        crop_y = (
            f"floor((ih-{config.height})/2)"
            f"+mod(floor((ih-{config.height})/2),2)*mod(ih-{config.height},2)"
        )
        spatial_filter = (
            f"format=rgb24,pad='max(iw,{config.width})':'max(ih,{config.height})':"
            "'floor((ow-iw)/2)':'floor((oh-ih)/2)':black"
        )
        if config.native_crops == 1:
            spatial_filter += (
                f",crop={config.width}:{config.height}:'{crop_x}':'{crop_y}'"
            )
        else:
            decoded_width = max(display_width, config.width)
            decoded_height = max(display_height, config.height)
    expected = len(unique) * decoded_height * decoded_width * 3
    if expected > config.max_output_bytes:
        raise MediaError("output_limit", "selected RGB frames exceed output limit")
    # Strict decoding fails on any diagnostic; tolerant keeps whatever still decodes.
    strict_flags = [] if config.tolerant else ["-xerror", "-err_detect", "explode"]
    raw = _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            *strict_flags[:1],
            "-threads",
            "1",
            "-protocol_whitelist",
            "file,pipe",
            *strict_flags[1:],
            "-i",
            source,
            "-map",
            f"0:{config.stream_index}" if config.stream_index is not None else "0:v:0",
            "-an",
            "-filter_threads",
            "1",
            "-vf",
            f"select={selection},{spatial_filter},setsar=1",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        config.timeout_seconds,
        config.max_output_bytes,
        video_only=True,
        tolerant=config.tolerant,
    )
    frame_bytes = decoded_height * decoded_width * 3
    decoded_count = len(unique)
    if len(raw) != expected:
        decoded_count = len(raw) // frame_bytes
        if not config.tolerant or decoded_count == 0:
            raise MediaError("frame_count_mismatch", "not all selected frames decoded")
        raw = raw[: decoded_count * frame_bytes]
        inverse = np.minimum(inverse, decoded_count - 1)
    frames = (
        np.frombuffer(raw, dtype=np.uint8)
        .reshape(decoded_count, decoded_height, decoded_width, 3)
        .copy()
    )
    if config.resize_mode == "pil_center_crop":
        frames = _pil_resize_crop(frames, config)
    frames = frames[inverse]
    if config.native_crops != 1:
        frames = _native_crops(frames, config)
    return VideoData(
        frames,
        tuple(float(timestamps[i]) for i in indices),
        tuple(int(i) for i in indices),
        tuple(float(t) for t in targets),
        width,
        height,
        duration,
        display_width,
        display_height,
    )

