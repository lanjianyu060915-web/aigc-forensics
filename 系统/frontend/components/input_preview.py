"""分析前的输入预检：只读文件头，告诉使用者这张图会走哪条鉴伪分支、有哪些质量风险。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
LOW_RESOLUTION_EDGE = 112
DEFAULT_ROUTING_EDGE = 256


@dataclass(frozen=True)
class ExampleQuery:
    """内置示例，接口与 Streamlit 上传对象一致。"""

    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def size(self) -> int:
        return self.path.stat().st_size

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


@dataclass
class InputPreview:
    kind: str
    file_name: str
    size_bytes: int
    sha256: str
    width: int | None = None
    height: int | None = None
    route_label: str | None = None
    route_reason: str | None = None
    notes: list[str] = field(default_factory=list)
    readable: bool = True

    @property
    def size_label(self) -> str:
        if self.size_bytes >= 1024 * 1024:
            return f"{self.size_bytes / 1048576:.1f} MB"
        return f"{self.size_bytes / 1024:.0f} KB"


def describe_input(
    name: str, data: bytes, *, routing_edge: int = DEFAULT_ROUTING_EDGE
) -> InputPreview:
    suffix = Path(name).suffix.lower()
    digest = hashlib.sha256(data).hexdigest()
    if suffix in VIDEO_SUFFIXES:
        return InputPreview(
            kind="video",
            file_name=name,
            size_bytes=len(data),
            sha256=digest,
            route_label="抽帧后逐帧路由",
            route_reason="先选关键帧，再按每帧的有效短边选择鉴伪分支；素材检索只用首要帧。",
            notes=["视频级分带跨数据来源不可移植，结果只作线索。"],
        )

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError):
        return InputPreview(
            kind="image",
            file_name=name,
            size_bytes=len(data),
            sha256=digest,
            readable=False,
            notes=["无法读取这张图片，文件可能已损坏或不是真正的图片格式。"],
        )
    short_edge = min(width, height)
    notes: list[str] = []
    if short_edge >= routing_edge:
        route_label = "二代整图分支（ViT）"
        route_reason = f"有效短边 {short_edge} px，不低于路由阈值 {routing_edge} px。"
    else:
        route_label = "人脸裁剪分支（空间+频域）"
        route_reason = f"有效短边 {short_edge} px，低于路由阈值 {routing_edge} px。"
    if short_edge < LOW_RESOLUTION_EDGE:
        notes.append("分辨率很低，人脸检测和素材局部复核都可能失败。")
    if max(width, height) / max(short_edge, 1) > 3:
        notes.append("长宽比异常，可能是长截图或拼接图，建议先裁出人物区域。")
    return InputPreview(
        kind="image",
        file_name=name,
        size_bytes=len(data),
        sha256=digest,
        width=width,
        height=height,
        route_label=route_label,
        route_reason=route_reason,
        notes=notes,
    )


def routing_edge_from_config(config: Any) -> int:
    try:
        value = config["deepfake"]["detector_v2"]["routing_short_edge_threshold_px"]
        return int(value)
    except (KeyError, TypeError, ValueError):
        return DEFAULT_ROUTING_EDGE


__all__ = [
    "ExampleQuery",
    "InputPreview",
    "describe_input",
    "routing_edge_from_config",
]
