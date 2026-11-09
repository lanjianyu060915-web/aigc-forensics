"""Select local media executables consistently for API calls and launchers."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path


_BUNDLE_BIN = Path(__file__).resolve().parents[2] / "ffmpeg"
_SUFFIX = ".exe" if os.name == "nt" else ""


@cache
def _selected_bin(explicit: str | None, bundle: Path) -> Path | None:
    """Validate each configured immutable tool directory once per process."""
    if explicit is not None:
        selected = Path(explicit)
        origin = "AIGC_MEDIA_TOOLS"
    elif bundle.is_dir():
        selected = bundle
        origin = "bundled media tools"
    else:
        return None
    for name in ("ffmpeg", "ffprobe"):
        binary = selected / f"{name}{_SUFFIX}"
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(f"{origin} requires executable {binary}")
    return selected


def _configured_bin() -> Path | None:
    explicit = os.environ.get("AIGC_MEDIA_TOOLS")
    if explicit:
        explicit = str(Path(explicit).expanduser().absolute())
    return _selected_bin(explicit or None, _BUNDLE_BIN)


def media_tool(name: str) -> str:
    """Keep host command lookup unchanged unless local tools are selected."""
    if name not in ("ffmpeg", "ffprobe"):
        return name
    selected = _configured_bin()
    return str(selected / f"{name}{_SUFFIX}") if selected is not None else name
