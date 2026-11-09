"""集中读取软件全称与简称，界面和新报告共用同一配置。"""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PRODUCT_NAME = "面向涉案图像视频的 AIGC 鉴伪与来源线索研判系统"
DEFAULT_PRODUCT_SHORT_NAME = "AIGC 鉴伪与来源线索研判"
RELEASE_NAME = "第四版"
RELEASE_VERSION = "0.4.0-v4-engineering"
RELEASE_SCOPE = "工程改进；检测策略默认保持原版，效果需独立验收"


def product_names(config: Mapping[str, Any] | None = None) -> tuple[str, str]:
    """读取当前配置；旧配置缺少名称时使用统一的默认值。"""
    if config is None:
        path = Path(os.environ.get("FSTS_CONFIG_PATH") or Path(__file__).with_name("config.json"))
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            config = value if isinstance(value, Mapping) else {}
        except (OSError, ValueError):
            config = {}
    def name(key: str, default: str) -> str:
        value = config.get(key)
        return value.strip() if isinstance(value, str) and value.strip() else default
    return name("product_name", DEFAULT_PRODUCT_NAME), name("product_short_name", DEFAULT_PRODUCT_SHORT_NAME)


PRODUCT_NAME, PRODUCT_SHORT_NAME = product_names()
