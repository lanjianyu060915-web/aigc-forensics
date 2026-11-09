from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


READY_WORDS = {"available", "ready", "ok", "healthy", "success", "initialized"}


@dataclass(frozen=True)
class ModuleAvailability:
    key: str
    title: str
    available: bool
    empty: bool
    value: str
    caption: str
    message: str
    raw: dict[str, Any]

    @property
    def runnable(self) -> bool:
        return self.available and not self.empty


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _walk(mapping: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    for key, value in mapping.items():
        yield str(key).casefold(), value
        if isinstance(value, Mapping):
            yield from _walk(value)


def _first(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    wanted = {key.casefold() for key in keys}
    for key, value in _walk(mapping):
        if key in wanted:
            return value
    return None


def _count(mapping: Mapping[str, Any], keys: Iterable[str]) -> int | None:
    value = _first(mapping, keys)
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    return None


def _available(mapping: Mapping[str, Any]) -> bool:
    explicit = mapping.get("available")
    if isinstance(explicit, bool):
        return explicit
    ready = mapping.get("ready")
    if isinstance(ready, bool):
        return ready
    status = str(mapping.get("status") or mapping.get("state") or "").casefold()
    return status in READY_WORDS


def _message(mapping: Mapping[str, Any], fallback: str) -> str:
    value = _first(mapping, ("message", "error", "reason", "detail"))
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def module_availability(snapshot: Mapping[str, Any], key: str) -> ModuleAvailability:
    raw = _mapping(snapshot.get(key))
    available = _available(raw)
    title = {
        "deepfake": "AIGC 模型",
        "identity": "身份数据库",
        "source": "素材数据库",
        "general_aigc": "通用 AIGC 模型",
    }.get(key, key)

    if key == "general_aigc":
        kinds = raw.get("calibrated_kinds")
        calibrated = "、".join(str(item) for item in kinds) if isinstance(kinds, list) and kinds else "无"
        value = "可用" if available else "未启用" if raw.get("status") == "disabled" else "不可用"
        caption = f"{raw.get('model_name') or '—'} · 已标定：{calibrated}"
        return ModuleAvailability(
            key, title, available, False, value, caption, _message(raw, caption), raw
        )

    if key == "deepfake":
        model = _first(raw, ("model_name", "model", "model_version", "architecture"))
        value = "可用" if available else "不可用"
        caption = str(model) if model else _message(raw, "模型状态已检查")
        return ModuleAvailability(
            key, title, available, False, value, caption, _message(raw, caption), raw
        )

    if key == "identity":
        people = _count(
            raw,
            ("identity_count", "person_count", "people_count", "persons", "identities"),
        )
        images = _count(raw, ("face_image_count", "image_count", "images", "file_count"))
        embeddings = _count(
            raw,
            ("embedding_count", "embeddings_count", "vector_count", "vectors"),
        )
        index_state = _first(raw, ("index_status", "index_state"))
        # embedding 为 0 只表示索引还没建，首次检索会自动建立，不算空库。
        known_counts = [count for count in (people, images) if count is not None]
        empty = bool(known_counts) and any(count == 0 for count in known_counts)
        state_text = "可用" if available and not empty else "未初始化" if empty else "不可用"
        value = f"{state_text} · {people} 人" if people is not None else state_text
        parts = []
        if images is not None:
            parts.append(f"{images} 张人脸图片")
        if embeddings is not None:
            parts.append(f"{embeddings} 条 embedding")
        if index_state is not None:
            parts.append(f"index={index_state}")
        caption = " · ".join(parts) or _message(raw, "未返回数量")
        return ModuleAvailability(
            key, title, available, empty, value, caption, _message(raw, caption), raw
        )

    images = _count(raw, ("source_image_count", "image_count", "images", "file_count"))
    features = _count(
        raw,
        ("feature_count", "features_count", "descriptor_count", "vector_count"),
    )
    index_state = _first(raw, ("index_status", "index_state"))
    index_text = str(index_state or "").casefold()
    realtime_index = index_text in {"not_persistent", "realtime", "realtime_scan"}
    empty = images == 0 or (
        images is None and features == 0 and not realtime_index
    )
    state_text = "可用" if available and not empty else "未初始化" if empty else "不可用"
    value = f"{state_text} · {images} 张" if images is not None else state_text
    parts = []
    if features is not None:
        if realtime_index:
            parts.append(f"持久特征 {features} 组（任务内实时提取）")
        else:
            parts.append(f"{features} 组特征")
    if index_state is not None:
        parts.append(f"index={index_state}")
    caption = " · ".join(parts) or _message(raw, "未返回数量")
    return ModuleAvailability(
        key, title, available, empty, value, caption, _message(raw, caption), raw
    )


def device_label(snapshot: Mapping[str, Any]) -> tuple[str, str]:
    raw = snapshot.get("device")
    if isinstance(raw, Mapping):
        value = _first(
            raw,
            (
                "actual_device",
                "selected_device",
                "effective",
                "device",
                "name",
                "type",
            ),
        )
        cuda_available = raw.get("cuda_available")
        deepfake_cuda_available = raw.get("deepfake_cuda_available")
        gpu_names = raw.get("gpu_names")
        detail_parts = []
        if isinstance(cuda_available, bool):
            detail_parts.append(f"硬件 CUDA={'可用' if cuda_available else '不可用'}")
        if isinstance(deepfake_cuda_available, bool):
            detail_parts.append(
                f"AIGC CUDA={'可用' if deepfake_cuda_available else '不可用'}"
            )
        if isinstance(gpu_names, list) and gpu_names:
            detail_parts.append("、".join(str(name) for name in gpu_names))
        deepfake_device = raw.get("deepfake_effective")
        identity_device = raw.get("identity_effective")
        if deepfake_device:
            detail_parts.append(f"AIGC={deepfake_device}")
        if identity_device:
            detail_parts.append(f"身份={identity_device}")
        detail = " · ".join(detail_parts) or _message(
            raw, "GPU 可用时优先，否则 CPU 回退"
        )
    else:
        value = raw
        detail = "GPU 可用时优先，否则 CPU 回退"
    if not value:
        deepfake = _mapping(snapshot.get("deepfake"))
        value = _first(deepfake, ("actual_device", "selected_device", "device"))
    return str(value or "未知").upper(), detail
