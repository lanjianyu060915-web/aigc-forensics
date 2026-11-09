"""在主脸结果完成后追加逐脸候选，不修改主脸检索结果。"""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any

from decision.open_set import decide_identity_open_set


# 六脸的默认 CPU 附加阶段超过三十秒，按性能约束将上限降为三张。
MAX_FACES = 3
FACE_LIMITATION = "逐脸候选沿用现有开放集阈值，该阈值尚未在逐脸场景专门验证。"
FACE_NOTICE = "系统不判断哪张脸被改动，每张脸的候选都只是候选。"


def _thumbnail(query: str, face: dict, index: int, root: Path) -> tuple[str, dict]:
    import cv2
    import numpy as np

    image = cv2.imdecode(np.fromfile(query, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("查询图解码失败")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in face["bbox"]]
    left, top = max(0, int(x1)), max(0, int(y1))
    right, bottom = min(width, int(x2 + 0.999)), min(height, int(y2 + 0.999))
    if left >= right or top >= bottom:
        raise ValueError("人脸框不在图像范围内")
    crop = image[top:bottom, left:right]
    scale = min(1.0, 256 / max(crop.shape[:2]))
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("缩略图编码失败")
    relative = f"identity_faces/face_{index:02d}/thumbnail.png"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.tobytes())
    return relative, {
        "id": f"identity-face-{index:02d}", "kind": "identity_face_thumbnail",
        "path": relative, "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        "size_bytes": path.stat().st_size, "status": "present",
        "description": f"查询图第 {index} 张脸的原图区域缩略图",
    }


def append_face_candidates(result: dict, query: str, faces: list[dict], selected: int,
                           root: Path, searcher: Any, search_kwargs: dict) -> None:
    """附加证据整体失败时仍保留已经完成的主结果。"""
    try:
        _append_face_candidates(result, query, faces, selected, root, searcher, search_kwargs)
    except Exception as exc:
        result["face_search"] = {"status": "error", "limitations": [FACE_NOTICE, FACE_LIMITATION],
                                 "error_message": f"逐脸附加证据未完成（{type(exc).__name__}）"}


def _append_face_candidates(result: dict, query: str, faces: list[dict], selected: int,
                            root: Path, searcher: Any, search_kwargs: dict) -> None:
    """复用已经完成的主脸结果；额外检索和缩略图错误逐脸隔离。"""
    from scale_fallback import _face_area, write_main_face_query

    started = time.perf_counter()
    ordered = sorted(range(len(faces)), key=lambda index: (
        -_face_area(faces[index]), -float(faces[index].get("det_score") or 0.0), index
    ))
    ordered = [selected] + [index for index in ordered if index != selected]
    rows, artifacts = [], []
    for area_rank, index in enumerate(ordered[:MAX_FACES], 1):
        face = faces[index]
        row = {
            "face_index": index, "is_main_face": index == selected, "area_rank": area_rank,
            "status": "success", "query_bbox_original": list(face.get("bbox") or []),
            "det_score": face.get("det_score"), "candidates": [], "face_thumbnail": None,
        }
        payload = result
        if index != selected:
            try:
                masked = write_main_face_query(query, faces, index, root / "identity_faces" / f"face_{index:02d}")
                # 不重复写主脸可视化，不覆盖主脸的任何文件，也不重复加载模型。
                kwargs = {key: value for key, value in search_kwargs.items() if key != "artifact_root"}
                payload = searcher.search(str(masked), top_k=3, progress_reporter=None, **kwargs)
                row["status"] = payload.get("status", "error")
            except Exception as exc:
                row["status"] = "error"
                row["error_message"] = f"该脸比对未完成（{type(exc).__name__}）"
                payload = {"status": "error", "candidates": []}
        row["candidates"] = [
            {key: candidate.get(key) for key in ("rank", "identity_id", "identity_name", "score")}
            for candidate in (payload.get("candidates") or [])[:3]
        ]
        row["open_set"] = decide_identity_open_set(payload)
        try:
            row["face_thumbnail"], artifact = _thumbnail(query, face, index, root)
            artifacts.append(artifact)
        except Exception as exc:
            row["thumbnail_error"] = f"缩略图未生成（{type(exc).__name__}）"
        rows.append(row)
    result["faces"] = rows
    result["face_search"] = {
        "detected_count": len(faces), "processed_count": len(rows), "limit": MAX_FACES,
        "additional_runtime_seconds": time.perf_counter() - started,
        "scope": "query_image", "limitations": [FACE_NOTICE, FACE_LIMITATION],
    }
    if len(faces) > MAX_FACES:
        result["face_search"]["limitations"].append(f"检出 {len(faces)} 张脸，本次仅处理面积最大的 {MAX_FACES} 张。")
    result.setdefault("artifact_manifest", []).extend(artifacts)
