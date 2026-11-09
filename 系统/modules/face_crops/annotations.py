"""生成逐脸编号示意图；原始输入不变，示意图作为新增证据图件保存。"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any


def write_face_annotations(task_dir: Path, query_path: Path, face_branch: dict[str, Any]) -> None:
    """每个已评分画面绘制全部检测框，以粗框指明该画面最高分人脸。"""
    rows = face_branch.get("per_face") or []
    if not rows:
        face_branch["annotated_frames"] = []
        return
    import cv2
    import numpy as np

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source") or query_path.name)].append(row)
    figures = []
    for index, (source, faces) in enumerate(grouped.items(), 1):
        source_path = query_path if source == query_path.name else task_dir / "key_frames" / Path(source).name
        if not source_path.is_file():
            continue
        original = cv2.imdecode(np.fromfile(str(source_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if original is None:
            continue
        highest = max(faces, key=lambda item: float(item["score"]))
        for row in faces:
            left, top, right, bottom = (int(value) for value in row["bbox"])
            selected = row is highest
            thickness = 4 if selected else 2
            cv2.rectangle(original, (left, top), (right, bottom), (0, 0, 0), thickness + 2)
            cv2.rectangle(original, (left, top), (right, bottom), (255, 255, 255), thickness)
            label = f"F{int(row['face_id']):02d}" + (" MAX" if selected else "")
            if row.get("too_small"):
                label += " SMALL"
            anchor = (max(0, left), max(18, top - 8))
            cv2.putText(original, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(original, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        name = f"faces_annotated_{index:03d}.jpg"
        output = task_dir / "face_crops" / name
        ok, encoded = cv2.imencode(".jpg", original, [cv2.IMWRITE_JPEG_QUALITY, 98])
        if not ok:
            continue
        encoded.tofile(str(output))
        figures.append({"file": name, "source": source, "highest_face_id": highest["face_id"],
                        "timestamp_seconds": highest.get("timestamp_seconds"),
                        "note": "编号为逐脸序号，粗框为该画面最高分；SMALL表示人脸过小，分数不可靠。此图为示意图，原始输入保留。"})
    face_branch["annotated_frames"] = figures
