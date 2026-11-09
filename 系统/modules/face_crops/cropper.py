# -*- coding: utf-8 -*-
"""检测逐脸并裁出带边距的方形区域，保留旧最大脸接口。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CROP_MARGIN = 1.3
MIN_FACE_PIXELS = 40


class FaceCropper:
    def __init__(self, model_root: str | Path, *, det_size: int = 640) -> None:
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name="buffalo_l",
            root=str(model_root),
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def largest_face_box(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        boxes = self.face_boxes(frame, max_faces=1)
        return boxes[0] if boxes else None

    def face_boxes(self, frame: np.ndarray, *, max_faces: int = 8) -> list[tuple[int, int, int, int]]:
        """按原检测顺序稳定地排序；同面积时最大脸仍与旧实现一致。"""
        return self.inspect_faces(frame, max_faces=max_faces)["boxes"]

    def inspect_faces(self, frame: np.ndarray, *, max_faces: int = 8) -> dict:
        """一次检测的完整清单；旧接口仍只返回同样顺序的保留框。

        序号只标识这次检测，不能据此跨图片认人。超限框只登记，不裁剪或评分。
        """
        if not isinstance(max_faces, int) or isinstance(max_faces, bool) or max_faces < 1:
            raise ValueError("最多处理人脸数必须为正整数")
        faces = list(enumerate(self.app.get(frame)))
        faces = sorted(faces, key=lambda pair: float(
            (pair[1].bbox[2] - pair[1].bbox[0]) * (pair[1].bbox[3] - pair[1].bbox[1])), reverse=True)
        boxes = [tuple(int(value) for value in item.bbox) for _, item in faces[:max_faces]]
        records = []
        for rank, (detector_index, item) in enumerate(faces, 1):
            # Invalid unselected coordinates must not change the legacy selected inputs.
            try:
                box = [int(value) for value in item.bbox]
            except (TypeError, ValueError, OverflowError):
                box = None
            records.append({"face_id": rank, "detector_index": detector_index,
                            "bbox": box, "selected_for_crop": rank <= max_faces,
                            "exclusion_reason": None if rank <= max_faces else "over_limit"})
        return {"status": "success" if faces else "no_face", "boxes": boxes,
                "detections": records, "detected_total": len(faces),
                "selected_count": len(boxes), "truncated_count": max(0, len(faces) - max_faces),
                "count_basis": "detector_output", "max_faces": max_faces}

    def largest_face_crop(self, frame: np.ndarray) -> np.ndarray | None:
        box = self.largest_face_box(frame)
        return square_crop(frame, box) if box is not None else None


def square_crop(frame: np.ndarray, box: tuple[int, int, int, int], *, min_pixels: int = MIN_FACE_PIXELS) -> np.ndarray | None:
    """以人脸框中心取边长为长边 1.3 倍的方块，超出画面的部分收缩，不补边。"""
    bounds = crop_bounds(frame.shape, box, min_pixels=min_pixels)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    return frame[y0:y1, x0:x1].copy()


def crop_bounds(shape, box: tuple[int, int, int, int], *, min_pixels: int = MIN_FACE_PIXELS) -> tuple[int, int, int, int] | None:
    """与旧裁剪完全相同的整数边界，供裁剪和坐标记录共同使用。"""
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    if min(width, height) < min_pixels:
        return None
    half = int(max(width, height) * CROP_MARGIN / 2)
    center_x, center_y = (left + right) // 2, (top + bottom) // 2
    x0, y0 = max(0, center_x - half), max(0, center_y - half)
    x1, y1 = min(shape[1], center_x + half), min(shape[0], center_y + half)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def video_frames(path: str | Path, count: int) -> list[np.ndarray]:
    """均匀取帧；读不出的帧跳过。"""
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        frames = []
        for index in np.linspace(0, max(total - 1, 0), count).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(frame)
        return frames
    finally:
        capture.release()
