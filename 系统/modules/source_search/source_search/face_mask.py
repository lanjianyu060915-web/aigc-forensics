from __future__ import annotations

import cv2
import numpy as np


class FaceMasker:
    def __init__(self, expansion: float = 0.25):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("OpenCV Haar cascade could not be loaded")
        self.expansion = expansion

    def detect(self, image: np.ndarray) -> tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
            )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))
        pad_x, pad_y = round(w * self.expansion), round(h * self.expansion)
        height, width = image.shape[:2]
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(width, x + w + pad_x), min(height, y + h + pad_y)
        return x1, y1, x2, y2

    @staticmethod
    def mask(shape: tuple[int, ...], box: tuple[int, int, int, int] | None) -> np.ndarray:
        mask = np.full(shape[:2], 255, dtype=np.uint8)
        if box is not None:
            x1, y1, x2, y2 = box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 0, -1)
        return mask
