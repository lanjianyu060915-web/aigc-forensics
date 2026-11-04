"""二代整图深伪检测器（CommunityForensics DeepfakeDet-ViT，ONNX Runtime 推理）。
只依赖 numpy 与 Pillow，运行在 packaging_v0.4/resources/portable_python_v2 下。
本文件不包含任何阈值常量：所有判定边界来自 calibration.json，便于审计与重新标定。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class DetectorError(RuntimeError):
    """可预期的检测器错误，供上层转成结构化失败结果。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Band:
    key: str
    label: str
    description: str


class CommunityForensicsDetector:
    """整图深伪风险筛查分支。

    该分支在整图/高分辨率照片上有效（SWAN-DF 案例11-20 留出集 AUC 0.9925），
    在 112px 对齐人脸裁剪输入上接近随机（data_formal AUC≈0.43），
    因此必须与 v0.4.0 人脸裁剪分支按输入分辨率路由使用。
    """

    def __init__(self, model_dir: Path, calibration_path: Path, *, verify_hash: bool = True) -> None:
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        self.calibration_path = Path(calibration_path)
        self.calibration = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        model_path = self.model_dir / "model.onnx"
        if not model_path.is_file():
            raise DetectorError(f"二代检测器模型缺失：{model_path}")
        expected = (self.calibration.get("model_provenance") or {}).get("sha256")
        self.model_sha256 = sha256_file(model_path)
        if verify_hash and expected and self.model_sha256 != expected:
            raise DetectorError(
                f"二代检测器模型哈希不匹配：期望 {expected}，实际 {self.model_sha256}"
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 0
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        preprocess = self.calibration["preprocessing"]
        self.shortest_edge = int(preprocess["resize"]["shortest_edge"])
        self.crop_size = int(preprocess["center_crop"])
        self.rescale = float(preprocess["rescale"])
        self.mean = np.asarray(preprocess["normalize_mean"], dtype=np.float32)
        self.std = np.asarray(preprocess["normalize_std"], dtype=np.float32)
        bands = self.calibration["bands"]
        self.low_max = float(bands["low_max"])
        self.medium_max = float(bands["medium_max"])
        self.band_labels = dict(bands["labels"])
        # 视频级分带阈值：由独立视频校准集标定，供 video_pipeline 聚合层使用。
        # 校准文件里没有这一段时保持空字典，聚合层会自动退回帧级一致性规则。
        video_bands = self.calibration.get("video_bands") or {}
        self.video_bands: dict[str, float] = {}
        for key in ("low_max", "medium_max"):
            value = video_bands.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.video_bands[key] = float(value)
        self.load_count = 0

    @classmethod
    def from_app_root(cls, app_root: Path) -> "CommunityForensicsDetector":
        root = Path(app_root) / "packaging_v0.4" / "resources" / "detectors_v2"
        return cls(root / "communityforensics_vit", root / "communityforensics_vit" / "calibration.json")

    def _preprocess(self, image) -> np.ndarray:
        from PIL import Image

        width, height = image.size
        if width <= 0 or height <= 0:
            raise DetectorError("输入图像尺寸非法")
        if width <= height:
            new_width = self.shortest_edge
            new_height = int(self.shortest_edge * height / width)
        else:
            new_height = self.shortest_edge
            new_width = int(self.shortest_edge * width / height)
        resized = image.resize((new_width, new_height), Image.BICUBIC)
        top = (new_height - self.crop_size) // 2
        left = (new_width - self.crop_size) // 2
        cropped = resized.crop((left, top, left + self.crop_size, top + self.crop_size))
        array = np.asarray(cropped, dtype=np.float32) * self.rescale
        array = (array - self.mean) / self.std
        return np.ascontiguousarray(np.transpose(array, (2, 0, 1))[None, ...])

    def band_for(self, score: float) -> Band:
        if score > self.medium_max:
            return Band("high", self.band_labels["high"],
                        "分数超过校准集真人图分布上界，提示高风险，仍需人工复核")
        if score > self.low_max:
            return Band("medium", self.band_labels["medium"],
                        "分数落在校准集真人图高分尾部，证据不足以直接判定，建议人工复核")
        return Band("low", self.band_labels["low"],
                    "分数与已知真人图分布无显著差异，当前证据不足以支持深伪判断")

    def predict(self, image_path: Path) -> dict[str, Any]:
        from PIL import Image

        path = Path(image_path)
        if not path.is_file():
            raise DetectorError(f"输入文件不存在：{path}")
        started = time.perf_counter()
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise DetectorError(f"图像解码失败：{exc}") from exc
        original_size = [int(image.size[0]), int(image.size[1])]
        tensor = self._preprocess(image)
        logits = self.session.run([self.output_name], {self.input_name: tensor})[0]
        raw = float(np.ravel(logits)[0])
        score = float(1.0 / (1.0 + np.exp(-raw)))
        band = self.band_for(score)
        self.load_count += 1
        return {
            "score": score,
            "logit": raw,
            "band": band.key,
            "band_label": band.label,
            "band_reason": band.description,
            "is_deepfake": band.key == "high",
            "original_size": original_size,
            "short_edge": min(original_size),
            "input_sha256": sha256_file(path),
            "model_sha256": self.model_sha256,
            "inference_seconds": time.perf_counter() - started,
            "bands": {"low_max": self.low_max, "medium_max": self.medium_max},
            "video_bands": dict(self.video_bands),
        }
