"""GenD 人脸后端；按上游 CLIP-L/14 结构离线加载固定权重。

结构依据 GenD 的 MIT 许可实现：视觉编码器池化输出，经归一化线性头分类。
通用 AIGC 模型、融合和打分路径不经过本模块。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

CHECKPOINT_SHA256 = "d76f0bdfd74a29fe1b1c1b84a80ac92486993e426878e8c7a3944281fbb96833"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_package(package: str | Path) -> dict[str, Any]:
    """先核对固定权重及配置摘要，再允许解析模型；路径必须在模型包内。"""
    root = Path(package).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("GenD 清单中的权重摘要不是已核对的固定版本")
    files = dict(manifest.get("files_sha256") or {})
    checkpoint = str(manifest.get("checkpoint") or "")
    if not checkpoint:
        raise ValueError("GenD 清单缺少权重文件名")
    files[checkpoint] = CHECKPOINT_SHA256
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("GenD 模型文件路径超出模型包")
        if _sha256(path) != expected:
            raise ValueError(f"GenD 文件 SHA-256 校验失败：{name}")
    return manifest


class GenDFaceScorer:
    """常驻进程内复用的 GenD 模型，输出伪造类别的 softmax 分数。"""

    def __init__(self, package: str | Path, *, device: str = "auto") -> None:
        started = time.perf_counter()
        self.package = Path(package).expanduser().resolve()
        self.manifest = verify_package(self.package)
        import torch
        from safetensors.torch import load_file
        from transformers import CLIPConfig, CLIPModel, CLIPProcessor

        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("GenD 设备配置无效")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GenD 指定了 CUDA，但当前没有可用显卡")
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        config = json.loads((self.package / "config.json").read_text(encoding="utf-8"))
        if config.get("backbone") != "openai/clip-vit-large-patch14" or config.get("head") != "LinearNorm":
            raise ValueError("GenD 结构配置与固定权重不一致")
        self.processor = CLIPProcessor.from_pretrained(self.package / "clip_config", local_files_only=True)
        clip_config = CLIPConfig.from_pretrained(self.package / "clip_config", local_files_only=True)

        class ClipFeatures(torch.nn.Module):
            def __init__(self):
                super().__init__()
                clip = CLIPModel(clip_config)
                self.vision_model = clip.vision_model
                self.visual_projection = clip.visual_projection

            def forward(self, images):
                return self.vision_model(images).pooler_output

        class NormalizedLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(clip_config.vision_config.hidden_size, 2)

            def forward(self, features):
                return self.linear(torch.nn.functional.normalize(features, p=2, dim=1))

        class GenD(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor = ClipFeatures()
                self.model = NormalizedLinear()

            def forward(self, images):
                return self.model(self.feature_extractor(images))

        with torch.device("meta"):
            model = GenD()
        weights = load_file(str(self.package / self.manifest["checkpoint"]), device="cpu")
        model.load_state_dict(weights, strict=True, assign=True)
        del weights
        embeddings = model.feature_extractor.vision_model.embeddings
        embeddings.position_ids = torch.arange(embeddings.num_positions, device="cpu").expand((1, -1))
        if any(value.device.type == "meta" for value in (*model.parameters(), *model.buffers())):
            raise RuntimeError("GenD 存在未载入的参数或缓冲区")
        self.model = model.eval().to(self.device)
        self.load_seconds = time.perf_counter() - started

    def preprocess(self, image):
        """严格使用模型包中保存的上游预处理配置。"""
        return self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]

    def score_face_crops(self, paths: list[str | Path]) -> list[float]:
        import torch
        from PIL import Image

        scores = []
        # 单图前向与已核对的上游入口一致，也避免多人数量改变单脸的计算批次。
        for path in paths:
            with Image.open(path) as image:
                tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                value = float(self.model(tensor).softmax(dim=-1)[0, 1].float().cpu())
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError("GenD 返回了无效的模型分数")
            scores.append(value)
        return scores
