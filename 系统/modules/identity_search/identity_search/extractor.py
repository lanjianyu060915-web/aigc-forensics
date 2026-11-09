"""Chinese-path-safe image decoding and delayed InsightFace loading."""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Sequence

from .errors import (
    DependencyError,
    ImageDecodeError,
    InputValidationError,
    InferenceError,
    InvalidEmbeddingError,
    MultipleFacesError,
    NoFaceError,
    ProviderError,
)
from .model import MODEL_NAME


def face_diagnostics(face) -> dict[str, object]:
    """Extract only fields genuinely returned by the InsightFace face object."""

    diagnostics: dict[str, object] = {}
    bbox = getattr(face, "bbox", None)
    if bbox is not None:
        try:
            values = [float(value) for value in bbox]
        except (TypeError, ValueError):
            values = []
        if len(values) == 4:
            diagnostics["bbox"] = values
    detection_score = getattr(face, "det_score", None)
    if detection_score is not None:
        try:
            score = float(detection_score)
        except (TypeError, ValueError):
            score = None
        if score is not None:
            diagnostics["det_score"] = score
    keypoints = getattr(face, "kps", None)
    if keypoints is not None:
        try:
            points = [
                [float(point[0]), float(point[1])]
                for point in keypoints
            ]
        except (TypeError, ValueError, IndexError):
            points = []
        if len(points) == 5:
            diagnostics["keypoints_5"] = points
            diagnostics["landmark_count"] = 5
    return diagnostics


def enforce_offline_import_policy() -> None:
    """Disable dependency update checks that would otherwise contact PyPI."""

    # Albumentations is imported transitively by InsightFace 0.7.3 and performs
    # a PyPI version request at import time unless this documented flag is set.
    # Assignment (rather than setdefault) makes the search policy unconditional.
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def providers_for_device(
    device: str, available_providers: Iterable[str]
) -> tuple[list[str], int]:
    """Return an explicit ONNX provider order and default InsightFace ctx id."""

    available = set(available_providers)
    if device == "cpu":
        if "CPUExecutionProvider" not in available:
            raise ProviderError(
                "当前 ONNX Runtime 不提供 CPUExecutionProvider，无法建立 CPU 基线"
            )
        return ["CPUExecutionProvider"], -1
    if device == "gpu":
        if "CUDAExecutionProvider" not in available:
            listed = ", ".join(sorted(available)) or "无"
            raise ProviderError(
                "请求 GPU，但 CUDAExecutionProvider 不可用；"
                f"当前 providers: {listed}"
            )
        providers = ["CUDAExecutionProvider"]
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        return providers, 0
    raise ProviderError(f"不支持的设备: {device}")


def read_image_unicode(image_path: os.PathLike[str] | str):
    """Decode an image through ``fromfile + imdecode`` for Windows Unicode paths."""

    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CLI test
        raise DependencyError(f"缺少图像依赖: {exc.name}") from exc

    path = Path(image_path)
    if not path.is_file():
        raise ImageDecodeError(f"图片文件不存在: {path}")
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    except (OSError, ValueError, cv2.error) as exc:
        raise ImageDecodeError(f"图片读取失败: {path}（{exc}）") from exc
    if image is None:
        raise ImageDecodeError(f"图片无法解码或文件已损坏: {path}")
    return image


class FeatureExtractor:
    """InsightFace detection and normalized embedding extraction.

    Heavy dependencies are imported only when a real app must be constructed.
    Tests may inject an ``app`` and ``image_loader`` without loading a model.
    """

    def __init__(
        self,
        *,
        model_root: os.PathLike[str] | str | None = None,
        device: str = "cpu",
        ctx_id: int | None = None,
        det_size: Sequence[int] = (1280, 1280),
        app=None,
        image_loader: Callable[[os.PathLike[str] | str], object] | None = None,
    ) -> None:
        self.device = device
        self.det_size = (int(det_size[0]), int(det_size[1]))
        self.image_loader = image_loader or read_image_unicode

        if app is None:
            enforce_offline_import_policy()
            try:
                import onnxruntime as ort
                from insightface.app import FaceAnalysis
            except ModuleNotFoundError as exc:
                raise DependencyError(f"缺少推理依赖: {exc.name}") from exc

            providers, default_ctx = providers_for_device(
                device, ort.get_available_providers()
            )
            actual_ctx = default_ctx if ctx_id is None else int(ctx_id)
            if device == "cpu" and actual_ctx != -1:
                raise ProviderError("CPU 模式的 ctx_id 必须为 -1")
            if device == "gpu" and actual_ctx < 0:
                raise ProviderError("GPU 模式的 ctx_id 必须为非负整数")
            try:
                self.app = FaceAnalysis(
                    name=MODEL_NAME,
                    root=str(Path(model_root).expanduser()),
                    providers=providers,
                    # 结果及证据只使用检测框、五点关键点与身份特征。
                    allowed_modules=["detection", "recognition"],
                )
                self.app.prepare(ctx_id=actual_ctx, det_size=self.det_size)
            except Exception as exc:
                raise InferenceError(f"InsightFace 初始化失败: {exc}") from exc
        else:
            self.app = app

    def extract(self, image_path: os.PathLike[str] | str):
        """Return one normalized embedding and its face, or raise a precise error."""

        try:
            import numpy as np
        except ModuleNotFoundError as exc:  # pragma: no cover - CLI catches first
            raise DependencyError("缺少数值依赖: numpy") from exc

        image = self.image_loader(image_path)
        if image is None:
            raise ImageDecodeError(f"图片无法解码或文件已损坏: {image_path}")
        try:
            faces = list(self.app.get(image))
        except Exception as exc:
            raise InferenceError(f"人脸推理失败: {image_path}（{exc}）") from exc
        if not faces:
            raise NoFaceError(
                f"图片中未检测到人脸: {image_path}",
                diagnostics={"face_count": 0, "detected_faces": []},
            )
        if len(faces) > 1:
            raise MultipleFacesError(
                f"图片中检测到 {len(faces)} 张人脸，当前要求恰好 1 张: {image_path}",
                diagnostics={
                    "face_count": len(faces),
                    "detected_faces": [face_diagnostics(item) for item in faces],
                },
            )

        face = faces[0]
        return self._normalized_face_embedding(face, image_path)

    def extract_selected_face(self, image_path, selected_face, *, selection_context,
                              selected_face_enabled=False, selected_face_experimental=False):
        """Experimental route: reuse the same recognition model and its alignment.

        No detector is run and no fallback is attempted. The original extract()
        path stays the production default. Return metadata with the input/selection
        key so a selected-face result cannot masquerade as a legacy cached query.
        """
        from .selected_face import require_selected_face_opt_in, validate_selected_face
        require_selected_face_opt_in(selected_face_enabled, selected_face_experimental)
        try:
            import cv2
            import numpy as np
        except ModuleNotFoundError as exc:
            raise DependencyError(f"缺少图像依赖: {exc.name}") from exc
        try:
            encoded = Path(image_path).read_bytes()
        except OSError as exc:
            raise ImageDecodeError("显式选脸输入无法读取。") from exc
        digest = hashlib.sha256(encoded).hexdigest()
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR) if encoded else None
        if image is None:
            raise ImageDecodeError("显式选脸输入无法解码。")
        metadata = validate_selected_face(selected_face, context=selection_context, input_sha256=digest,
                                          image_size=(int(image.shape[1]), int(image.shape[0])))
        models = getattr(self.app, "models", {})
        recognition = models.get("recognition") if isinstance(models, dict) else None
        if recognition is None or not callable(getattr(recognition, "get", None)):
            raise InputValidationError("当前特征器未提供可复用的 recognition.get，不能启用显式选脸。")
        face = SimpleNamespace(bbox=np.asarray(metadata["bbox"], dtype=np.float32),
                               kps=np.asarray(metadata["keypoints_5"], dtype=np.float32),
                               det_score=metadata["det_score"], embedding=None)
        try:
            # The frozen recognition implementation performs its existing norm_crop
            # and feature computation. No alignment or preprocessing is reimplemented.
            recognition.get(image, face)
        except Exception as exc:
            raise InferenceError(f"显式选脸特征提取失败（{type(exc).__name__}）。") from exc
        vector, face = self._normalized_face_embedding(face, image_path)
        metadata.update(detection_performed=False, equivalence_status="not_validated_on_independent_data")
        return vector, face, metadata

    def _normalized_face_embedding(self, face, image_path):
        """Same normalization for the legacy and opt-in routes."""
        import numpy as np
        detected_face = face_diagnostics(face)
        embedding = getattr(face, "embedding", None)
        if embedding is None:
            raise InvalidEmbeddingError(
                f"模型未返回人脸特征: {image_path}",
                diagnostics={
                    "face_count": 1,
                    "detected_faces": [detected_face],
                    "embedding_dimension": None,
                },
            )
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise InvalidEmbeddingError(
                f"模型返回了无效人脸特征: {image_path}",
                diagnostics={
                    "face_count": 1,
                    "detected_faces": [detected_face],
                    "embedding_dimension": int(vector.shape[0]),
                    "embedding_norm": norm,
                },
            )
        vector = vector / norm
        if not np.all(np.isfinite(vector)):
            raise InvalidEmbeddingError(
                f"模型返回了非有限人脸特征: {image_path}",
                diagnostics={
                    "face_count": 1,
                    "detected_faces": [detected_face],
                    "embedding_dimension": int(vector.shape[0]),
                    "embedding_norm": norm,
                },
            )
        return vector, face
