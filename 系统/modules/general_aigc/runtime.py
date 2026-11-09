"""加载模型包并对单个文件打分；单次运行和常驻服务共用这一条路径。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP_ROOT / "engine"))

import torch  # noqa: E402

from aigc_engine.media import MediaError  # noqa: E402
from aigc_engine.scorer import GeneralAigcScorer, logit_to_score  # noqa: E402
from general_aigc_core import (  # noqa: E402
    error_payload,
    enrich_face_branch,
    face_branch_payload,
    input_kind,
    load_calibration,
    success_payload,
    write_json,
)

DEFAULT_PACKAGE = APP_ROOT / "models" / "video-c38-refit3-cf384"


class GeneralAigcRuntime:
    def __init__(self, package: str | Path = DEFAULT_PACKAGE, *, device: str = "auto") -> None:
        self.package = Path(package).expanduser().resolve()
        manifest = json.loads((self.package / "manifest.json").read_text(encoding="utf-8"))
        self.manifest: dict[str, Any] = manifest
        self.scorer = GeneralAigcScorer(
            self.package / manifest["checkpoint"],
            sha256=manifest["checkpoint_sha256"],
            device=device,
        )
        self.calibration = load_calibration(self.package / "calibration.json")
        self.model_info = {
            "model_id": manifest["model_id"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "calibration_version": self.calibration.get("version"),
        }
        self.request_count = 0
        self._gend_scorers: dict[str, Any] = {}

    def health_details(self) -> dict[str, Any]:
        return {
            "worker": "general_aigc",
            "model_id": self.manifest["model_id"],
            "device": str(self.scorer.device),
            "device_name": self.scorer.device_name,
            "load_seconds": round(self.scorer.load_seconds, 3),
            "request_count": self.request_count,
            "calibrated_kinds": sorted(
                kind for kind in ("image", "video") if isinstance(self.calibration.get(kind), dict)
            ),
        }

    def _face_branch(self, face_crops: Any, face_context: Any = None) -> dict[str, Any] | None:
        """face_crops 为 None 表示这次没有做裁脸；空列表表示做了但没有人脸。"""
        if face_crops is None:
            return None
        paths = [Path(str(item)) for item in face_crops]
        try:
            if face_context is not None:
                return self._multiface_branch(paths, face_context)
            logits = self.scorer.score_face_crops(paths) if paths else []
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            return {
                "status": "error",
                "logit": None,
                "band": None,
                "band_label": "未完成",
                "error_message": f"人脸分支执行失败：{exc}",
            }
        return face_branch_payload(logits, [path.name for path in paths], self.calibration)

    def _multiface_branch(self, paths: list[Path], context: dict[str, Any]) -> dict[str, Any]:
        """先按旧批次计算最大脸，再计算额外人脸，保证默认路径的浮点结果。"""
        backend = context.get("backend", "cf384")
        if backend not in {"cf384", "gend"}:
            raise ValueError("人脸后端配置无效")
        faces = context.get("faces") or []
        if not isinstance(faces, list) or not all(isinstance(row, dict) for row in faces):
            raise ValueError("逐脸记录必须为对象列表")
        started = time.perf_counter()
        calibration = self.calibration
        scoring = self.scorer.score_face_crops
        model_info = None
        load_seconds = 0.0
        if backend == "gend":
            package = Path(context.get("model_package") or APP_ROOT / "models" / "gend-face").resolve()
            calibration = load_calibration(package / "calibration.json")
            model_info = {"model_id": "GenD_CLIP_L_14", "score_kind": "model_probability",
                          "score_note": "伪造类别的模型分数，不是材料真伪的概率。"}
            if paths or faces:
                from aigc_engine.gend_face import GenDFaceScorer
                key = str(package)
                if key not in self._gend_scorers:
                    self._gend_scorers[key] = GenDFaceScorer(package, device=str(self.scorer.device))
                    load_seconds = self._gend_scorers[key].load_seconds
                scorer = self._gend_scorers[key]
                scoring = scorer.score_face_crops
                model_info.update(checkpoint_sha256=scorer.manifest["checkpoint_sha256"], device=str(scorer.device))
        inference_started = time.perf_counter()
        legacy_scores = scoring(paths) if paths else []
        by_path = {str(path): value for path, value in zip(paths, legacy_scores, strict=True)}
        extra_paths = [Path(str(row["path"])) for row in faces if str(Path(str(row["path"]))) not in by_path]
        extra_scores = scoring(extra_paths) if extra_paths else []
        by_path.update((str(path), value) for path, value in zip(extra_paths, extra_scores, strict=True))
        scores = [by_path[str(Path(str(row["path"])))] for row in faces]
        original = face_branch_payload(legacy_scores, [path.name for path in paths], calibration if backend == "cf384" else {})
        result = enrich_face_branch(original, faces, scores, calibration, backend=backend,
                                    band_aggregation=context.get("band_aggregation", "largest"),
                                    min_face_side=context.get("min_face_side", 64))
        result["inference_seconds"] = time.perf_counter() - inference_started
        result["load_seconds"] = load_seconds
        result["total_seconds"] = time.perf_counter() - started
        if model_info is not None:
            result["model"] = model_info
            result["score_kind"] = "model_probability"
        result["max_faces"] = context.get("max_faces", 8)
        return result

    def run(self, input_value: Any, output_value: Any, face_crops: Any = None, face_context: Any = None) -> dict[str, Any]:
        started = time.perf_counter()
        query = Path(str(input_value)).expanduser().resolve(strict=False)
        output = Path(str(output_value)).expanduser().resolve(strict=False)
        try:
            result = (
                self.scorer.score_image(query)
                if input_kind(query) == "image"
                else self.scorer.score_video(query)
            )
            payload = success_payload(
                query,
                result,
                score=logit_to_score(result.logit),
                model=self.model_info,
                calibration=self.calibration,
                device=str(self.scorer.device),
                device_name=self.scorer.device_name,
                started=started,
            )
        except MediaError as exc:
            payload = error_payload(
                query, f"文件解码失败（{exc.code}）：{exc.message}", started=started, error_type=exc.code
            )
        except torch.cuda.OutOfMemoryError as exc:
            payload = error_payload(
                query,
                "显存不足，通用 AIGC 分支没有完成。请关闭占用显卡的程序后重试。",
                started=started,
                error_type=type(exc).__name__,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            payload = error_payload(
                query, f"通用 AIGC 分支执行失败：{exc}", started=started, error_type=type(exc).__name__
            )
        general_seconds = time.perf_counter() - started
        face_branch = self._face_branch(face_crops, face_context)
        if face_branch is not None:
            payload["face_branch"] = face_branch
            payload["runtime_seconds"] = time.perf_counter() - started
            if face_context is not None:
                payload["branch_timing"] = {"general_seconds": general_seconds,
                                            "face_seconds": face_branch.get("inference_seconds")}
        self.request_count += 1
        write_json(output, payload)
        return payload
