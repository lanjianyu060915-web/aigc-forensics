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

    def run(self, input_value: Any, output_value: Any) -> dict[str, Any]:
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
        self.request_count += 1
        write_json(output, payload)
        return payload
