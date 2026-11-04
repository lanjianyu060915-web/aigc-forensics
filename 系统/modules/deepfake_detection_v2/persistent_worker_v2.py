"""二代鉴伪模块持久工作进程（JSON 行协议，与 v0.4.0 冻结 worker 完全一致）。
与冻结 worker 的差别只在检测内核：本进程不加载 MobileNetV3/频域模型，
而是加载 packaging_v0.4/resources/detectors_v2 下的 ONNX 整图检测器。
协议：face-source-trace-runtime-v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from vit_detector import CommunityForensicsDetector, DetectorError, sha256_file
from frozen_branch import FrozenBranchError, FrozenBranchProxy


PROTOCOL_NAME = "face-source-trace-runtime-v1"
PROTOCOL_OUTPUT: TextIO = sys.stdout
DEFAULT_APP_ROOT = Path(__file__).resolve().parents[2]

LIMITATIONS = [
    "模型输出为深伪风险筛查分数，不是系统准确率或司法鉴定结论。",
    "三个模块的分数属于不同任务，不能相加或解释为综合概率。",
    "二代整图分支在 112px 级对齐人脸裁剪输入上无效，必须与 v0.4.0 人脸裁剪分支按输入分辨率路由使用。",
    "判定分带来自独立校准集的分位数，更换数据来源后必须重新标定。",
]


def _emit(message_type: str, *, request_id: str | None, ready: bool, health: str,
          **fields: Any) -> None:
    message = {
        "protocol": PROTOCOL_NAME,
        "type": message_type,
        "request_id": request_id,
        "ready": bool(ready),
        "health": str(health),
        **fields,
    }
    PROTOCOL_OUTPUT.write("\n" + json.dumps(message, ensure_ascii=False) + "\n")
    PROTOCOL_OUTPUT.flush()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _base_payload(query: Path, runtime_seconds: float) -> dict[str, Any]:
    return {
        "schema_version": "3.0",
        "module": "deepfake",
        "status": "error",
        "query_image": str(query),
        "is_deepfake": None,
        "score": None,
        "label": None,
        "runtime_seconds": max(0.0, float(runtime_seconds)),
        "details": {},
        "error_message": None,
        "method": {
            "name": "二代整图深伪风险甄别（ViT 分支）",
            "branch": "vit_full_image",
            "model": "CommunityForensics-DeepfakeDet-ViT (ViT-Small/384, ONNX Runtime)",
            "fusion_method": "不做跨分支加权融合，按输入有效分辨率路由",
        },
        "metrics": {},
        "explanation": {},
        "visual_evidence": {},
        "artifact_manifest": [],
        "progress_events": [],
        "warnings": [],
        "limitations": list(LIMITATIONS),
        "runtime": {"elapsed_seconds": max(0.0, float(runtime_seconds))},
        "provenance": {},
    }


def _short_edge(path: Path) -> int:
    """只读图像头拿尺寸，失败时返回 0（交由调用方决定如何处理）。"""
    from PIL import Image

    try:
        with Image.open(path) as handle:
            width, height = handle.size
        return int(min(width, height))
    except Exception:  # noqa: BLE001
        return 0


class PersistentVitRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.app_root = Path(args.app_root).expanduser().resolve(strict=False)
        self.writable_root = Path(args.writable_root).expanduser().resolve(strict=False)
        self.writable_root.mkdir(parents=True, exist_ok=True)
        resources = self.app_root / "packaging_v0.4" / "resources" / "detectors_v2"
        started = time.perf_counter()
        self.detector = CommunityForensicsDetector(
            resources / "communityforensics_vit",
            resources / "communityforensics_vit" / "calibration.json",
            verify_hash=not args.no_hash_check,
        )
        self.load_seconds = time.perf_counter() - started
        self.routing = json.loads((resources / "routing.json").read_text(encoding="utf-8"))
        self.routing_threshold = int(self.routing.get("short_edge_threshold_px", 256))
        self.delegation_enabled = not args.no_delegation
        self.frozen = None
        if self.delegation_enabled:
            self.frozen = FrozenBranchProxy(
                self.app_root, self.writable_root / "frozen_branch",
                device=args.device, install_root=Path(args.install_root),
            )
        self.device = args.device
        self.request_count = 0

    def health_details(self) -> dict[str, Any]:
        return {
            "worker": "deepfake_v2",
            "pid": os.getpid(),
            "branch": "vit_full_image",
            "model_sha256": self.detector.model_sha256,
            "load_seconds": round(self.load_seconds, 3),
            "request_count": self.request_count,
            "routing_short_edge_threshold_px": self.routing.get("short_edge_threshold_px"),
            "delegation_enabled": self.delegation_enabled,
            "frozen_branch_available": bool(self.frozen and self.frozen.available),
            "frozen_request_count": int(self.frozen.request_count if self.frozen else 0),
            "frozen_install_root": str(self.frozen.install_root) if self.frozen else None,
            "requested_device": self.device,
            "actual_device": "CPU",
            "dependency_isolation": "dedicated_process",
        }

    def run(self, query_value: Any, output_value: Any) -> dict[str, Any]:
        started = time.perf_counter()
        query = Path(str(query_value)).expanduser().resolve(strict=False)
        output = Path(str(output_value)).expanduser().resolve(strict=False)
        short_edge = _short_edge(query)
        use_frozen = (
            self.delegation_enabled
            and self.frozen is not None
            and self.frozen.available
            and 0 < short_edge < self.routing_threshold
        )
        if use_frozen:
            return self._run_frozen(query, output, short_edge, started)
        return self._run_vit(query, output, short_edge, started)

    def _run_frozen(self, query: Path, output: Path, short_edge: int,
                    started: float) -> dict[str, Any]:
        payload = None
        try:
            payload = self.frozen.run(query, output)
        except FrozenBranchError as exc:
            payload = None
            failure = str(exc)
        else:
            failure = ""
        if payload is None:
            payload = _base_payload(query, time.perf_counter() - started)
            payload["error_message"] = f"人脸裁剪分支回退失败：{failure}"
            payload["details"] = {"error_type": "FrozenBranchError",
                                  "routed_branch": "v040_face_crop"}
        else:
            payload.setdefault("details", {})
            payload["details"].update({
                "routed_branch": "v040_face_crop",
                "routing_reason": (f"输入短边 {short_edge}px 低于阈值 "
                                   f"{self.routing_threshold}px，交由冻结人脸裁剪分支处理"),
                "routing_threshold_px": self.routing_threshold,
                "short_edge_px": short_edge,
            })
            payload["method"] = {
                **(payload.get("method") or {}),
                "branch": "v040_face_crop",
                "routing": "二代 worker 内部代理到 v0.4.0 冻结运行时",
            }
        payload["runtime_seconds"] = time.perf_counter() - started
        payload.setdefault("runtime", {})["elapsed_seconds"] = payload["runtime_seconds"]
        self.request_count += 1
        _write_json(output, payload)
        return payload

    def _run_vit(self, query: Path, output: Path, short_edge: int,
                 started: float) -> dict[str, Any]:
        payload = _base_payload(query, 0.0)
        try:
            result = self.detector.predict(query)
            short_edge = int(result["short_edge"])
            threshold = self.routing_threshold
            payload["status"] = "success"
            payload["is_deepfake"] = bool(result["is_deepfake"])
            payload["score"] = result["score"]
            payload["label"] = "fake" if result["is_deepfake"] else "real"
            payload["details"] = {
                "score_semantics": "二代整图检测器的 sigmoid 深伪风险分；不是系统准确率。",
                "branch": "vit_full_image",
                "band": result["band"],
                "band_label": result["band_label"],
                "band_reason": result["band_reason"],
                "band_thresholds": result["bands"],
                "video_band_thresholds": result["video_bands"],
                "short_edge_px": short_edge,
                "routing_threshold_px": threshold,
                "routing_matched": short_edge >= threshold,
                "logit": round(result["logit"], 6),
                "original_size": result["original_size"],
                "inference_seconds": round(result["inference_seconds"], 4),
                "calibration_set": self.detector.calibration.get("calibration_set", {}),
                "validation_set": self.detector.calibration.get("validation_set", {}),
                "actual_device": "CPU",
            }
            if short_edge < threshold:
                payload["warnings"].append(
                    "输入短边低于路由阈值，本分支在此类输入上未经证实，"
                    "结果只作参考，应改用人脸裁剪分支。"
                )
            payload["provenance"] = {
                "input_sha256": result["input_sha256"],
                "model_sha256": result["model_sha256"],
                "calibration_path": str(self.detector.calibration_path),
                "routing_config": str(self.app_root / "packaging_v0.4" / "resources"
                                       / "detectors_v2" / "routing.json"),
            }
        except (DetectorError, OSError, ValueError) as exc:
            payload = _base_payload(query, time.perf_counter() - started)
            payload["error_message"] = f"二代检测器执行失败：{exc}"
            payload["details"] = {"error_type": type(exc).__name__}
        payload["runtime_seconds"] = time.perf_counter() - started
        payload["runtime"]["elapsed_seconds"] = payload["runtime_seconds"]
        self.request_count += 1
        _write_json(output, payload)
        return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="二代鉴伪模型持久工作进程")
    parser.add_argument("--app-root", default=str(DEFAULT_APP_ROOT))
    parser.add_argument(
        "--install-root",
        default=str(DEFAULT_APP_ROOT / "packaging_v0.3" / "resources" / "deepfake_runtime"),
        help="冻结回退分支的运行时根目录（与冻结 worker 同名参数，便于配置级替换）",
    )
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu",
                        help="接收冻结 worker 同名参数；本分支当前固定使用 CPU")
    parser.add_argument("--writable-root",
                        default=str(DEFAULT_APP_ROOT / "runtime" / "persistent_services"
                                    / "deepfake_v2"))
    parser.add_argument("--no-hash-check", action="store_true")
    parser.add_argument("--no-delegation", action="store_true",
                        help="关闭低于路由阈值时的冻结分支代理（仅用于隔离测试）")
    return parser


def _serve(runtime: PersistentVitRuntime) -> int:
    _emit("ready", request_id=None, ready=True, health="ok", ok=True,
          details=runtime.health_details())
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict) or message.get("protocol") != PROTOCOL_NAME \
                or message.get("type") != "request":
            continue
        request_id = message.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        action = message.get("action")
        if action == "health":
            _emit("health", request_id=request_id, ready=True, health="ok", ok=True,
                  details=runtime.health_details())
            continue
        if action == "shutdown":
            _emit("response", request_id=request_id, ready=True, health="ok", ok=True,
                  shutting_down=True)
            return 0
        if action != "run":
            _emit("response", request_id=request_id, ready=True, health="ok", ok=False,
                  error=f"不支持的 action：{action!r}")
            continue
        payload = runtime.run(message.get("query_path"), message.get("output_path"))
        _emit("response", request_id=request_id, ready=True, health="ok",
              ok=payload.get("status") == "success", payload=payload)
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        runtime = PersistentVitRuntime(args)
    except Exception as exc:  # noqa: BLE001
        _emit("ready", request_id=None, ready=False, health="error", ok=False,
              error=f"{type(exc).__name__}: {exc}")
        return 2
    return _serve(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
