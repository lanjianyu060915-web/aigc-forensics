# -*- coding: utf-8 -*-
"""视频抽帧与关键帧选择（v2 视频入口第一步）。

职责：把一个视频文件变成「少量、可复核、被三个模块共同使用」的关键帧图片。

流程（全部写进结果 JSON，可复算）：

1. 探测视频基本信息（时长、fps、分辨率、总帧数）并计算 SHA-256；
2. 在去掉首尾的时间轴上均匀取 ``--candidate-frames`` 个候选帧；
3. 每个候选帧计算客观指标：清晰度（Laplacian 方差）、亮度、对比度；
4. 可选启用 InsightFace 人脸检测，记录人脸数量、最大人脸框与检测分；
5. 调 ``video_pipeline.selection.select_key_frames`` 选出关键帧并落盘为 JPEG。

刻意不做的事（避免把工程启发式伪装成证据）：

- 不做跨帧人脸跟踪/身份关联：只做逐帧检测，报告里写清"时序覆盖抽样"；
- 不修改任何算法参数：抽帧结果只决定"给下游看哪几帧"；
- 不产生任何判定分数：质量分只用于排序，三个模块的结论仍来自各自模型。

本模块运行在开发解释器（OpenCV + InsightFace）下，输出纯 JPEG，因此下游
二代鉴伪 worker 仍可只依赖 numpy/Pillow 运行。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parent
PRODUCT_ROOT = MODULE_ROOT.parents[1]
for _entry in (str(PRODUCT_ROOT),):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from progress_events import create_progress_writer  # noqa: E402
from video_pipeline.selection import (  # noqa: E402
    SAMPLE_STRATEGY,
    candidate_indices,
    select_key_frames,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SHARPNESS_SHORT_EDGE = 360
JPEG_QUALITY = 95
SEQUENTIAL_SCAN_LIMIT = 20000

LIMITATIONS = [
    "关键帧是有损抽样：未被选中的帧没有参与任何模块，帧覆盖情况以本文件为准。",
    "质量分是工程启发式（清晰度/亮度/人脸数），只用于排序，不构成证据。",
    "本模块不做跨帧人脸跟踪，只做逐帧人脸检测；时序结论请以时间戳为准。",
]


class FrameSelectionError(RuntimeError):
    """可预期的抽帧失败，供上层转成结构化错误结果。"""


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ascii_media_path(path: Path) -> tuple[Path, str | None]:
    """OpenCV 的 FFMPEG 后端在 Windows 上对非 ASCII 路径不可靠。

    这里不改动原始文件：必要时把视频复制到 ASCII 临时路径解码，并返回
    清理目录。复制不会进入任务目录，也不影响 SHA-256 计算（对原文件计算）。
    """

    try:
        str(path).encode("ascii")
        return path, None
    except UnicodeEncodeError:
        pass
    import shutil
    import tempfile

    staging = Path(tempfile.mkdtemp(prefix="fsts_video_"))
    target = staging / f"input{path.suffix.lower() or '.mp4'}"
    shutil.copy2(path, target)
    return target, str(staging)


def _open_capture(path: Path):
    import cv2

    return cv2.VideoCapture(str(path))


def probe_video(capture) -> dict[str, Any]:
    import cv2

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise FrameSelectionError(
            "无法读取视频基本信息（可能是损坏文件或缺少解码器）："
            f"frame_count={frame_count}, size={width}x{height}"
        )
    duration = frame_count / fps if fps > 0 else 0.0
    return {
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": round(duration, 6),
    }


def _grab_sequential(capture, targets: list[int]) -> dict[int, Any]:
    frames: dict[int, Any] = {}
    position = -1
    for target in targets:
        while position < target - 1:
            if not capture.grab():
                break
            position += 1
        ok, frame = capture.read()
        position += 1
        frames[target] = frame if ok else None
    return frames


def _grab_seek(capture, targets: list[int]) -> dict[int, Any]:
    import cv2

    frames: dict[int, Any] = {}
    for target in targets:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(target))
        ok, frame = capture.read()
        frames[target] = frame if ok else None
    return frames


def sample_frames(capture, indices: list[int], *, frame_count: int) -> tuple[dict[int, Any], str]:
    """按顺序取回候选帧；大视频用定位读取，小视频顺序解码（对编码器更稳）。"""

    strategy = "sequential_grab" if frame_count <= SEQUENTIAL_SCAN_LIMIT else "seek_by_index"
    frames = _grab_sequential(capture, indices) if strategy == "sequential_grab" else _grab_seek(capture, indices)
    missing = [index for index, frame in frames.items() if frame is None]
    if missing and strategy == "seek_by_index":
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        retry = _grab_sequential(capture, missing)
        for index, frame in retry.items():
            if frame is not None:
                frames[index] = frame
        strategy += "+fallback_sequential"
    return frames, strategy


def frame_metrics(image) -> dict[str, float]:
    """客观指标：清晰度（Laplacian 方差）、亮度、对比度。"""

    import cv2
    import numpy as np

    height, width = image.shape[:2]
    scale = SHARPNESS_SHORT_EDGE / float(min(height, width)) if min(height, width) > 0 else 1.0
    if scale < 1.0:
        gray = cv2.cvtColor(
            cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    return {
        "sharpness": round(sharpness, 6),
        "brightness": round(brightness, 6),
        "contrast": round(contrast, 6),
    }


def load_face_detector(*, model_root: Path, det_size: int, device: str, ctx_id: int):
    """加载 InsightFace 检测器；模型缺失时返回 (None, 原因) 而不是崩溃。"""

    try:
        sys.path.insert(0, str(PRODUCT_ROOT / "modules" / "identity_search"))
        from identity_search.extractor import enforce_offline_import_policy, providers_for_device
        from identity_search.model import MODEL_NAME, preflight_model

        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        enforce_offline_import_policy()
        model = preflight_model(model_root, verify_hashes=True)
        providers, default_ctx = providers_for_device(device, ort.get_available_providers())
        actual_ctx = default_ctx if ctx_id is None else int(ctx_id)
        # 只加载检测网络：本模块不做识别，加载 landmark/recognition 会在每帧多花数秒。
        app = FaceAnalysis(
            name=MODEL_NAME,
            root=str(model.root),
            providers=providers,
            allowed_modules=["detection"],
        )
        app.prepare(ctx_id=actual_ctx, det_size=(det_size, det_size))
        return app, ""
    except Exception as exc:  # noqa: BLE001 - 检测器缺失必须降级而不是终止
        return None, f"{type(exc).__name__}: {exc}"


def detect_faces(detector, image) -> list[dict[str, Any]]:
    faces = list(detector.get(image))
    output: list[dict[str, Any]] = []
    for face in faces:
        bbox = getattr(face, "bbox", None)
        try:
            box = [float(value) for value in bbox]
        except (TypeError, ValueError):
            box = []
        score = getattr(face, "det_score", None)
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = None
        output.append({"bbox": box, "det_score": score_value})
    output.sort(key=lambda item: (item["bbox"][3] - item["bbox"][1]) * (item["bbox"][2] - item["bbox"][0]) if len(item["bbox"]) == 4 else 0.0, reverse=True)
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    import cv2

    started = time.perf_counter()
    video = Path(args.video).expanduser().resolve(strict=False)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    selection_path = Path(args.selection).expanduser().resolve(strict=False)
    reporter = create_progress_writer(
        task_id=getattr(args, "task_id", None),
        module="video",
        events_path=getattr(args, "progress_events", None),
        latest_path=getattr(args, "progress_latest", None),
    )

    def emit(stage: str, message: str, *, status: str = "running", details: dict[str, Any] | None = None) -> None:
        if reporter is None:
            return
        try:
            reporter.emit(stage, status, message, progress_percent=None, details=details or {}, force=True)
        except Exception:  # noqa: BLE001 - 进度不得影响抽帧结果
            return

    if not video.is_file():
        raise FrameSelectionError(f"视频文件不存在：{video}")
    if video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise FrameSelectionError(
            f"不支持的视频扩展名：{video.suffix}（支持 {', '.join(sorted(VIDEO_EXTENSIONS))}）"
        )
    if args.max_key_frames <= 0 or args.candidate_frames <= 0:
        raise FrameSelectionError("关键帧数量与候选帧数量必须为正整数")

    emit("video_frame_probe", "正在读取视频基本信息。", details={"file_name": video.name})
    digest = sha256_file(video)
    decode_path, staging = _ascii_media_path(video)
    capture = _open_capture(decode_path)
    if not capture.isOpened():
        if staging:
            import shutil

            shutil.rmtree(staging, ignore_errors=True)
        raise FrameSelectionError(f"OpenCV 无法打开视频（缺少解码器或文件损坏）：{video.name}")
    try:
        info = probe_video(capture)
        indices = candidate_indices(
            info["frame_count"], min(int(args.candidate_frames), int(info["frame_count"]))
        )
        emit(
            "video_frame_decode",
            f"正在解码 {len(indices)} 个候选帧。",
            details={"candidate_count": len(indices), "duration_seconds": info["duration_seconds"]},
        )
        frames, decode_strategy = sample_frames(capture, indices, frame_count=info["frame_count"])
    finally:
        capture.release()

    if staging:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)

    detector, detector_error = (None, "")
    if args.face_detector != "none":
        emit("video_face_detection", "正在初始化人脸检测器（InsightFace buffalo_l）。")
        detector, detector_error = load_face_detector(
            model_root=Path(args.model_root).expanduser(),
            det_size=int(args.det_size),
            device=str(args.device),
            ctx_id=int(args.ctx),
        )

    candidates: list[dict[str, Any]] = []
    decoded = 0
    for index in indices:
        image = frames.get(index)
        timestamp = round(index / info["fps"], 6) if info["fps"] > 0 else None
        if image is None:
            candidates.append(
                {
                    "frame_index": index,
                    "timestamp_seconds": timestamp,
                    "decoded": False,
                    "face_count": None,
                }
            )
            continue
        decoded += 1
        metrics = frame_metrics(image)
        entry: dict[str, Any] = {
            "frame_index": index,
            "timestamp_seconds": timestamp,
            "decoded": True,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            **metrics,
            "face_count": None,
        }
        if detector is not None:
            faces = detect_faces(detector, image)
            entry["face_count"] = len(faces)
            entry["faces"] = faces
            if faces:
                best = faces[0]
                bbox = best.get("bbox") or []
                if len(bbox) == 4:
                    area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
                    entry["largest_face_bbox"] = bbox
                    entry["largest_face_area_ratio"] = round(
                        area / float(max(1, image.shape[0] * image.shape[1])), 8
                    )
                    entry["largest_face_det_score"] = best.get("det_score")
                    entry["largest_face_short_edge_px"] = int(
                        round(min(max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1])))
                    )
        candidates.append(entry)

    usable = [item for item in candidates if item.get("decoded")]
    if not usable:
        raise FrameSelectionError("所有候选帧解码失败，无法抽帧")

    emit(
        "video_frame_selection",
        f"正在按质量与人脸约束挑选关键帧（候选 {len(usable)} 帧）。",
        details={
            "face_frames": sum(1 for item in usable if item.get("face_count") == 1),
            "no_face_frames": sum(1 for item in usable if item.get("face_count") == 0),
        },
    )
    selection = select_key_frames(
        usable,
        max_frames=int(args.max_key_frames),
        duration_seconds=info["duration_seconds"],
        min_sharpness_ratio=float(args.min_sharpness_ratio),
        prefer_single_face=bool(args.prefer_single_face),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    key_frames: list[dict[str, Any]] = []
    for position, entry in enumerate(selection["key_frames"], start=1):
        image = frames.get(int(entry["frame_index"]))
        if image is None:
            continue
        file_name = f"frame_{position:03d}.jpg"
        target = output_dir / file_name
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise FrameSelectionError(f"关键帧编码失败：{file_name}")
        encoded.tofile(target)
        key_frames.append(
            {
                **entry,
                "frame_rank": position,
                "file_name": file_name,
                "path": file_name,
                "sha256": sha256_file(target),
                "size_bytes": target.stat().st_size,
            }
        )

    if not key_frames:
        raise FrameSelectionError("没有选出任何可用关键帧")

    for item in selection["candidates"]:
        item.pop("faces", None)

    face_counts = {
        "frames_with_single_face": sum(1 for item in usable if item.get("face_count") == 1),
        "frames_with_multiple_faces": sum(1 for item in usable if (item.get("face_count") or 0) > 1),
        "frames_without_face": sum(1 for item in usable if item.get("face_count") == 0),
    }
    runtime = time.perf_counter() - started
    warnings: list[str] = []
    if detector is None and args.face_detector != "none":
        warnings.append(
            "人脸检测不可用，选帧退化为清晰度/亮度排序（原因：" + (detector_error or "未知") + "）"
        )
    if face_counts["frames_with_single_face"] == 0 and detector is not None:
        warnings.append("候选帧中没有检出恰好一张人脸，身份检索将无法在这些帧上执行。")

    payload = {
        "schema_version": "3.0",
        "module": "video",
        "status": "success",
        "generated_at": _now_iso(),
        "runtime_seconds": runtime,
        "video_path": str(video),
        "video_sha256": digest,
        "video": {
            "file_name": video.name,
            "size_bytes": video.stat().st_size,
            **info,
        },
        "sampling": {
            "strategy": SAMPLE_STRATEGY,
            "candidate_frames_requested": int(args.candidate_frames),
            "candidate_frames_decoded": len(usable),
            "candidate_indices": [int(item.get("frame_index")) for item in usable],
            "decode_strategy": decode_strategy,
            "jpeg_quality": JPEG_QUALITY,
        },
        "face_detection": {
            "requested": args.face_detector != "none",
            "available": detector is not None,
            "method": "insightface_buffalo_l" if detector is not None else "unavailable",
            "det_size": int(args.det_size) if detector is not None else None,
            "error": detector_error or None,
            **face_counts,
        },
        "selection": selection["policy"],
        "key_frames": key_frames,
        "primary_frame": key_frames[0],
        "candidates": selection["candidates"],
        "warnings": warnings,
        "limitations": list(LIMITATIONS),
    }
    _write_json(selection_path, payload)
    emit(
        "video_frames_ready",
        f"已选出 {len(key_frames)} 个关键帧。",
        status="success",
        details={
            "key_frames": [item["file_name"] for item in key_frames],
            "timestamps": [item["timestamp_seconds"] for item in key_frames],
        },
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="视频抽帧与关键帧选择（v2）")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--output-dir", required=True, help="关键帧输出目录")
    parser.add_argument("--selection", required=True, help="抽帧证据 JSON 输出路径")
    parser.add_argument("--candidate-frames", type=int, default=24, help="候选帧数量（默认 24）")
    parser.add_argument("--max-key-frames", type=int, default=5, help="关键帧数量上限（默认 5）")
    parser.add_argument(
        "--face-detector",
        choices=("insightface", "none"),
        default="insightface",
        help="人脸检测后端；none 时仅按清晰度/亮度选帧",
    )
    parser.add_argument("--det-size", type=int, default=640, help="人脸检测输入尺度（默认 640）")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--ctx", type=int, default=-1)
    parser.add_argument(
        "--model-root",
        default=str(PRODUCT_ROOT / "packaging_v0.3" / "resources"),
        help="InsightFace 模型根目录（离线预检用）",
    )
    parser.add_argument("--min-sharpness-ratio", type=float, default=0.35)
    parser.add_argument("--prefer-single-face", dest="prefer_single_face", action="store_true", default=True)
    parser.add_argument("--no-prefer-single-face", dest="prefer_single_face", action="store_false")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--progress-events", default=None)
    parser.add_argument("--progress-latest", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = run(args)
    except FrameSelectionError as exc:
        payload = {
            "schema_version": "3.0",
            "module": "video",
            "status": "error",
            "video_path": str(Path(args.video).expanduser()),
            "key_frames": [],
            "error_message": str(exc),
            "limitations": list(LIMITATIONS),
        }
        _write_json(Path(args.selection).expanduser().resolve(strict=False), payload)
        print(f"抽帧失败：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - 结构化失败优于堆栈
        payload = {
            "schema_version": "3.0",
            "module": "video",
            "status": "error",
            "video_path": str(Path(args.video).expanduser()),
            "key_frames": [],
            "error_message": f"{type(exc).__name__}: {exc}",
            "limitations": list(LIMITATIONS),
        }
        _write_json(Path(args.selection).expanduser().resolve(strict=False), payload)
        print(f"抽帧异常：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": payload["status"],
                "key_frames": len(payload.get("key_frames") or []),
                "runtime_seconds": round(float(payload.get("runtime_seconds") or 0.0), 3),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
