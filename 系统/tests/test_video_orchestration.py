# -*- coding: utf-8 -*-
"""视频入口的编排层回归测试（不加载模型、不解码真实视频）。

抽帧模块替换为 fixture runner，鉴伪与身份用假服务，底图沿用既有 fixture runner，
因此这里只验证编排契约：视频识别、关键帧落盘、逐帧分发、聚合口径与证据清单。
真实视频端到端结果见 experiments/m4_video/results/。
"""

from __future__ import annotations

import json
import secrets
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

from orchestrator import OrchestrationError, run_task


FIXTURES = Path(__file__).resolve().parent / "fixtures"
TEMP_ROOT = Path(__file__).resolve().parent / ".run_tmp"


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    """避开 pytest 的用户临时目录（Windows 上可能被 ACL 限制）。"""

    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEMP_ROOT / f"test_{secrets.token_hex(8)}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class VideoServices:
    """假模型服务：前两帧给高风险分带，身份返回固定候选。"""

    def __init__(self) -> None:
        self.deepfake_frames: list[str] = []
        self.identity_frames: list[str] = []

    def run_deepfake(self, **kwargs: Any) -> dict[str, Any]:
        query = Path(kwargs["query_path"])
        self.deepfake_frames.append(query.name)
        high = query.name in {"frame_001.jpg", "frame_002.jpg"}
        return {
            "status": "success",
            "query_image": str(query),
            "is_deepfake": high,
            "score": 0.9 if high else 0.1,
            "label": "fake" if high else "real",
            "runtime_seconds": 0.01,
            "details": {
                "band": "high" if high else "low",
                "band_label": "高风险提示" if high else "证据不足",
                "routed_branch": "vit_full_image",
            },
            "limitations": ["帧级限制：模型输出不是鉴定结论。"],
            "error_message": None,
        }

    def run_identity(self, **kwargs: Any) -> dict[str, Any]:
        query = Path(kwargs["query_path"])
        self.identity_frames.append(query.name)
        return {
            "status": "success",
            "query_image": str(query),
            "candidates": [
                {
                    "rank": 1,
                    "identity_id": "person_001",
                    "mean_similarity": 0.5,
                    "reference_count": 5,
                }
            ],
            "metrics": {"query_face_count": 1, "top1_top2_margin": 0.2},
            "runtime_seconds": 0.01,
            "error_message": None,
        }


class FlakyVideoServices(VideoServices):
    """第 2 帧鉴伪抛异常，用于验证单帧失败不会终止视频任务。"""

    def run_deepfake(self, **kwargs: Any) -> dict[str, Any]:
        if Path(kwargs["query_path"]).name == "frame_002.jpg":
            raise RuntimeError("单帧失败 fixture")
        return super().run_deepfake(**kwargs)


def _write_config(tmp_path: Path, *, video_runner: Path | None = None) -> Path:
    identity_library = tmp_path / "identity_library"
    source_library = tmp_path / "source_library"
    (identity_library / "person_001").mkdir(parents=True)
    source_library.mkdir(parents=True)
    (identity_library / "person_001" / "reference.jpg").write_bytes(b"fixture")
    (source_library / "source.jpg").write_bytes(b"fixture")

    config = {
        "schema_version": "test",
        "runtime": {
            "tasks_root": str(tmp_path / "tasks"),
            "inbox_root": str(tmp_path / "inbox"),
            "identity_cache": str(tmp_path / "cache"),
        },
        "video": {
            "enabled": True,
            "python": sys.executable,
            "python_args": ["-B"],
            "runner": str(video_runner or (FIXTURES / "fake_video_runner.py")),
            "candidate_frames": 6,
            "max_key_frames": 3,
            "det_size": 320,
            "device": "cpu",
            "ctx": -1,
            "face_detector": "none",
            "prefer_single_face": True,
            "min_sharpness_ratio": 0.35,
            "frame_selection_timeout_seconds": 30,
            "deepfake_frames": 4,
            "identity_frames": 3,
            "source_frames": 1,
        },
        "deepfake": {"enabled": True, "timeout_seconds": 10},
        "identity": {
            "python": sys.executable,
            "runner": str(FIXTURES / "failing_identity_runner.py"),
            "library": str(identity_library),
            "top_k": 5,
            "timeout_seconds": 10,
            "ctx": -1,
            "model_root": str(tmp_path / "models"),
            "offline": True,
        },
        "source": {
            "python": sys.executable,
            "runner": str(FIXTURES / "successful_source_runner.py"),
            "library": str(source_library),
            "top_k": 5,
            "timeout_seconds": 10,
        },
        "frontend": {"python": sys.executable},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path


def test_video_task_runs_each_module_on_its_frame_plan(workspace_tmp_path: Path) -> None:
    config_path = _write_config(workspace_tmp_path)
    video = workspace_tmp_path / "query.mp4"
    video.write_bytes(b"fixture-video-bytes")
    services = VideoServices()
    progress: list[dict[str, Any]] = []

    status = run_task(
        video,
        config_path=config_path,
        services=services,
        progress_callback=progress.append,
        selected_modules=["deepfake", "identity", "source"],
    )
    task_dir = Path(status["task_dir"])

    assert status["status"] == "success"
    assert (task_dir / "query_video.mp4").is_file()
    assert (task_dir / "query.jpg").is_file()
    assert sorted(item.name for item in (task_dir / "key_frames").glob("*.jpg")) == [
        "frame_001.jpg",
        "frame_002.jpg",
        "frame_003.jpg",
    ]
    # 鉴伪与身份各跑 3 个关键帧；底图只跑首要帧。
    assert services.deepfake_frames == ["frame_001.jpg", "frame_002.jpg", "frame_003.jpg"]
    assert services.identity_frames == ["frame_001.jpg", "frame_002.jpg", "frame_003.jpg"]
    for rank in (1, 2, 3):
        assert (task_dir / "frame_results" / "deepfake" / f"frame_{rank:03d}.json").is_file()
        assert (task_dir / "frame_results" / "identity" / f"frame_{rank:03d}.json").is_file()
    assert (task_dir / "frame_results" / "source" / "frame_001.json").is_file()
    assert not (task_dir / "frame_results" / "source" / "frame_002.json").exists()

    detection = json.loads((task_dir / "detection_result.json").read_text(encoding="utf-8"))
    details = detection["details"]
    assert detection["status"] == "success"
    assert detection["is_deepfake"] is True
    assert details["video_band_rule"] == "band_majority"
    assert details["band"] == "high"
    assert details["band_rule"] == "calibrated_band"
    assert details["frames_scored"] == 3
    assert details["frame_band_sources"]["calibrated_band"] == 3
    assert details["per_frame"][0]["branch"] == "vit_full_image"

    identity = json.loads((task_dir / "identity_result.json").read_text(encoding="utf-8"))
    assert identity["details"]["video_aggregation"]["rule"] == "mean_across_frames"
    assert identity["metrics"]["frames_with_faces"] == 3
    assert identity["candidates"][0]["frame_support"] == 3

    source = json.loads((task_dir / "source_result.json").read_text(encoding="utf-8"))
    source_aggregation = source["details"]["video_aggregation"]
    assert source_aggregation["rule"] == "primary_frame"
    assert source_aggregation["frame"]["file_name"] == "frame_001.jpg"

    unified = json.loads((task_dir / "unified_result.json").read_text(encoding="utf-8"))
    assert unified["input"]["type"] == "video"
    assert unified["input"]["video_file"] == "query_video.mp4"
    assert len(unified["input"]["key_frames"]) == 3

    summary = json.loads((task_dir / "video_summary.json").read_text(encoding="utf-8"))
    assert summary["input_type"] == "video"
    assert summary["video_sha256"] == "c" * 64
    assert summary["modules"]["deepfake"]["band"] == "high"
    assert summary["modules"]["identity"]["top1_frame_support"] == 3

    manifest = json.loads((task_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    paths = {item["path"] for item in manifest["artifacts"]}
    assert {"query_video.mp4", "query.jpg", "frame_selection.json", "video_summary.json"} <= paths
    assert "key_frames/frame_001.jpg" in paths

    stages = [event["stage"] for event in progress]
    assert "video_frame_selection_started" in stages
    assert "video_frames_selected" in stages
    assert "deepfake_frame_finished" in stages


def test_single_frame_failure_does_not_abort_video_task(workspace_tmp_path: Path) -> None:
    config_path = _write_config(workspace_tmp_path)
    video = workspace_tmp_path / "query.mp4"
    video.write_bytes(b"fixture-video-bytes")

    status = run_task(
        video,
        config_path=config_path,
        services=FlakyVideoServices(),
        selected_modules=["deepfake"],
    )
    task_dir = Path(status["task_dir"])
    detection = json.loads((task_dir / "detection_result.json").read_text(encoding="utf-8"))

    assert status["status"] == "success"
    assert detection["status"] == "success"
    assert detection["details"]["frames_scored"] == 2
    assert detection["details"]["frames_total"] == 3
    assert detection["details"]["frame_failures"][0]["file_name"] == "frame_002.jpg"
    assert any("未取得有效鉴伪结果" in item for item in detection["warnings"])


def test_video_extraction_failure_is_a_controlled_error(workspace_tmp_path: Path) -> None:
    config_path = _write_config(
        workspace_tmp_path, video_runner=FIXTURES / "failing_video_runner.py"
    )
    video = workspace_tmp_path / "query.mp4"
    video.write_bytes(b"fixture-video-bytes")

    with pytest.raises(OrchestrationError) as excinfo:
        run_task(video, config_path=config_path, selected_modules=["deepfake"])

    assert "视频抽帧失败" in str(excinfo.value)
    tasks_root = workspace_tmp_path / "tasks"
    task_dirs = [item for item in tasks_root.iterdir() if item.is_dir()]
    assert len(task_dirs) == 1
    status = json.loads((task_dirs[0] / "task_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "error"
    assert status["input"]["type"] == "video"
    assert "fixture：视频无法解码" in status["input"]["error"]


def test_image_query_still_uses_single_image_path(workspace_tmp_path: Path) -> None:
    config_path = _write_config(workspace_tmp_path)
    query = workspace_tmp_path / "query.jpg"
    query.write_bytes(b"fixture-image-bytes")

    status = run_task(
        query,
        config_path=config_path,
        services=VideoServices(),
        selected_modules=["deepfake", "source"],
    )
    task_dir = Path(status["task_dir"])
    unified = json.loads((task_dir / "unified_result.json").read_text(encoding="utf-8"))

    assert unified["input"]["type"] == "image"
    assert not (task_dir / "frame_selection.json").exists()
    assert not (task_dir / "video_summary.json").exists()
    assert not (task_dir / "key_frames").exists()
    assert (task_dir / "query.jpg").is_file()
