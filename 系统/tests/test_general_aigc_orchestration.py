"""编排层接通用 AIGC 分支：用假 runner 和假服务验证接线，不加载真实模型。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from orchestrator import _identity_not_applicable, _run_general_aigc, run_task


class _Progress:
    def __init__(self) -> None:
        self.events: list[str] = []

    def emit(self, stage, message, module=None, **_kwargs):
        self.events.append(stage)


GOOD_RUNNER = """
import argparse, json
parser = argparse.ArgumentParser()
parser.add_argument('--input'); parser.add_argument('--output')
parser.add_argument('--device'); parser.add_argument('--model-package', default=None)
parser.add_argument('--face-crops-json', default=None)
args = parser.parse_args()
payload = {'status': 'success', 'module': 'general_aigc', 'score': 0.8, 'logit': 1.4,
           'band': 'medium', 'band_label': '需人工复核', 'error_message': None,
           'runtime_seconds': 0.01, 'seen_device': args.device, 'seen_input': args.input}
if args.face_crops_json:
    crops = json.loads(open(args.face_crops_json, encoding='utf-8').read())
    payload['face_branch'] = {'status': 'success' if crops else 'no_face', 'logit': -4.0 if crops else None,
                              'band': 'high' if crops else None, 'band_label': '高风险提示' if crops else '未检出人脸',
                              'seen_crops': crops}
open(args.output, 'w', encoding='utf-8').write(json.dumps(payload, ensure_ascii=False))
"""


def _write_runner(path: Path, body: str = GOOD_RUNNER) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _section(runner: Path, **extra) -> dict:
    return {
        "python": sys.executable,
        "python_args": ["-B"],
        "runner": str(runner),
        "device": "cpu",
        **extra,
    }


def _call(tmp_path: Path, section: dict, services=None) -> tuple[dict, _Progress]:
    progress = _Progress()
    input_path = tmp_path / "query.jpg"
    input_path.write_bytes(b"x")
    payload = _run_general_aigc(
        section=section,
        config_path=tmp_path / "config.json",
        input_path=input_path,
        output_path=tmp_path / "general_aigc_result.json",
        logs_dir=tmp_path,
        progress=progress,  # type: ignore[arg-type]
        services=services,
    )
    return payload, progress


def test_subprocess_path_writes_result(tmp_path: Path) -> None:
    payload, progress = _call(tmp_path, _section(_write_runner(tmp_path / "runner.py")))
    assert payload["status"] == "success"
    assert payload["seen_device"] == "cpu"
    assert payload["runtime"]["execution_mode"] == "subprocess"
    saved = json.loads((tmp_path / "general_aigc_result.json").read_text(encoding="utf-8"))
    assert saved["logit"] == 1.4
    assert "general_aigc_started" in progress.events
    assert "general_aigc_finished" in progress.events


def test_service_is_preferred(tmp_path: Path) -> None:
    def service(*, input_path, output_path, timeout, face_crops=None):
        return {"status": "success", "logit": 2.0, "band": "high", "band_label": "高风险提示"}

    payload, _ = _call(
        tmp_path, _section(tmp_path / "does_not_exist.py"), services={"run_general_aigc": service}
    )
    assert payload["logit"] == 2.0
    assert payload["runtime"]["execution_mode"] == "service"


def test_service_failure_falls_back_and_records_reason(tmp_path: Path) -> None:
    def service(*, input_path, output_path, timeout, face_crops=None):
        raise RuntimeError("服务挂了")

    payload, progress = _call(
        tmp_path, _section(_write_runner(tmp_path / "runner.py")), services={"run_general_aigc": service}
    )
    assert payload["status"] == "success"
    assert "服务挂了" in payload["runtime"]["fallback_reason"]
    assert "general_aigc_service_fallback" in progress.events


def test_late_worker_write_cannot_replace_the_recorded_result(tmp_path: Path) -> None:
    seen = {}

    def service(*, input_path, output_path, timeout, face_crops=None):
        seen["output"] = Path(output_path)
        return {"status": "success", "logit": 2.0, "band": "high", "band_label": "高风险提示"}

    _call(tmp_path, _section(tmp_path / "x.py"), services={"run_general_aigc": service})
    assert seen["output"] != tmp_path / "general_aigc_result.json"
    assert (tmp_path / "general_aigc_result.json").is_file()


@pytest.mark.parametrize(
    ("reasons", "applicable"),
    [
        (["图片中未检测到人脸: a", "图片中未检测到人脸: b"], False),
        (["图片中未检测到人脸: a", "模型崩溃"], True),
    ],
)
def test_video_identity_is_not_applicable_only_when_every_frame_has_no_face(reasons, applicable) -> None:
    payload = {
        "status": "error",
        "error_message": "全部关键帧的身份检索均未成功，视频级候选不可用。",
        "details": {"frame_failures": [{"error_message": reason} for reason in reasons]},
    }
    assert (_identity_not_applicable(payload) is None) is applicable


def test_missing_python_is_a_result_not_an_exception(tmp_path: Path) -> None:
    section = {**_section(_write_runner(tmp_path / "runner.py")), "python": str(tmp_path / "nope.exe")}
    payload, _ = _call(tmp_path, section)
    assert payload["status"] == "error"
    assert "引擎 Python 不存在" in payload["error_message"]


def test_crashing_runner_is_a_result_not_an_exception(tmp_path: Path) -> None:
    payload, _ = _call(tmp_path, _section(_write_runner(tmp_path / "runner.py", "raise SystemExit(3)\n")))
    assert payload["status"] == "error"


@pytest.mark.parametrize("bad", [0, -1, "x", float("inf")])
def test_invalid_timeout_is_reported(tmp_path: Path, bad) -> None:
    payload, _ = _call(tmp_path, _section(_write_runner(tmp_path / "runner.py"), timeout_seconds=bad))
    assert payload["status"] == "error"


def _config(tmp_path: Path, *, general: dict | None) -> Path:
    identity_library = tmp_path / "identity_library"
    source_library = tmp_path / "source_library"
    (identity_library / "person_001").mkdir(parents=True)
    source_library.mkdir()
    (identity_library / "person_001" / "reference.jpg").write_bytes(b"fixture")
    (source_library / "source.jpg").write_bytes(b"fixture")
    config = {
        "schema_version": "test",
        "runtime": {
            "tasks_root": str(tmp_path / "tasks"),
            "inbox_root": str(tmp_path / "inbox"),
            "identity_cache": str(tmp_path / "cache"),
        },
        "deepfake": {"enabled": False, "timeout_seconds": 10},
        "identity": {
            "python": sys.executable,
            "runner": str(tmp_path / "x.py"),
            "library": str(identity_library),
            "timeout_seconds": 10,
            "model_root": str(tmp_path / "models"),
        },
        "source": {
            "python": sys.executable,
            "runner": str(tmp_path / "x.py"),
            "library": str(source_library),
            "top_k": 5,
            "timeout_seconds": 10,
        },
        "frontend": {"python": sys.executable},
    }
    if general is not None:
        config["general_aigc"] = general
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return path


def _query(tmp_path: Path) -> Path:
    query = tmp_path / "upload.jpg"
    query.write_bytes(b"fixture")
    return query


def test_task_merges_general_branch_when_legacy_branch_fails(tmp_path: Path) -> None:
    config = _config(tmp_path, general=_section(_write_runner(tmp_path / "runner.py")))
    status = run_task(_query(tmp_path), selected_modules={"deepfake"}, config_path=config)
    task_dir = Path(status["task_dir"])
    detection = json.loads((task_dir / "detection_result.json").read_text(encoding="utf-8"))
    assert status["status"] == "success"
    assert detection["general_aigc"]["logit"] == 1.4
    assert detection["overall"]["band"] == "medium"
    assert detection["legacy_error_message"]
    assert status["deepfake"]["general_aigc"]["status"] == "success"
    manifest = json.loads((task_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert "general_aigc_result.json" in {item["path"] for item in manifest["artifacts"]}


def test_task_without_general_section_keeps_old_shape(tmp_path: Path) -> None:
    config = _config(tmp_path, general=None)
    status = run_task(_query(tmp_path), selected_modules={"deepfake"}, config_path=config)
    detection = json.loads(
        (Path(status["task_dir"]) / "detection_result.json").read_text(encoding="utf-8")
    )
    assert "general_aigc" not in detection
    assert "overall" not in detection


def test_general_branch_is_skipped_when_deepfake_not_selected(tmp_path: Path) -> None:
    config = _config(tmp_path, general=_section(tmp_path / "must_not_run.py"))
    status = run_task(_query(tmp_path), selected_modules={"source"}, config_path=config)
    assert not (Path(status["task_dir"]) / "general_aigc_result.json").exists()


NO_FACE_RUNNER = """
import json, sys
output = sys.argv[sys.argv.index('--output') + 1]
payload = {'status': 'error', 'query_image': 'query.jpg', 'candidates': [], 'runtime_seconds': 0.1,
           'error_message': '图片中未检测到人脸: query.jpg'}
open(output, 'w', encoding='utf-8').write(json.dumps(payload, ensure_ascii=False))
raise SystemExit(1)
"""


def test_no_face_is_not_applicable_rather_than_a_failure(tmp_path: Path) -> None:
    config_path = _config(tmp_path, general=None)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["identity"]["runner"] = str(_write_runner(tmp_path / "no_face.py", NO_FACE_RUNNER))
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    status = run_task(_query(tmp_path), selected_modules={"identity"}, config_path=config_path)
    identity = json.loads((Path(status["task_dir"]) / "identity_result.json").read_text(encoding="utf-8"))
    assert identity["status"] == "skipped"
    assert identity["not_applicable"] is True
    assert identity["error_message"] is None
    assert "不适用" in identity["details"]["reason"]
    assert status["identity"]["not_applicable"] is True
    assert not status["errors"]


GOOD_CROPPER = """
import argparse, json, pathlib
parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', type=pathlib.Path); parser.add_argument('--image', default=None)
parser.add_argument('--frame-selection', default=None); parser.add_argument('--key-frames-dir', default=None)
parser.add_argument('--model-root', default=None)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
(args.out_dir / 'image_face.jpg').write_bytes(b'crop')
result = {'status': 'success', 'crops': [{'file': 'image_face.jpg', 'source': 'query.jpg', 'bbox': [1, 2, 3, 4]}]}
(args.out_dir / 'crops.json').write_text(json.dumps(result), encoding='utf-8')
"""


def _face_section(runner: Path) -> dict:
    return {
        "python": sys.executable,
        "runner": str(runner),
        "model_root": str(runner.parent),
        "timeout_seconds": 30,
    }


def test_subprocess_path_passes_face_crops_to_the_runner(tmp_path: Path) -> None:
    progress = _Progress()
    input_path = tmp_path / "query.jpg"
    input_path.write_bytes(b"x")
    payload = _run_general_aigc(
        section=_section(_write_runner(tmp_path / "runner.py")),
        config_path=tmp_path / "config.json",
        input_path=input_path,
        output_path=tmp_path / "general_aigc_result.json",
        logs_dir=tmp_path,
        progress=progress,  # type: ignore[arg-type]
        services=None,
        face_crops=["甲.jpg"],
    )
    assert payload["face_branch"]["seen_crops"] == ["甲.jpg"]


def test_task_runs_the_face_branch_and_keeps_its_crops(tmp_path: Path) -> None:
    general = _section(
        _write_runner(tmp_path / "runner.py"),
        face_crops=_face_section(_write_runner(tmp_path / "cropper.py", GOOD_CROPPER)),
    )
    status = run_task(_query(tmp_path), selected_modules={"deepfake"}, config_path=_config(tmp_path, general=general))
    task_dir = Path(status["task_dir"])
    detection = json.loads((task_dir / "detection_result.json").read_text(encoding="utf-8"))
    face = detection["general_aigc"]["face_branch"]
    assert face["seen_crops"] == [str(task_dir / "face_crops" / "image_face.jpg")]
    assert face["records"][0]["bbox"] == [1, 2, 3, 4]
    assert detection["overall"]["band"] == "high"
    manifest = json.loads((task_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert "face_crops/image_face.jpg" in {item["path"] for item in manifest["artifacts"]}


def test_face_crop_failure_only_drops_the_face_branch(tmp_path: Path) -> None:
    general = _section(
        _write_runner(tmp_path / "runner.py"),
        face_crops=_face_section(_write_runner(tmp_path / "cropper.py", "raise SystemExit(3)")),
    )
    status = run_task(_query(tmp_path), selected_modules={"deepfake"}, config_path=_config(tmp_path, general=general))
    task_dir = Path(status["task_dir"])
    detection = json.loads((task_dir / "detection_result.json").read_text(encoding="utf-8"))
    assert status["status"] == "success"
    assert "face_branch" not in detection["general_aigc"]
    assert detection["overall"]["band"] == "medium"
    assert [item["status"] for item in detection["overall"]["branches"]][1] == "skipped"
