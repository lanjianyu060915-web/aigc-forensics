"""通用 AIGC 分支的分档和结果结构；不依赖 torch，能在界面环境里跑。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules" / "general_aigc"))

from general_aigc_core import (  # noqa: E402
    band_of,
    error_payload,
    face_branch_payload,
    input_kind,
    load_calibration,
    success_payload,
)


@dataclass
class _Result:
    formula = "公式"
    kind: str = "video"
    logit: float = 2.0
    clip_logit: float = 2.5
    clip_original_logit: float = 2.0
    clip_refit_logit: float | None = 3.0
    community_logit: float = 0.0
    frame_logits: tuple = ((1.0, 2.0, 3.0, 4.0, 5.0), (0.0, 0.0, 0.0, 0.0, 0.0))
    frame_timestamps: tuple = (0.0, 1.0)
    community_frame_logits: tuple = (0.5, -0.5)
    community_timestamps: tuple = (0.0, 1.0)
    width: int = 640
    height: int = 360
    duration_seconds: float | None = 2.0
    decode_mode: str = "strict"
    decode_seconds: float = 0.1
    inference_seconds: float = 0.2
    frame_batch_size: int = 16
    warnings: tuple = ()


THRESHOLDS = {"low_max": 0.5, "medium_max": 1.5, "calibration_set": {"n_real": 100}}


def test_band_boundaries() -> None:
    assert band_of(0.49, THRESHOLDS)[0] == "low"
    assert band_of(0.5, THRESHOLDS)[0] == "medium"
    assert band_of(1.49, THRESHOLDS)[0] == "medium"
    assert band_of(1.5, THRESHOLDS)[0] == "high"


def test_missing_thresholds_do_not_invent_a_band() -> None:
    band, label, reason = band_of(9.0, None)
    assert band is None
    assert label == "未标定"
    assert "校准" in reason


def test_input_kind_uses_suffix() -> None:
    assert input_kind(Path("a.JPG")) == "image"
    assert input_kind(Path("a.webp")) == "image"
    assert input_kind(Path("a.mp4")) == "video"
    assert input_kind(Path("query_video.mkv")) == "video"


def test_load_calibration_tolerates_missing_file(tmp_path: Path) -> None:
    assert load_calibration(tmp_path / "none.json") == {}
    assert load_calibration(None) == {}


def test_success_payload_keeps_frame_details() -> None:
    payload = success_payload(
        Path("q.mp4"),
        _Result(),
        score=0.88,
        model={"model_id": "m"},
        calibration={"video": THRESHOLDS},
        device="cuda",
        device_name="GPU",
        started=time.perf_counter(),
    )
    assert payload["status"] == "success"
    assert payload["band"] == "high"
    assert payload["band_thresholds"]["low_max"] == 0.5
    assert payload["frames"][0]["clip_logit"] == 3.0
    assert payload["frames"][0]["crop_logits"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert payload["community_frames"][1]["logit"] == -0.5
    assert payload["media"]["decode_mode"] == "strict"
    assert payload["error_message"] is None


def test_kind_without_calibration_warns() -> None:
    payload = success_payload(
        Path("q.jpg"),
        _Result(kind="image"),
        score=0.88,
        model={},
        calibration={"video": THRESHOLDS},
        device="cpu",
        device_name="CPU",
        started=time.perf_counter(),
    )
    assert payload["band"] is None
    assert payload["band_thresholds"] is None
    assert any("校准" in text for text in payload["warnings"])


def test_error_payload_shape() -> None:
    payload = error_payload(Path("q.mp4"), "坏文件", started=time.perf_counter(), error_type="decode_error")
    assert payload["status"] == "error"
    assert payload["score"] is None
    assert payload["error_message"] == "坏文件"
    assert payload["details"]["error_type"] == "decode_error"


def test_incomplete_face_thresholds_do_not_raise() -> None:
    payload = face_branch_payload([-1.0], ["a.jpg"], {"face_crop": {"low_max": -5.0}})
    assert payload["status"] == "success"
    assert payload["band"] is None
    assert payload["band_thresholds"] is None


def test_worker_survives_a_request_that_raises(monkeypatch) -> None:
    import io
    import json

    from modules.general_aigc import persistent_worker

    class _Runtime:
        def health_details(self):
            return {}

        def run(self, *_args):
            raise KeyError("坏配置")

    request = {"protocol": persistent_worker.PROTOCOL_NAME, "type": "request", "request_id": "r1", "action": "run"}
    shutdown = {**request, "request_id": "r2", "action": "shutdown"}
    output = io.StringIO()
    monkeypatch.setattr(persistent_worker, "PROTOCOL_OUTPUT", output)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request) + "\n" + json.dumps(shutdown) + "\n"))
    assert persistent_worker._serve(_Runtime()) == 0
    messages = [json.loads(line) for line in output.getvalue().splitlines() if line.strip()]
    failed = next(item for item in messages if item["request_id"] == "r1")
    assert failed["ok"] is False
    assert "坏配置" in failed["error"]
    assert any(item["request_id"] == "r2" for item in messages)
