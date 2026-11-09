from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = ROOT / "modules" / "identity_search_v2"
for entry in (str(ROOT), str(WORKER_DIR), str(ROOT / "modules" / "identity_search")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from identity_search.errors import (  # noqa: E402
    InputValidationError,
    MultipleFacesError,
    NoFaceError,
)
from run_identity_search_v2 import (  # noqa: E402
    DEFAULT_SCALE_CHAIN,
    FALLBACK_POLICY,
    attempt_scale_chain,
    build_parser,
    parse_scale_chain,
)
from scale_fallback import apply_detection_scale  # noqa: E402


class TestDetectionScaleSessionReuse:
    @staticmethod
    def make_extractor(*, static_size=None, fallback=False):
        calls = {"prepare": 0, "set_providers": 0}

        def prepare(*args, **kwargs):
            calls["prepare"] += 1
            detector.session.set_providers(["CPUExecutionProvider"])

        def set_providers(*args, **kwargs):
            calls["set_providers"] += 1

        detector = SimpleNamespace(
            static_input_size=static_size, input_size=(1280, 1280),
            input_sizes=[(1280, 1280)],
            session=SimpleNamespace(set_providers=set_providers),
        )
        app = SimpleNamespace(
            prepare=prepare, det_size=(1280, 1280), models={"detection": detector}
        )
        if not fallback:
            app.det_model = detector
        return SimpleNamespace(app=app, det_size=(1280, 1280)), detector, calls

    @pytest.mark.parametrize("fallback", [False, True])
    def test_scale_chain_never_reinitializes_sessions(self, fallback):
        extractor, detector, calls = self.make_extractor(fallback=fallback)
        for scale in (1280, 640, 320, 1280):
            assert apply_detection_scale(extractor, scale, -1) is extractor
            assert detector.input_sizes == [(scale, scale)]
            assert detector.input_size == (scale, scale)
            assert extractor.app.det_size == extractor.det_size == (scale, scale)
        assert calls == {"prepare": 0, "set_providers": 0}

    def test_same_scale_preserves_original_input_list(self):
        extractor, detector, calls = self.make_extractor()
        original_sizes = detector.input_sizes
        apply_detection_scale(extractor, 1280, -1)
        assert detector.input_sizes is original_sizes
        assert calls == {"prepare": 0, "set_providers": 0}

    def test_static_input_model_is_unchanged(self):
        extractor, detector, calls = self.make_extractor(static_size=(1280, 1280))
        original_sizes = detector.input_sizes
        apply_detection_scale(extractor, 640, -1)
        assert detector.input_sizes is original_sizes
        assert detector.input_size == extractor.app.det_size == extractor.det_size == (1280, 1280)
        assert calls == {"prepare": 0, "set_providers": 0}

    def test_missing_detector_is_rejected(self):
        with pytest.raises(InputValidationError, match="检测模型"):
            apply_detection_scale(SimpleNamespace(app=SimpleNamespace()), 640, -1)


class _FakeSearcher:
    """按注入的 extractor 尺度返回不同结果，用于验证回退策略而不加载模型。"""

    def __init__(self, behavior) -> None:
        self.behavior = behavior
        self.extractor = None
        self.scales_tried: list[int] = []
        self.options: list[dict] = []

    def search(self, query, top_k=5, progress_reporter=None, **kwargs):
        scale = self.extractor.scale
        self.scales_tried.append(scale)
        self.options.append({"query": query, "top_k": top_k})
        outcome = self.behavior(scale)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _extractor_factory(scale: int) -> SimpleNamespace:
    return SimpleNamespace(scale=scale)


class TestParseScaleChain:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("1280,640,320", (1280, 640, 320)),
            ("1280, 640", (1280, 640)),
            ("1280;640", (1280, 640)),
            ("640,640,1280", (640, 1280)),
            ("320", (320,)),
        ],
    )
    def test_valid_chains(self, text: str, expected: tuple[int, ...]) -> None:
        assert parse_scale_chain(text) == expected

    @pytest.mark.parametrize("text", ["", "  ", ",,,", "abc", "1280,abc", "32"])
    def test_invalid_chains_raise(self, text: str) -> None:
        with pytest.raises(InputValidationError):
            parse_scale_chain(text)


class TestAttemptScaleChain:
    def test_falls_back_only_when_no_face(self) -> None:
        searcher = _FakeSearcher(
            lambda scale: NoFaceError("未检出人脸", diagnostics={"face_count": 0})
            if scale == 1280
            else {"status": "success", "candidates": [{"identity_id": "person_008"}]}
        )
        result, attempts = attempt_scale_chain(
            searcher, "query.jpg", (1280, 640, 320), _extractor_factory, top_k=5
        )
        assert result is not None
        assert searcher.scales_tried == [1280, 640]
        assert [item["status"] for item in attempts] == ["error", "success"]
        assert attempts[0]["error_type"] == "NoFaceError"
        assert attempts[0]["face_count"] == 0
        assert attempts[-1]["scale"] == 640

    def test_primary_scale_keeps_frozen_behaviour(self) -> None:
        searcher = _FakeSearcher(lambda scale: {"status": "success"})
        result, attempts = attempt_scale_chain(
            searcher, "query.jpg", DEFAULT_SCALE_CHAIN, _extractor_factory, top_k=5
        )
        assert result == {"status": "success"}
        assert searcher.scales_tried == [1280]
        assert len(attempts) == 1

    def test_all_tiers_without_face_report_every_attempt(self) -> None:
        searcher = _FakeSearcher(
            lambda scale: NoFaceError("未检出人脸", diagnostics={"face_count": 0})
        )
        result, attempts = attempt_scale_chain(
            searcher, "query.jpg", (1280, 640), _extractor_factory, top_k=5
        )
        assert result is None
        assert searcher.scales_tried == [1280, 640]
        assert [item["scale"] for item in attempts] == [1280, 640]
        assert all(item["status"] == "error" for item in attempts)

    def test_multiple_faces_does_not_trigger_fallback(self) -> None:
        searcher = _FakeSearcher(
            lambda scale: MultipleFacesError(
                "检测到 2 张人脸", diagnostics={"face_count": 2}
            )
        )
        with pytest.raises(MultipleFacesError):
            attempt_scale_chain(
                searcher, "query.jpg", (1280, 640), _extractor_factory, top_k=5
            )
        assert searcher.scales_tried == [1280]

    def test_top_k_is_passed_through(self) -> None:
        searcher = _FakeSearcher(lambda scale: {"status": "success"})
        attempt_scale_chain(searcher, "query.jpg", (640,), _extractor_factory, top_k=3)
        assert searcher.options == [{"query": "query.jpg", "top_k": 3}]


class TestParserContract:
    def test_default_chain_matches_frozen_parity(self) -> None:
        args = build_parser().parse_args(
            ["--query", "q.jpg", "--library", "lib", "--output", "out.json"]
        )
        assert parse_scale_chain(args.det_size_chain) == DEFAULT_SCALE_CHAIN
        assert args.det_size_chain.split(",")[0] == "1280"

    @pytest.mark.parametrize(
        "option",
        [
            "--query",
            "--library",
            "--top-k",
            "--output",
            "--device",
            "--ctx",
            "--cache-dir",
            "--model-root",
            "--offline",
            "--task-id",
            "--progress-events",
            "--progress-latest",
        ],
    )
    def test_worker_keeps_frozen_cli_surface(self, option: str) -> None:
        actions = {action.option_strings[0] for action in build_parser()._actions}
        assert option in actions

    def test_policy_name_is_recorded(self) -> None:
        assert FALLBACK_POLICY == "scale_fallback_on_no_face"
