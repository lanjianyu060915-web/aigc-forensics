"""常驻 worker v2（查询侧检测尺度回退）与配置透传的回归测试。

这里不加载 InsightFace：常驻运行时的模型初始化被替换成轻量替身，
只验证「协议字段、回退判定、证据写入、配置透传」这四件事。
真实模型路径由 experiments/m2_preprocess/probe_persistent_worker_v2.py 覆盖。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = ROOT / "modules" / "identity_search_v2"
for entry in (str(ROOT), str(WORKER_DIR), str(ROOT / "modules" / "identity_search")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import orchestrator  # noqa: E402
import persistent_worker as frozen_worker  # noqa: E402
import persistent_worker_v2  # noqa: E402
import runtime_services  # noqa: E402
from identity_search.errors import MultipleFacesError, NoFaceError  # noqa: E402
from run_identity_search_v2 import build_parser as build_v2_parser  # noqa: E402


SUCCESS_PAYLOAD = {
    "schema_version": "3.0",
    "status": "success",
    "query_image": "query.jpg",
    "candidates": [{"identity_id": "person_008", "mean_similarity": 0.13}],
    "metrics": {"query_face_count": 1},
    "method": {"name": "insightface_closed_set_person_mean_cosine"},
    "runtime": {"seconds": 0.0},
}


class _FakeSearcher:
    """按当前注入的检测尺度返回预置结果，不加载任何模型。"""

    def __init__(self, behavior) -> None:
        self.behavior = behavior
        self.extractor = None
        self.library_path = "fake-library"
        self._fingerprint = "fake-fingerprint"
        self.model_fingerprint = "fake-model"
        self.scales_tried: list[int] = []

    def search(self, query, top_k=5, progress_reporter=None, **kwargs):
        scale = int(self.extractor.det_size[0])
        self.scales_tried.append(scale)
        outcome = self.behavior(scale)
        if isinstance(outcome, Exception):
            raise outcome
        # 深拷贝：annotate_result 会就地写入 method/metrics，避免跨用例串味。
        return json.loads(json.dumps(outcome))


def _no_face() -> NoFaceError:
    return NoFaceError("图片中未检测到人脸", diagnostics={"face_count": 0})


def _make_runtime(monkeypatch, tmp_path, behavior, scales=(1280, 640, 320)):
    calls: list[tuple] = []

    def fake_apply(extractor, scale, ctx_id):
        calls.append(("scale", int(scale)))
        extractor.det_size = (int(scale), int(scale))
        return extractor

    def fake_init(self, args):
        self.cache_dir = tmp_path / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device, self.ctx_id = "cpu", -1
        self.model = SimpleNamespace(root=tmp_path / "models", fingerprint="fp-test")
        self._searchers = {}
        self._extractor = SimpleNamespace(det_size=None)

    monkeypatch.setattr(frozen_worker.PersistentIdentityRuntime, "__init__", fake_init)
    monkeypatch.setattr(persistent_worker_v2, "apply_detection_scale", fake_apply)
    monkeypatch.setattr(
        persistent_worker_v2, "create_progress_writer", lambda **kwargs: None
    )

    library_dir = tmp_path / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    query = tmp_path / "query.jpg"
    query.write_bytes(b"fixture")
    output = tmp_path / "identity_result.json"

    searcher = _FakeSearcher(behavior)

    def fake_searcher_for(library_value, progress_reporter=None):
        calls.append(("searcher", str(library_value)))
        return searcher

    args = persistent_worker_v2._parser().parse_args(
        [
            "--library",
            str(library_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--model-root",
            str(tmp_path / "models"),
        ]
    )
    runtime = persistent_worker_v2.ScaleFallbackIdentityRuntime(args, tuple(scales))
    monkeypatch.setattr(runtime, "_searcher_for", fake_searcher_for)
    return runtime, searcher, query, output, library_dir, calls


class TestParserAndHealth:
    def test_parser_defaults_to_calibrated_chain(self) -> None:
        args = persistent_worker_v2._parser().parse_args(
            ["--library", "l", "--cache-dir", "c"]
        )
        assert args.det_size_chain == "1280,640,320"

    def test_parser_accepts_explicit_chain(self) -> None:
        args = persistent_worker_v2._parser().parse_args(
            ["--library", "l", "--cache-dir", "c", "--det-size-chain", "960,640"]
        )
        assert args.det_size_chain == "960,640"

    def test_frozen_worker_has_no_scale_chain_option(self) -> None:
        """尺度链是 v2 专属参数；冻结 worker 收到它会直接报错退出。"""

        frozen_options = {
            option
            for action in frozen_worker._parser()._actions
            for option in action.option_strings
        }
        assert "--det-size-chain" not in frozen_options
        v2_options = {
            option for action in build_v2_parser()._actions for option in action.option_strings
        }
        assert "--det-size-chain" in v2_options

    def test_health_details_reports_v2_fields(self, monkeypatch, tmp_path) -> None:
        runtime, _searcher, _query, _output, _library, calls = _make_runtime(
            monkeypatch, tmp_path, lambda scale: dict(SUCCESS_PAYLOAD)
        )
        details = runtime.health_details()
        assert details["worker"] == "identity_v2"
        assert details["detection_policy"] == "scale_fallback_on_no_face"
        assert details["det_size_chain"] == [1280, 640, 320]
        assert list(details["detection_scale_current"]) == [1280, 1280]
        # 构造即回到主尺度，保证库索引与冻结口径一致。
        assert calls == [("scale", 1280)]
        assert details["model"] == "buffalo_l"
        assert details["model_instances"] == 1


class TestRunScales:
    def test_primary_scale_is_restored_before_library_lookup(
        self, monkeypatch, tmp_path
    ) -> None:
        runtime, _searcher, query, output, library_dir, calls = _make_runtime(
            monkeypatch, tmp_path, lambda scale: dict(SUCCESS_PAYLOAD)
        )
        runtime.run(str(query), str(output), str(library_dir), 5)
        # 取库之前必须有一次主尺度复位，否则新库索引会按查询侧的档位构建。
        searcher_index = next(
            index for index, call in enumerate(calls) if call[0] == "searcher"
        )
        assert calls[searcher_index - 1] == ("scale", 1280)
        assert calls[searcher_index + 1] == ("scale", 1280)

    def test_no_fallback_when_primary_scale_succeeds(self, monkeypatch, tmp_path) -> None:
        runtime, searcher, query, output, library_dir, _calls = _make_runtime(
            monkeypatch, tmp_path, lambda scale: dict(SUCCESS_PAYLOAD)
        )
        result = runtime.run(str(query), str(output), str(library_dir), 5)
        assert searcher.scales_tried == [1280]
        block = result["method"]["preprocessing"]
        assert block["detection_scale_used"] == 1280
        assert block["fallback_applied"] is False
        assert block["library_det_size"] == 1280
        assert [item["status"] for item in block["attempts"]] == ["success"]
        assert result["metrics"]["detection_scale_used"] == 1280
        assert result["metrics"]["detection_attempt_count"] == 1

    def test_fallback_on_no_face_annotates_every_evidence_layer(
        self, monkeypatch, tmp_path
    ) -> None:
        def behavior(scale: int):
            return _no_face() if scale == 1280 else dict(SUCCESS_PAYLOAD)

        runtime, searcher, query, output, library_dir, _calls = _make_runtime(
            monkeypatch, tmp_path, behavior
        )
        result = runtime.run(str(query), str(output), str(library_dir), 5)

        assert searcher.scales_tried == [1280, 640]
        assert result["status"] == "success"
        block = result["method"]["preprocessing"]
        assert block["policy"] == "scale_fallback_on_no_face"
        assert block["det_size_chain"] == [1280, 640, 320]
        assert block["library_det_size"] == 1280
        assert block["detection_scale_used"] == 640
        assert block["fallback_applied"] is True
        assert [item["scale"] for item in block["attempts"]] == [1280, 640]
        assert [item["status"] for item in block["attempts"]] == ["error", "success"]
        assert block["attempts"][0]["error_type"] == "NoFaceError"
        assert result["metrics"]["detection_scale_used"] == 640
        assert result["metrics"]["detection_attempt_count"] == 2

        on_disk = json.loads(output.read_text(encoding="utf-8"))
        assert on_disk["method"]["preprocessing"] == block
        assert on_disk["candidates"][0]["identity_id"] == "person_008"

    def test_all_scales_failing_keeps_attempt_evidence(self, monkeypatch, tmp_path) -> None:
        runtime, searcher, query, output, library_dir, _calls = _make_runtime(
            monkeypatch, tmp_path, lambda scale: _no_face()
        )
        result = runtime.run(str(query), str(output), str(library_dir), 5)

        assert searcher.scales_tried == [1280, 640, 320]
        assert result["status"] == "error"
        assert result["diagnostics"]["det_size_chain"] == [1280, 640, 320]
        assert len(result["diagnostics"]["attempts"]) == 3
        block = result["method"]["preprocessing"]
        assert block["detection_scale_used"] is None
        assert block["fallback_applied"] is False
        assert len(block["attempts"]) == 3
        assert json.loads(output.read_text(encoding="utf-8"))["status"] == "error"

    def test_multiple_faces_does_not_trigger_fallback(self, monkeypatch, tmp_path) -> None:
        def behavior(scale: int):
            return MultipleFacesError("检测到多张人脸", diagnostics={"face_count": 2})

        runtime, searcher, query, output, library_dir, _calls = _make_runtime(
            monkeypatch, tmp_path, behavior
        )
        result = runtime.run(str(query), str(output), str(library_dir), 5)

        assert searcher.scales_tried == [1280]
        assert result["status"] == "error"
        assert result["diagnostics"]["det_size_chain"] == [1280, 640, 320]
        assert result["diagnostics"]["attempts"] == []

    def test_invalid_top_k_is_rejected_before_any_detection(
        self, monkeypatch, tmp_path
    ) -> None:
        runtime, searcher, query, output, library_dir, _calls = _make_runtime(
            monkeypatch, tmp_path, lambda scale: dict(SUCCESS_PAYLOAD)
        )
        result = runtime.run(str(query), str(output), str(library_dir), 0)
        assert result["status"] == "error"
        assert "top-k" in result["error_message"]
        assert searcher.scales_tried == []


class TestScaleChainConfigPlumbing:
    def test_helper_ignores_absent_key(self) -> None:
        assert orchestrator._detection_scale_chain({}) is None
        assert runtime_services._detection_scale_chain({}) is None

    @pytest.mark.parametrize("bad", [[], "1280,640", [1280, "640"], [32], [True]])
    def test_helpers_reject_invalid_chains(self, bad) -> None:
        with pytest.raises(ValueError):
            orchestrator._detection_scale_chain({"det_size_chain": bad})
        with pytest.raises(runtime_services.RuntimeServiceUnavailable):
            runtime_services._detection_scale_chain({"det_size_chain": bad})

    def test_helpers_deduplicate_and_keep_order(self) -> None:
        payload = {"det_size_chain": [1280, 640, 1280, 320]}
        assert orchestrator._detection_scale_chain(payload) == [1280, 640, 320]
        assert runtime_services._detection_scale_chain(payload) == [1280, 640, 320]

    def test_persistent_worker_command_forwards_chain(self, tmp_path) -> None:
        worker = tmp_path / "persistent_worker_stub.py"
        worker.write_text("# stub\n", encoding="utf-8")
        library = tmp_path / "library"
        library.mkdir()
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "runtime": {"identity_cache": "runtime/cache/identity"},
                    "identity": {
                        "persistent_enabled": True,
                        "python": sys.executable,
                        "persistent_worker": str(worker),
                        "det_size_chain": [1280, 640, 320],
                        "library": str(library),
                        "model_root": str(tmp_path / "models"),
                        "ctx": -1,
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        manager = runtime_services.RuntimeServiceManager(config_path)
        try:
            spec = manager._worker_spec("identity")
        finally:
            manager.close()
        assert spec is not None
        command = spec[0]
        assert "--det-size-chain" in command
        assert command[command.index("--det-size-chain") + 1] == "1280,640,320"
        assert command[command.index("--library") + 1] == str(library)

    def test_persistent_worker_command_omits_chain_when_unset(self, tmp_path) -> None:
        worker = tmp_path / "persistent_worker_stub.py"
        worker.write_text("# stub\n", encoding="utf-8")
        library = tmp_path / "library"
        library.mkdir()
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "runtime": {"identity_cache": "runtime/cache/identity"},
                    "identity": {
                        "persistent_enabled": True,
                        "python": sys.executable,
                        "persistent_worker": str(worker),
                        "library": str(library),
                        "model_root": str(tmp_path / "models"),
                        "ctx": -1,
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        manager = runtime_services.RuntimeServiceManager(config_path)
        try:
            spec = manager._worker_spec("identity")
        finally:
            manager.close()
        assert spec is not None
        assert "--det-size-chain" not in spec[0]


class TestWorkspaceConfigWiring:
    """真实 config.json 的回归护栏：身份模块必须走 v2，且指向真实存在的文件。"""

    @staticmethod
    def _section() -> dict:
        return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

    def test_identity_uses_v2_runner_and_worker(self) -> None:
        config = self._section()
        section = config["identity"]
        assert "executable" not in section
        assert "worker_executable" not in section
        assert config_path_is_v2(section["runner"])
        assert config_path_is_v2(section["persistent_worker"])
        assert section["det_size_chain"][0] == 1280
        assert config["upgrade"]["upgraded_kernels"]["identity"] == "identity_search_v2"

    def test_identity_paths_exist_on_disk(self) -> None:
        config_path = ROOT / "config.json"
        section = self._section()["identity"]
        for key in ("python", "runner", "persistent_worker"):
            resolved = orchestrator.resolve_path(section[key], config_path)
            assert resolved.is_file(), f"{key} 不存在：{resolved}"
        library = orchestrator.resolve_path(section["library"], config_path)
        assert library.is_dir()

    def test_orchestrator_command_shape_matches_v2_cli(self, tmp_path) -> None:
        """命令面必须是 v2 CLI 能解析的形态（含尺度链）。"""

        config_path = ROOT / "config.json"
        section = self._section()["identity"]
        chain = orchestrator._detection_scale_chain(section)
        assert chain == [1280, 640, 320]
        argparse_namespace = build_v2_parser().parse_args(
            [
                "--query",
                "q.jpg",
                "--library",
                "l",
                "--top-k",
                "5",
                "--output",
                "o.json",
                "--det-size-chain",
                ",".join(str(item) for item in chain),
            ]
        )
        assert argparse_namespace.det_size_chain == "1280,640,320"


def config_path_is_v2(value: str) -> bool:
    return "identity_search_v2" in value.replace("\\", "/")
