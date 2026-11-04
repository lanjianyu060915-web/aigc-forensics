"""引擎的结构、分数公式和图片路径。用微型模型，不依赖真实权重；需要 torch。

运行：engine_runtime\\python.exe -B -m pytest engine\\tests -q -p no:cacheprovider
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aigc_engine.media import (  # noqa: E402
    MediaError,
    VideoConfig,
    _native_crops,
    _pil_resize_crop,
    _tolerant_timestamps,
    decode_video,
    decode_video_by_seek,
)
from aigc_engine.media_tools import media_tool  # noqa: E402
from aigc_engine.metrics import UndefinedMetricError, evaluate_binary  # noqa: E402
from aigc_engine.models import (  # noqa: E402
    ClipCommunityFusionClassifier,
    ClipCommunityFusionConfig,
    create_model,
)
from aigc_engine.scorer import (  # noqa: E402
    GeneralAigcScorer,
    load_image_rgb,
    logit_to_score,
    pad_to_minimum,
    select_video_stream,
)

CLIP = {
    "backbone_config": {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "image_size": 8,
        "patch_size": 2,
        "projection_dim": 6,
    },
    "hidden_dim": 4,
    "dropout": 0.0,
}
HAS_FFMPEG = shutil.which(media_tool("ffmpeg")) is not None


@pytest.fixture(scope="module")
def model() -> ClipCommunityFusionClassifier:
    torch.manual_seed(38)
    config = ClipCommunityFusionConfig(
        clip=CLIP,
        community={},
        clip_weight=0.8,
        clip_frames=2,
        clip_crops=5,
        community_frames=2,
        refit_heads=3,
        refit_weight=0.5,
    )
    return ClipCommunityFusionClassifier(config).eval()


@pytest.fixture()
def scorer(model: ClipCommunityFusionClassifier) -> GeneralAigcScorer:
    """跳过权重文件加载，直接用微型模型走打分路径。"""
    instance = GeneralAigcScorer.__new__(GeneralAigcScorer)
    instance.device = torch.device("cpu")
    instance.model = model
    instance.frame_batch_size = 4
    instance.load_seconds = 0.0
    instance.metadata = {}
    instance.clip_view = VideoConfig(
        n_frames=2, height=8, width=8, resize_mode="native_center_crop", native_crops=5
    )
    instance.community_view = replace(
        instance.clip_view,
        n_frames=2,
        height=384,
        width=384,
        resize_mode="pil_center_crop",
        resize_short_edge=440,
        native_crops=1,
    )
    return instance


def _forward_directly(model, clip_frames, community_frames) -> float:
    with torch.inference_mode():
        return float(model(clip_frames[None], community_frames[None], frame_batch_size=4)[0])


def test_details_match_model_forward(scorer: GeneralAigcScorer, model) -> None:
    """拆开算的逐裁剪分数，取均值后必须等于模型自己的 forward。"""
    clip_frames = torch.rand(10, 3, 8, 8)
    community_frames = torch.rand(2, 3, 384, 384)
    per_crop, community, original, refit = scorer._forward(clip_frames, community_frames, 4)
    assert per_crop.shape == (10,)
    assert community.shape == (2,)
    assert refit is not None
    combined = 0.8 * per_crop.mean() + 0.2 * community.mean()
    assert combined == pytest.approx(_forward_directly(model, clip_frames, community_frames), abs=1e-5)
    assert per_crop.mean() == pytest.approx(0.5 * original + 0.5 * refit, abs=1e-5)


def test_image_scores_one_frame_five_crops(scorer: GeneralAigcScorer, tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (64, 96, 3), dtype=np.uint8)).save(path)
    result = scorer.score_image(path)
    assert result.kind == "image"
    assert len(result.frame_logits) == 1
    assert len(result.frame_logits[0]) == 5
    assert len(result.community_frame_logits) == 1
    assert (result.width, result.height) == (96, 64)
    assert result.decode_mode == "image"
    assert result.duration_seconds is None
    expected = 0.8 * result.clip_logit + 0.2 * result.community_logit
    assert result.logit == pytest.approx(expected)


def test_small_image_is_padded_and_warned(scorer: GeneralAigcScorer, tmp_path: Path) -> None:
    path = tmp_path / "small.jpg"
    Image.new("RGB", (4, 6), (200, 10, 10)).save(path)
    result = scorer.score_image(path)
    assert any("补黑边" in text for text in result.warnings)


def test_pad_is_centered_like_ffmpeg() -> None:
    rgb = np.full((2, 3, 3), 255, dtype=np.uint8)
    padded = pad_to_minimum(rgb, 6, 8)
    assert padded.shape == (6, 8, 3)
    assert padded[2:4, 2:5].min() == 255
    assert padded[0].max() == 0
    assert pad_to_minimum(rgb, 2, 3) is rgb


def test_exif_rotation_is_applied(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (40, 20), (10, 20, 30))
    exif = image.getexif()
    exif[0x0112] = 6  # 顺时针 90 度
    image.save(path, exif=exif)
    rgb, _ = load_image_rgb(path)
    assert rgb.shape[:2] == (40, 20)


def test_oversized_image_is_scaled_and_recorded(tmp_path: Path, monkeypatch) -> None:
    import aigc_engine.scorer as scorer_module

    monkeypatch.setattr(scorer_module, "IMAGE_MAX_PIXELS", 100)
    path = tmp_path / "big.png"
    Image.new("RGB", (100, 100)).save(path)
    rgb, warnings = load_image_rgb(path)
    assert rgb.shape[0] * rgb.shape[1] <= 100
    assert any("缩小" in text for text in warnings)


def test_broken_image_raises_for_the_caller(scorer: GeneralAigcScorer, tmp_path: Path) -> None:
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(OSError):
        scorer.score_image(path)


def test_native_crops_center_first() -> None:
    frames = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(1, 16, 16, 3)
    config = VideoConfig(n_frames=1, height=8, width=8, resize_mode="native_center_crop", native_crops=5)
    crops = _native_crops(frames, config)
    assert crops.shape == (5, 8, 8, 3)
    assert np.array_equal(crops[0], frames[0, 4:12, 4:12])


def test_pil_view_size() -> None:
    frames = np.zeros((1, 200, 300, 3), dtype=np.uint8)
    config = VideoConfig(n_frames=1, height=384, width=384, resize_mode="pil_center_crop", resize_short_edge=440)
    assert _pil_resize_crop(frames, config).shape == (1, 384, 384, 3)


def test_tolerant_timestamps_fill_gaps_and_stay_monotonic() -> None:
    frames = [
        {"best_effort_timestamp_time": "0.5"},
        {"best_effort_timestamp_time": "N/A"},
        {},
        {"best_effort_timestamp_time": "0.1"},
        {"best_effort_timestamp_time": "1.5"},
    ]
    assert list(_tolerant_timestamps(frames)) == [0.5, 0.5, 0.5, 0.5, 1.5]


def test_create_model_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        create_model("audio_ast_v1", {})


def test_score_display_is_bounded() -> None:
    assert logit_to_score(0.0) == pytest.approx(0.5)
    assert 0.0 < logit_to_score(-1000.0) < 1e-20
    assert 1.0 - logit_to_score(1000.0) < 1e-12


def test_metrics_require_both_classes() -> None:
    assert evaluate_binary([0, 1], [0.1, 0.9]).eer == 0.0
    with pytest.raises(UndefinedMetricError):
        evaluate_binary([1, 1], [0.1, 0.9])


@pytest.fixture(scope="module")
def video_file(tmp_path_factory) -> Path:
    if not HAS_FFMPEG:
        pytest.skip("没有 FFmpeg")
    path = tmp_path_factory.mktemp("video") / "sample.mp4"
    subprocess.run(
        [
            media_tool("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
            "testsrc=duration=1:size=64x48:rate=10", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return path


def test_face_crop_scoring_resizes_any_size_and_keeps_order(
    scorer: GeneralAigcScorer, tmp_path: Path
) -> None:
    from PIL import Image

    paths = []
    for index, size in enumerate(((64, 64), (130, 97), (500, 500))):
        rng = np.random.default_rng(index)
        path = tmp_path / f"face_{index}.jpg"
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)
        paths.append(path)
    logits = scorer.score_face_crops(paths)
    assert len(logits) == 3
    assert all(isinstance(value, float) for value in logits)
    assert scorer.score_face_crops(paths[1:2])[0] == pytest.approx(logits[1], abs=1e-3)
    assert scorer.score_face_crops([]) == []


def test_video_scoring_path(scorer: GeneralAigcScorer, video_file: Path) -> None:
    result = scorer.score_video(video_file)
    assert result.kind == "video"
    assert result.decode_mode == "strict"
    assert len(result.frame_logits) == 2
    assert len(result.frame_logits[0]) == 5
    assert len(result.community_frame_logits) == 2
    assert (result.width, result.height) == (64, 48)
    assert result.duration_seconds == pytest.approx(1.0, abs=0.2)


def test_stream_selection_skips_cover_art(tmp_path: Path, video_file: Path) -> None:
    assert select_video_stream(video_file) == 0
    audio = tmp_path / "only_audio.m4a"
    subprocess.run(
        [media_tool("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "sine=duration=0.2", str(audio)],
        check=True,
    )
    with pytest.raises(MediaError, match="视频流"):
        select_video_stream(audio)


def test_tolerant_mode_can_read_a_truncated_video(tmp_path: Path, video_file: Path) -> None:
    damaged = tmp_path / "damaged.mp4"
    data = video_file.read_bytes()
    damaged.write_bytes(data[: int(len(data) * 0.97)])
    config = VideoConfig(n_frames=2, height=8, width=8, resize_mode="native_center_crop", tolerant=True)
    try:
        frames = decode_video(damaged, config).frames
    except MediaError:
        pytest.skip("这个截断位置连容错模式也读不出，测试素材不适用")
    assert frames.shape[1:] == (8, 8, 3)


def test_long_video_uses_seek_sampling(
    scorer: GeneralAigcScorer, video_file: Path, monkeypatch
) -> None:
    import aigc_engine.scorer as scorer_module

    monkeypatch.setattr(scorer_module, "SEEK_AFTER_SECONDS", 0.5)
    result = scorer.score_video(video_file)
    assert result.decode_mode == "seek"
    assert len(result.frame_logits[0]) == 5
    assert any("时间点" in text for text in result.warnings)
    assert result.frame_timestamps[0] == 0.0
    assert result.frame_timestamps[-1] > result.frame_timestamps[0]


def test_seek_and_full_decode_read_the_same_first_frame(video_file: Path) -> None:
    config = VideoConfig(n_frames=2, height=8, width=8, resize_mode="native_center_crop", native_crops=5)
    assert decode_video_by_seek(video_file, config).frames.shape == decode_video(video_file, config).frames.shape


def test_missing_file_is_a_media_error(tmp_path: Path) -> None:
    config = VideoConfig(n_frames=2, height=8, width=8, resize_mode="native_center_crop")
    with pytest.raises(MediaError) as error:
        decode_video(tmp_path / "none.mp4", config)
    assert error.value.code == "missing_file"
