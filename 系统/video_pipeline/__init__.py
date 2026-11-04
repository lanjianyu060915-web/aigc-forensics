# -*- coding: utf-8 -*-
"""视频任务规则层：抽帧选帧（selection）与多帧聚合（aggregate）。"""

from .aggregate import (
    CALIBRATED_VIDEO_RULE,
    DEEPFAKE_RULE,
    IDENTITY_RULE,
    SOURCE_RULE,
    VIDEO_LIMITATIONS,
    VIDEO_LIMITATIONS_DEEPFAKE,
    VIDEO_LIMITATIONS_SHARED,
    aggregate_deepfake_frames,
    aggregate_identity_frames,
    annotate_single_frame,
    build_video_summary,
    rebase_frame_artifacts,
)
from .selection import (
    SAMPLE_STRATEGY,
    SELECTION_RULE,
    candidate_indices,
    percentile,
    score_candidates,
    select_key_frames,
)

__all__ = [
    "CALIBRATED_VIDEO_RULE",
    "DEEPFAKE_RULE",
    "IDENTITY_RULE",
    "SAMPLE_STRATEGY",
    "SELECTION_RULE",
    "SOURCE_RULE",
    "VIDEO_LIMITATIONS",
    "VIDEO_LIMITATIONS_DEEPFAKE",
    "VIDEO_LIMITATIONS_SHARED",
    "aggregate_deepfake_frames",
    "aggregate_identity_frames",
    "annotate_single_frame",
    "build_video_summary",
    "candidate_indices",
    "percentile",
    "rebase_frame_artifacts",
    "score_candidates",
    "select_key_frames",
]
