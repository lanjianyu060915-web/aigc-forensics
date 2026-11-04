from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "modules" / "source_search"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from source_search.local_matcher import LocalMatcher


def _full_mask(image: np.ndarray) -> np.ndarray:
    return np.full(image.shape[:2], 255, dtype=np.uint8)


def test_sift_insufficient_descriptors_returns_auditable_fallback() -> None:
    """A texture-free image must not turn a local-search edge into an error."""

    query = np.full((192, 192, 3), 127, dtype=np.uint8)
    candidate = np.full((192, 192, 3), 129, dtype=np.uint8)
    stages: list[tuple[str, dict]] = []

    result = LocalMatcher().match(
        query,
        _full_mask(query),
        candidate,
        _full_mask(candidate),
        progress_callback=lambda stage, details: stages.append((stage, details)),
    )

    assert result.good_count == 0
    assert result.inlier_count == 0
    assert result.ransac_attempted is False
    assert result.ransac_applied is False
    assert result.homography is None
    assert result.score == pytest.approx(0.0)
    assert result.score_components == {
        "mode": "insufficient_matches",
        "good_match_count": 0,
        "normalized_good_match_count": 0.0,
        "score_cap": 0.15,
        "score": 0.0,
        "formula": "min(good_match_count / 40, 0.15)",
    }
    assert stages[-1][0] == "ransac_verified"
    assert stages[-1][1]["reason"] == "insufficient_matches"


def test_ransac_estimation_failure_is_recorded_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject a failed homography after real SIFT/Lowe matching.

    The descriptors and Lowe matches are computed by OpenCV from the actual
    images.  Only the RANSAC estimator return value is fault-injected so that
    this otherwise rare branch is deterministic.
    """

    rng = np.random.default_rng(20260831)
    query = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    candidate = query.copy()
    stages: list[tuple[str, dict]] = []

    monkeypatch.setattr(cv2, "findHomography", lambda *_args, **_kwargs: (None, None))
    result = LocalMatcher().match(
        query,
        _full_mask(query),
        candidate,
        _full_mask(candidate),
        progress_callback=lambda stage, details: stages.append((stage, details)),
    )

    assert result.good_count >= 4
    assert result.ransac_attempted is True
    assert result.ransac_applied is False
    assert result.inlier_count == 0
    assert result.homography is None
    assert result.score == pytest.approx(0.0)
    ransac = next(details for stage, details in stages if stage == "ransac_verified")
    assert ransac["ransac_attempted"] is True
    assert ransac["ransac_applied"] is False
    assert ransac["homography_available"] is False

