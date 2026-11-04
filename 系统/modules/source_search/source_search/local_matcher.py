from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class LocalMatch:
    score: float
    keypoints_a: list
    keypoints_b: list
    matches: list
    inlier_mask: list[int]
    good_count: int
    inlier_count: int
    ransac_attempted: bool = False
    ransac_applied: bool = False
    ransac_threshold: float = 5.0
    score_components: dict | None = None
    homography: np.ndarray | None = None
    raw_knn_pair_count: int = 0


def calculate_local_score(
    good_count: int,
    inlier_count: int,
    *,
    ransac_attempted: bool,
) -> tuple[float, dict]:
    """Apply the frozen v0.2 local-score formula and expose its true inputs."""

    if good_count < 0 or inlier_count < 0 or inlier_count > good_count:
        raise ValueError("invalid local match counts")
    if not ransac_attempted:
        normalized_good_count = good_count / 40.0
        score = min(normalized_good_count, 0.15)
        return score, {
            "mode": "insufficient_matches",
            "good_match_count": good_count,
            "normalized_good_match_count": normalized_good_count,
            "score_cap": 0.15,
            "score": score,
            "formula": "min(good_match_count / 40, 0.15)",
        }

    inlier_ratio = inlier_count / max(good_count, 1)
    normalized_inlier_count = min(inlier_count / 40.0, 1.0)
    score = float(
        np.clip(0.6 * inlier_ratio + 0.4 * normalized_inlier_count, 0, 1)
    )
    return score, {
        "mode": "normal",
        "ransac_inlier_ratio": inlier_ratio,
        "normalized_inlier_count": normalized_inlier_count,
        "inlier_ratio_weight": 0.6,
        "inlier_count_weight": 0.4,
        "score": score,
        "formula": (
            "0.6 * ransac_inlier_ratio + "
            "0.4 * min(ransac_inlier_count / 40, 1)"
        ),
    }


class LocalMatcher:
    def __init__(self, ratio: float = 0.75, ransac_threshold: float = 5.0):
        self.ratio = ratio
        self.ransac_threshold = ransac_threshold
        self.sift = cv2.SIFT_create(nfeatures=2500)

    def match(
        self,
        a: np.ndarray | None,
        mask_a: np.ndarray | None,
        b: np.ndarray | None,
        mask_b: np.ndarray | None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        *,
        features_a: tuple[Any, Any] | None = None,
        features_b: tuple[Any, Any] | None = None,
    ) -> LocalMatch:
        # features_a/b 是已算好的 (关键点, 描述子)，只省去重复提取，匹配与打分不变。
        if features_a is None:
            gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            kp_a, des_a = self.sift.detectAndCompute(gray_a, mask_a)
        else:
            kp_a, des_a = features_a
        if features_b is None:
            gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            kp_b, des_b = self.sift.detectAndCompute(gray_b, mask_b)
        else:
            kp_b, des_b = features_b
        if progress_callback is not None:
            progress_callback(
                "sift_features_extracted",
                {
                    "query_keypoint_count": len(kp_a or []),
                    "candidate_keypoint_count": len(kp_b or []),
                    "query_descriptor_count": (
                        int(len(des_a)) if des_a is not None else 0
                    ),
                    "candidate_descriptor_count": (
                        int(len(des_b)) if des_b is not None else 0
                    ),
                },
            )
        empty_score, empty_components = calculate_local_score(
            0, 0, ransac_attempted=False
        )
        empty = LocalMatch(
            empty_score,
            kp_a or [],
            kp_b or [],
            [],
            [],
            0,
            0,
            ransac_attempted=False,
            ransac_applied=False,
            ransac_threshold=self.ransac_threshold,
            score_components=empty_components,
            homography=None,
            raw_knn_pair_count=0,
        )
        if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
            if progress_callback is not None:
                progress_callback(
                    "lowe_ratio_filtered",
                    {
                        "raw_knn_pair_count": 0,
                        "good_match_count": 0,
                        "lowe_ratio": self.ratio,
                        "reason": "insufficient_descriptors",
                    },
                )
                progress_callback(
                    "ransac_verified",
                    {
                        "ransac_attempted": False,
                        "ransac_applied": False,
                        "ransac_inlier_count": 0,
                        "ransac_inlier_ratio": 0.0,
                        "reason": "insufficient_matches",
                    },
                )
            return empty
        raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
        good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < self.ratio * n.distance]
        if progress_callback is not None:
            progress_callback(
                "lowe_ratio_filtered",
                {
                    "raw_knn_pair_count": len(raw),
                    "good_match_count": len(good),
                    "lowe_ratio": self.ratio,
                },
            )
        if len(good) < 4:
            score, components = calculate_local_score(
                len(good), 0, ransac_attempted=False
            )
            if progress_callback is not None:
                progress_callback(
                    "ransac_verified",
                    {
                        "ransac_attempted": False,
                        "ransac_applied": False,
                        "ransac_inlier_count": 0,
                        "ransac_inlier_ratio": 0.0,
                        "reason": "fewer_than_four_lowe_matches",
                    },
                )
            return LocalMatch(
                score,
                kp_a,
                kp_b,
                good,
                [0] * len(good),
                len(good),
                0,
                ransac_attempted=False,
                ransac_applied=False,
                ransac_threshold=self.ransac_threshold,
                score_components=components,
                homography=None,
                raw_knn_pair_count=len(raw),
            )
        src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        try:
            homography, inliers = cv2.findHomography(
                src, dst, cv2.RANSAC, self.ransac_threshold
            )
        except cv2.error:
            homography = None
            inliers = None
        mask = inliers.ravel().astype(int).tolist() if inliers is not None else [0] * len(good)
        count = int(sum(mask))
        if progress_callback is not None:
            progress_callback(
                "ransac_verified",
                {
                    "ransac_attempted": True,
                    "ransac_applied": inliers is not None,
                    "ransac_inlier_count": count,
                    "ransac_inlier_ratio": count / len(good),
                    "ransac_threshold": self.ransac_threshold,
                    "homography_available": homography is not None,
                },
            )
        score, components = calculate_local_score(
            len(good), count, ransac_attempted=True
        )
        return LocalMatch(
            score,
            kp_a,
            kp_b,
            good,
            mask,
            len(good),
            count,
            ransac_attempted=True,
            ransac_applied=inliers is not None,
            ransac_threshold=self.ransac_threshold,
            score_components=components,
            homography=homography,
            raw_knn_pair_count=len(raw),
        )
