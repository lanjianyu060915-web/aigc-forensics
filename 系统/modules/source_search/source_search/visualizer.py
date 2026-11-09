from __future__ import annotations

import cv2
import numpy as np

from .local_matcher import LocalMatch


def _with_header(image: np.ndarray, label: str, height: int = 46) -> np.ndarray:
    bar = np.zeros((height, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        bar,
        label,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([bar, image])


def make_query_face_detection(
    query: np.ndarray,
    face_box: tuple[int, int, int, int] | None,
    search_mask: np.ndarray,
) -> np.ndarray:
    """Show the detector box and the exact mask consumed by retrieval."""

    shown = query.copy()
    contours, _ = cv2.findContours(
        (search_mask == 0).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        cv2.drawContours(shown, contours, -1, (0, 165, 255), 5)
    if face_box is not None:
        x1, y1, x2, y2 = face_box
        cv2.rectangle(shown, (x1, y1), (x2, y2), (255, 180, 0), 5)
        cv2.putText(
            shown,
            "detected face / masked region",
            (max(0, x1), max(36, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 180, 0),
            3,
            cv2.LINE_AA,
        )
        label = "Face detected; orange contour is the actual retrieval mask"
    else:
        label = "No face detected; the full query image is used"
    return _with_header(shown, label)


def make_keypoints_visualization(
    image: np.ndarray, keypoints: list, label: str
) -> np.ndarray:
    shown = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 220, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return _with_header(shown, f"{label} | SIFT keypoints={len(keypoints)}")


def make_all_matches_visualization(
    query: np.ndarray,
    candidate: np.ndarray,
    match: LocalMatch,
    label: str,
) -> np.ndarray:
    """Draw every match that really passed the configured Lowe ratio test."""

    shown = cv2.drawMatches(
        query,
        match.keypoints_a,
        candidate,
        match.keypoints_b,
        match.matches,
        None,
        matchColor=(0, 220, 255),
        singlePointColor=(100, 100, 100),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return _with_header(
        shown,
        f"{label} | all Lowe-passed matches={match.good_count}",
    )


def make_inlier_outlier_visualization(
    query: np.ndarray,
    candidate: np.ndarray,
    match: LocalMatch,
    label: str,
) -> np.ndarray:
    """Draw RANSAC inliers in green and excluded Lowe matches in red."""

    height = max(query.shape[0], candidate.shape[0])
    width = query.shape[1] + candidate.shape[1]
    shown = np.zeros((height, width, 3), dtype=np.uint8)
    shown[: query.shape[0], : query.shape[1]] = query
    shown[: candidate.shape[0], query.shape[1] :] = candidate
    for index, item in enumerate(match.matches):
        query_point = tuple(
            int(round(value)) for value in match.keypoints_a[item.queryIdx].pt
        )
        candidate_point_raw = match.keypoints_b[item.trainIdx].pt
        candidate_point = (
            int(round(candidate_point_raw[0])) + query.shape[1],
            int(round(candidate_point_raw[1])),
        )
        is_inlier = (
            index < len(match.inlier_mask) and bool(match.inlier_mask[index])
        )
        color = (50, 220, 50) if is_inlier else (40, 40, 230)
        cv2.circle(shown, query_point, 5, color, 2, cv2.LINE_AA)
        cv2.circle(shown, candidate_point, 5, color, 2, cv2.LINE_AA)
        cv2.line(shown, query_point, candidate_point, color, 2, cv2.LINE_AA)
    return _with_header(
        shown,
        (
            f"{label} | green=inlier {match.inlier_count}; "
            f"red=outlier {match.good_count - match.inlier_count}"
        ),
    )


def make_homography_overlay(
    query: np.ndarray,
    candidate: np.ndarray,
    match: LocalMatch,
    label: str,
) -> np.ndarray | None:
    """Project the query boundary onto the candidate when a valid H exists."""

    homography = match.homography
    if homography is None or homography.shape != (3, 3):
        return None
    if not np.isfinite(homography).all():
        return None
    query_height, query_width = query.shape[:2]
    corners = np.float32(
        [[0, 0], [query_width - 1, 0], [query_width - 1, query_height - 1], [0, query_height - 1]]
    ).reshape(-1, 1, 2)
    try:
        projected = cv2.perspectiveTransform(corners, homography)
    except cv2.error:
        return None
    if not np.isfinite(projected).all():
        return None
    # Avoid creating a misleading polygon when the numerical projection is far
    # outside any plausible view of the candidate.
    bound = float(max(candidate.shape[:2]) * 20)
    if float(np.abs(projected).max(initial=0.0)) > bound:
        return None
    shown = candidate.copy()
    cv2.polylines(
        shown,
        [np.int32(np.round(projected))],
        True,
        (0, 220, 255),
        5,
        cv2.LINE_AA,
    )
    return _with_header(
        shown,
        f"{label} | projected query boundary from estimated homography",
    )


def make_visualization(
    query: np.ndarray,
    candidate: np.ndarray,
    match: LocalMatch,
    label: str,
) -> np.ndarray:
    """Backward-compatible legacy visualization: RANSAC inliers only."""

    shown = cv2.drawMatches(
        query, match.keypoints_a, candidate, match.keypoints_b, match.matches, None,
        matchesMask=match.inlier_mask or None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return _with_header(shown, label)
