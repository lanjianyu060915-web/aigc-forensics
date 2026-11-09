"""Tie-grouped EER and ROC AUC for fake=1 and larger scores meaning fake.

FAR is fake accepted as real (false negative rate); FRR is real rejected as
fake (false positive rate). The EER interpolation is between achievable DET
points, not a claim that an interpolated numerical threshold attains that EER.
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite


class MetricInputError(ValueError):
    """Malformed labels/scores, including nonfinite scores."""


class UndefinedMetricError(MetricInputError):
    """A metric is undefined because at least one class is absent."""


@dataclass(frozen=True)
class DETPoint:
    # None denotes the all-real endpoint: a threshold strictly above max(score).
    # This is JSON-safe even if max(score) is the largest representable float.
    threshold: float | None
    far: float
    frr: float


@dataclass(frozen=True)
class BinaryMetrics:
    protocol: str
    eer: float
    roc_auc: float
    real_count: int
    fake_count: int
    lower_point: DETPoint
    upper_point: DETPoint
    interpolation_weight: float
    nearest_point: DETPoint
    nearest_eer: float
    det_curve: tuple[DETPoint, ...]


def validate_inputs(labels: Sequence[int], scores: Sequence[float]) -> None:
    """Validate values and require both classes; errors include row indices."""
    if len(labels) != len(scores):
        raise MetricInputError("labels and scores must have equal length")
    invalid_labels = [
        index
        for index, label in enumerate(labels)
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1)
    ]
    if invalid_labels:
        raise MetricInputError(f"labels must be integer 0 or 1; rows={invalid_labels}")
    invalid_scores = [
        index
        for index, score in enumerate(scores)
        if isinstance(score, bool)
        or not isinstance(score, (float, int))
        or (isinstance(score, int) and abs(score) > sys.float_info.max)
        or not isfinite(score)
    ]
    if invalid_scores:
        raise MetricInputError(f"scores must be finite numbers; rows={invalid_scores}")
    if not labels or sum(labels) in (0, len(labels)):
        raise UndefinedMetricError("EER and ROC AUC require both real and fake samples")


def evaluate_binary(labels: Sequence[int], scores: Sequence[float]) -> BinaryMetrics:
    """Compute ``eer_interpolated_v1`` using ``score >= threshold``.

    Equal scores move together, so row ordering never influences the result.
    ``lower_point`` and ``upper_point`` are ordered by FRR (ascending), not by
    threshold. Their thresholds describe the crossing interval. An exact
    crossing returns the same point twice and interpolation_weight=0.
    The nearest discrete point minimizes abs(FAR-FRR), resolving equal-distance
    choices toward the higher threshold; nearest_eer=(FAR+FRR)/2 at that point.
    """
    validate_inputs(labels, scores)
    fake_count = sum(labels)
    real_count = len(labels) - fake_count
    tied: dict[float, list[int]] = {}
    for label, score in zip(labels, scores, strict=True):
        # Keep exact integer ordering; float conversion can merge distinct scores
        # above 2**53 into an artificial tie.
        counts = tied.setdefault(score, [0, 0])
        counts[label] += 1
    points = [DETPoint(None, far=1.0, frr=0.0)]
    rejected_real = 0
    rejected_fake = 0
    for threshold in sorted(tied, reverse=True):
        real, fake = tied[threshold]
        rejected_real += real
        rejected_fake += fake
        points.append(
            DETPoint(
                threshold,
                far=(fake_count - rejected_fake) / fake_count,
                frr=rejected_real / real_count,
            )
        )
    nearest = min(points, key=lambda point: abs(point.far - point.frr))
    auc = sum(
        (right.frr - left.frr) * (2.0 - left.far - right.far) / 2.0
        for left, right in pairwise(points)
    )
    for index, point in enumerate(points):
        difference = point.far - point.frr
        if difference == 0:
            lower = upper = point
            weight = 0.0
            eer = point.far
            break
        if difference < 0:
            lower, upper = points[index - 1], point
            lower_difference = lower.far - lower.frr
            weight = lower_difference / (lower_difference - difference)
            eer = lower.far + weight * (upper.far - lower.far)
            break
    else:  # Both endpoints are present, so this cannot occur for valid inputs.
        raise RuntimeError("DET curve has no crossing")
    return BinaryMetrics(
        protocol="eer_interpolated_v1",
        eer=eer,
        roc_auc=auc,
        real_count=real_count,
        fake_count=fake_count,
        lower_point=lower,
        upper_point=upper,
        interpolation_weight=weight,
        nearest_point=nearest,
        nearest_eer=(nearest.far + nearest.frr) / 2.0,
        det_curve=tuple(points),
    )
