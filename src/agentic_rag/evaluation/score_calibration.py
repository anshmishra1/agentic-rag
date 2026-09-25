"""Pure score analysis for a labeled retrieval calibration run."""

from __future__ import annotations

import math
from statistics import median
from typing import Any


def analyze_labeled_scores(rows: list[dict[str, Any]], *, minimum_per_class: int = 5) -> dict[str, Any]:
    """Suggest a score floor only when labeled classes separate cleanly.

    A cross-encoder score is a ranking signal. A clean gap on a small sample
    is only a candidate for later validation, never a probability estimate.
    """
    if minimum_per_class < 1:
        raise ValueError("minimum_per_class must be positive")
    if not isinstance(rows, list):
        raise ValueError("Score input must be a JSON list")

    positives: list[float] = []
    negatives: list[float] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or type(row.get("should_match")) is not bool:
            raise ValueError(f"Row {index} needs a boolean should_match label")
        try:
            score = float(row["top_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Row {index} needs a numeric top_score") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Row {index} top_score must be finite and within [0, 1]")
        (positives if row["should_match"] else negatives).append(score)

    def summary(scores: list[float]) -> dict[str, float | int | None]:
        return {
            "count": len(scores),
            "minimum": min(scores) if scores else None,
            "median": median(scores) if scores else None,
            "maximum": max(scores) if scores else None,
        }

    candidate_floor = None
    if len(positives) < minimum_per_class or len(negatives) < minimum_per_class:
        reason = "insufficient_labeled_examples"
    elif min(positives) <= max(negatives):
        reason = "score_distributions_overlap"
    else:
        candidate_floor = (min(positives) + max(negatives)) / 2
        reason = "separated_sample_requires_holdout_validation"

    return {
        "positive": summary(positives),
        "negative": summary(negatives),
        "minimum_per_class": minimum_per_class,
        "candidate_min_top_score": candidate_floor,
        "reason": reason,
    }
