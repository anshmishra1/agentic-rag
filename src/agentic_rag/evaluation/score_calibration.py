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


def evaluate_score_floor(rows: list[dict[str, Any]], floor: float) -> dict[str, int | float]:
    """Evaluate one fixed score floor without selecting it from these rows."""
    if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be finite and within [0, 1]")
    # Reuse the score and label validation before computing errors.
    analyze_labeled_scores(rows, minimum_per_class=1)
    true_positive = false_positive = true_negative = false_negative = 0
    for row in rows:
        predicted = float(row["top_score"]) >= floor
        actual = row["should_match"]
        if predicted and actual:
            true_positive += 1
        elif predicted:
            false_positive += 1
        elif actual:
            false_negative += 1
        else:
            true_negative += 1
    return {
        "floor": floor,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def analyze_scored_rows(rows: list[dict[str, Any]], *, minimum_per_class: int = 5) -> dict[str, Any]:
    """Tune on development rows and use holdout rows only for evaluation."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("Score input must be a nonempty JSON list")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Every score row must be an object")
    splits = {row.get("split") for row in rows}
    if splits == {"development", "holdout"}:
        development = [row for row in rows if row["split"] == "development"]
        holdout = [row for row in rows if row["split"] == "holdout"]
        development_report = analyze_labeled_scores(development, minimum_per_class=minimum_per_class)
        candidate = development_report["candidate_min_top_score"]
        return {
            "development": development_report,
            "holdout_count": len(holdout),
            "holdout_at_candidate_floor": (
                evaluate_score_floor(holdout, candidate) if candidate is not None else None
            ),
        }
    if splits in ({None}, {"development"}):
        return analyze_labeled_scores(rows, minimum_per_class=minimum_per_class)
    raise ValueError("split must be absent or consistently contain development and holdout rows")
