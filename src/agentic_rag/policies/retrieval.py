"""Retrieval confidence policy.

The retrieval pipeline now uses:

    Dense + BM25
        ↓
       RRF
        ↓
Cross-encoder reranking
        ↓
retrieval assessment

The cross-encoder sigmoid score is NOT treated as a calibrated probability.
It is primarily used for ranking.

Confidence therefore combines:
    1. top-vs-second separation
    2. top-vs-mean separation
    3. minimum ranking evidence
    4. retry state
    5. overview/content evidence where available

The policy deliberately avoids treating the old Pinecone cosine-score
thresholds as valid for the new cross-encoder score distribution.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from agentic_rag.config import settings


RelevanceGrade = Literal["relevant", "irrelevant", "uncertain"]


class RelevanceVerdict(BaseModel):
    """Closed semantic-grading result used by provider adapters."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["relevant", "irrelevant"]


def groq_relevance_response_format() -> dict:
    """Return Groq's strict JSON-schema envelope for relevance grading."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "retrieval_relevance",
            "strict": True,
            "schema": RelevanceVerdict.model_json_schema(),
        },
    }


def parse_relevance_grade(content: str) -> RelevanceGrade:
    """Parse JSON or an exact legacy word; malformed output is uncertain."""

    text = content.strip()
    if not text:
        return "uncertain"

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    if text.casefold() in {"relevant", "irrelevant"}:
        return text.casefold()

    try:
        return RelevanceVerdict.model_validate(json.loads(text)).verdict
    except (json.JSONDecodeError, TypeError, ValueError):
        return "uncertain"


def is_repeated_rewrite(candidate: str, original: str, previous: str) -> bool:
    """Reject empty or unchanged queries before repeating a retrieval call."""
    normalized = " ".join(re.findall(r"\w+", candidate.casefold()))
    if not normalized:
        return True
    return normalized in {
        " ".join(re.findall(r"\w+", query.casefold()))
        for query in (original, previous)
    }


def build_rewrite_prompt(
    original: str, previous: str, reason: str, relevance_grade: str | None
) -> str:
    """Give the rewrite model the failure signal and a distinct next strategy."""
    if relevance_grade == "irrelevant":
        guidance = (
            "The retrieved passages were judged irrelevant; "
            "search a different aspect or section."
        )
    elif reason == "grounding_insufficient_evidence":
        guidance = (
            "The answer could not be fully supported; search for the specific "
            "mechanism, stages, definitions, or relationships needed to answer."
        )
    elif reason.startswith("top_score_below_floor"):
        guidance = (
            "The match was weak; correct wording or use more specific "
            "terms from the question."
        )
    elif reason.startswith("weak_distribution"):
        guidance = (
            "Candidates were similarly ranked; focus on the question's "
            "distinguishing terms."
        )
    else:
        guidance = "Try a different search angle while preserving every part of the question."

    return (
        "Rewrite the search query for this document. Preserve every factual "
        "constraint in the original question, but remove presentation requests "
        "such as flowchart, table, bullets, or LaTeX because they do not help "
        "retrieval. Do not add facts the user did not ask about.\n\n"
        f"Original question: {original}\n"
        f"Most recent search attempt: {previous}\n"
        f"Retrieval failure: {reason}; relevance grade: {relevance_grade or 'not graded'}.\n"
        f"Next strategy: {guidance}\n"
        "Return one different search query. Do not repeat the original or "
        "most recent attempt. Return only the query."
    )


def assess_retrieval_confidence(
    metrics: dict,
    top_doc_type: str | None,
    overview_top_score: float | None,
    content_top_score: float | None,
    retry_count: int = 0,
) -> dict:
    """Classify retrieved evidence as strong, ambiguous, or weak.

    Returns:
        {
            "decision": "generate" | "grade" | "rewrite_query",
            "evidence_strength": "strong" | "ambiguous" | "weak",
            "reason": str,
        }

    Important:
        Cross-encoder sigmoid scores are ranking signals, not calibrated
        probabilities. Therefore absolute CE thresholds are deliberately
        not used as the primary decision mechanism.
    """

    top_score = float(metrics.get("top_score", 0.0))
    second_score = float(metrics.get("second_score", 0.0))
    mean_score = float(metrics.get("mean_score", 0.0))

    top_to_mean = float(
        metrics.get("top_to_mean_ratio", 0.0)
    )
    gap_ratio = float(
        metrics.get("gap_ratio", 0.0)
    )

    score_gap = top_score - second_score

    # ---------------------------------------------------------
    # 1. No usable retrieval evidence
    # ---------------------------------------------------------
    if top_score <= 0.0 or not metrics:
        if retry_count >= settings.max_retries:
            return {
                "decision": "grade",
                "evidence_strength": "weak",
                "reason": "no_usable_retrieval_evidence_retries_exhausted",
            }

        return {
            "decision": "rewrite_query",
            "evidence_strength": "weak",
            "reason": "no_usable_retrieval_evidence",
        }
        # ---------------------------------------------------------
    # 1b. Top score below the provisional floor
    #
    # Ratios computed from near-zero scores are numerically
    # unstable and can look "strong" by pure noise. The floor is not
    # calibrated, so it may prevent distribution-based auto-generation, but
    # it must not discard candidates or trigger a blind rewrite. Ask the
    # semantic grader whether the retrieved text actually answers the query.
    # ---------------------------------------------------------
    if top_score < settings.retrieval_min_top_score:
        return {
            "decision": "grade",
            "evidence_strength": "weak",
            "reason": "top_score_below_floor_requires_semantic_grading",
        }
    # ---------------------------------------------------------
    # 2. Close-ranked candidates need semantic grading
    #
    # Several relevant passages can all score highly, leaving the top result
    # close to its peers. Once the minimum score floor is met, score shape
    # alone is not evidence that a different query would retrieve better.
    # ---------------------------------------------------------
    weak_distribution = (
        top_to_mean < settings.retrieval_top_to_mean_ratio
        and gap_ratio < settings.retrieval_gap_ratio
    )

    if weak_distribution:
        return {
            "decision": "grade",
            "evidence_strength": "ambiguous",
            "reason": "weak_candidate_separation",
        }

    # ---------------------------------------------------------
    # 3. Strong evidence
    #
    # We require BOTH:
    #
    #   - meaningful top-vs-mean separation
    #   - meaningful top-vs-second separation
    #
    # This prevents one tiny numerical difference from being
    # interpreted as strong evidence.
    # ---------------------------------------------------------
    strong_distribution = (
        top_to_mean >= settings.retrieval_top_to_mean_ratio
        and gap_ratio >= settings.retrieval_gap_ratio
    )

    if strong_distribution:
        return {
            "decision": "generate",
            "evidence_strength": "strong",
            "reason": "strong_cross_encoder_score_distribution",
        }

    # ---------------------------------------------------------
    # 4. Overview evidence
    #
    # An overview chunk can legitimately answer broad document-level
    # questions even if individual content chunks have weaker scores.
    # ---------------------------------------------------------
    overview_dominant = (
        top_doc_type == "overview"
        and overview_top_score is not None
        and content_top_score is not None
        and overview_top_score
        >= content_top_score
        * (1.0 + settings.retrieval_overview_margin)
    )

    if overview_dominant and gap_ratio >= settings.retrieval_gap_ratio:
        return {
            "decision": "generate",
            "evidence_strength": "strong",
            "reason": "overview_dominant_with_clear_margin",
        }

    # ---------------------------------------------------------
    # 5. Ambiguous evidence
    #
    # The retrieval has signal, but not enough confidence to
    # bypass semantic grading.
    # ---------------------------------------------------------
    if gap_ratio < settings.retrieval_gap_ratio:
        reason = "top_candidates_too_close"
    elif top_to_mean < settings.retrieval_top_to_mean_ratio:
        reason = "weak_top_candidate_separation"
    else:
        reason = "retrieval_requires_semantic_grading"

    return {
        "decision": "grade",
        "evidence_strength": "ambiguous",
        "reason": reason,
    }
