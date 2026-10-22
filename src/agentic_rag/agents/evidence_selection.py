"""One-pass global selection for bounded evidence-agent observations."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from agentic_rag.agents.evidence_contracts import (
    EvidenceSelection,
    EvidenceToolObservation,
)
from agentic_rag.agents.observability import emit_research_event
from agentic_rag.policies.evidence import (
    evidence_requirement_is_satisfied,
    select_required_evidence,
)
from agentic_rag.retrieval.candidates import deduplicate_candidates


Reranker = Callable[..., dict[str, list[tuple]]]
EventSink = Callable[..., None]


def _default_reranker(*args, **kwargs):
    # Keep contract and unit-test imports independent of the local Torch stack.
    from agentic_rag.retrieval.reranker import rerank_many

    return rerank_many(*args, **kwargs)


def merge_and_select_evidence(
    *,
    query: str,
    observations: Sequence[EvidenceToolObservation],
    evidence_requirement: str,
    top_k: int,
    required_slots: int,
    companion_slots: int,
    reranker: Reranker = _default_reranker,
    event_sink: EventSink = emit_research_event,
) -> EvidenceSelection:
    """Deduplicate all tool results, rerank once, and enforce evidence lanes."""

    if not query.strip():
        raise ValueError("query must be nonempty")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if required_slots < 0 or companion_slots < 0:
        raise ValueError("evidence slot counts cannot be negative")

    started = time.perf_counter()
    flattened = [candidate for observation in observations for candidate in observation.candidates]
    # Scores use the same V3 RRF contract, so sorting before deduplication keeps
    # the strongest occurrence when independent searches return one record.
    flattened.sort(key=lambda item: float(item[1]), reverse=True)
    candidates, duplicates_removed = deduplicate_candidates(flattened)

    if candidates:
        reranked = reranker(
            query,
            {"content": candidates},
            {"content": len(candidates)},
        ).get("content", [])
        selected = select_required_evidence(
            reranked,
            candidates,
            requirement=evidence_requirement,
            top_k=top_k,
            required_slots=required_slots,
            companion_slots=companion_slots,
        )
    else:
        selected = []

    requirement_satisfied = evidence_requirement_is_satisfied(
        [document for document, _ in selected],
        evidence_requirement,
    )
    event_sink(
        "selection_completed",
        evidence_requirement=evidence_requirement,
        candidate_count=len(candidates),
        duplicates_removed=duplicates_removed,
        selected_count=len(selected),
        requirement_satisfied=requirement_satisfied,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )
    return EvidenceSelection(
        selected=tuple(selected),
        candidate_count=len(candidates),
        duplicates_removed=duplicates_removed,
        requirement_satisfied=requirement_satisfied,
    )

