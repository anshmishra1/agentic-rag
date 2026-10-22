"""Content-free structured events for evidence-agent execution."""

from __future__ import annotations

from typing import Any, Literal


ResearchEvent = Literal[
    "tool_started",
    "tool_completed",
    "tool_failed",
    "selection_completed",
]

_ALLOWED_FIELDS = frozenset(
    {
        "tool",
        "purpose_code",
        "evidence_requirement",
        "candidate_count",
        "structure_count",
        "selected_count",
        "duplicates_removed",
        "requirement_satisfied",
        "elapsed_ms",
        "error_type",
    }
)


def emit_research_event(event: ResearchEvent, **fields: Any) -> None:
    """Write operational metadata without questions, passages, or identities."""

    unexpected = set(fields) - _ALLOWED_FIELDS
    if unexpected:
        rejected = ", ".join(sorted(unexpected))
        raise ValueError(
            f"Evidence-agent events must remain content-free; rejected: {rejected}"
        )

    # Lazy import keeps pure contracts and their unit tests free from logging
    # initialization side effects.
    from agentic_rag.core.logging import log_event

    log_event(
        "evidence_agent",
        message=f"Evidence agent event: {event}",
        agent_event=event,
        **fields,
    )

