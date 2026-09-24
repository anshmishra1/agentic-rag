"""Flat typed state contracts for the Agentic RAG graph."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages


# =============================================================
# Request / conversation
# =============================================================

class RequestState(TypedDict, total=False):
    """User-facing request and persistent conversation inputs."""

    question: str
    document_id: str | None
    messages: Annotated[list, add_messages]


# =============================================================
# Query / routing context
# =============================================================

class QueryState(TypedDict, total=False):
    """Query representation and intent information."""

    retrieval_query: str
    query_intent: Literal["new_question", "follow_up", "control"] | str
    query_is_control: bool
    contextualization_used: bool


# =============================================================
# Retrieval and evidence assessment
# =============================================================

class RetrievalState(TypedDict, total=False):
    """Retrieved documents and the flat signals used by routing."""

    documents: list[Document]
    retrieval_scores: list[float]
    retrieval_top_score: float
    retrieval_second_score: float
    retrieval_score_gap: float
    retrieval_mean_score: float
    retrieval_top_to_mean_ratio: float
    retrieval_gap_ratio: float
    retrieval_overview_top_score: float | None
    retrieval_content_top_score: float | None
    retrieval_decision: str | None
    retrieval_evidence_strength: str | None
    retrieval_decision_reason: str | None
    relevance_grade: str | None


# =============================================================
# Answer / verification
# =============================================================

class AnswerState(TypedDict, total=False):
    """Generated answer and its verification state."""

    generation: str
    answer_status: Literal[
        "answered",
        "insufficient_evidence",
        "unsupported",
        "verification_uncertain",
        "control",
    ] | str | None
    hallucination_grade: str | None
    grounding_diagnosis: str | None
    grounding_unsupported_claims: list[str]
    grounding_parse_success: bool | None
    verification_exhausted: bool
    citations: list[str]


# =============================================================
# Control flow
# =============================================================

class ControlState(TypedDict, total=False):
    """Counters and explicit control decisions."""

    retry_count: int
    hallucination_retry_count: int
    correction_attempted: bool


# =============================================================
# Current graph state
# =============================================================

class RAGState(
    RequestState,
    QueryState,
    RetrievalState,
    AnswerState,
    ControlState,
    total=False,
):
    """Complete flat graph state; observability stays outside checkpoints."""


# =============================================================
# Public graph schemas
# =============================================================

class GraphInput(TypedDict, total=False):
    """Fields accepted from the API/UI when invoking the graph."""

    question: str
    document_id: str | None
    messages: Annotated[list, add_messages]


class GraphOutput(TypedDict, total=False):
    """Fields exposed to the caller after graph execution."""

    generation: str
    answer_status: str
    grounding_diagnosis: str
    verification_exhausted: bool
    citations: list[str]
