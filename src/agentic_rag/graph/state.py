"""Typed state contracts for the Agentic RAG graph.

The graph still uses one internal state object for now, but its fields are
organized into domain-specific TypedDicts so retrieval, evidence, answer,
control-flow, and observability responsibilities can be separated without
changing graph behavior in this step.

The legacy flat fields are intentionally retained on RAGState during the
migration. They will be removed only after all nodes have been migrated to the
new domain contracts and the application has been regression-tested.
"""

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
    document_id: str
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
# Retrieval
# =============================================================

class RetrievalMetrics(TypedDict, total=False):
    """Retrieval diagnostics kept together during the migration.

    The legacy ratio/gap metrics remain here temporarily for observability and
    backwards compatibility. They are not intended to remain routing signals
    after the evidence-policy migration is complete.
    """

    top_score: float
    second_score: float
    score_gap: float
    mean_score: float
    top_to_mean_ratio: float
    gap_ratio: float
    overview_top_score: float | None
    content_top_score: float | None


class EvidenceRef(TypedDict, total=False):
    """Compact reference to a retrieved chunk."""

    chunk_id: str
    document_id: str
    rank: int
    score: float
    page: str | int | None
    source: str | None
    chunk_type: str | None


class RetrievalState(TypedDict, total=False):
    """Working retrieval data.

    Full Document objects are still retained during this migration because the
    current generation path consumes them directly. A later step can reduce
    durable state to EvidenceRef + selected_context.
    """

    documents: list[Document]
    retrieval_scores: list[float]
    metrics: RetrievalMetrics
    evidence_ids: list[str]
    selected_context: str


# =============================================================
# Evidence assessment
# =============================================================

class NLIResult(TypedDict, total=False):
    """Reserved contract for the upcoming NLI evidence layer."""

    entailment: float
    neutral: float
    contradiction: float
    label: Literal["entailment", "neutral", "contradiction"] | str


class EvidenceProfile(TypedDict, total=False):
    """Interpretation of retrieval signals.

    This is deliberately separate from RetrievalMetrics: retrieval metrics say
    what the retrieval stack produced; the evidence profile will say what those
    results imply about answerability.
    """

    candidate_count: int
    dense_bm25_agreement: float | None
    rrf_concentration: float | None
    ce_top_score: float | None
    ce_separation: float | None
    supporting_chunk_count: int
    overview_support: bool
    content_support: bool
    nli: NLIResult | None


class EvidenceState(TypedDict, total=False):
    """Evidence interpretation and semantic judgement."""

    profile: EvidenceProfile
    strength: Literal["clear", "ambiguous", "insufficient"] | str
    decision: Literal["generate", "grade", "rewrite_query"] | str
    decision_reason: str
    llm_judgement: dict
    answerability: str


# =============================================================
# Answer / verification
# =============================================================

class GroundingState(TypedDict, total=False):
    """Answer-grounding and unsupported-claim information."""

    hallucination_grade: (
        Literal["grounded", "hallucinated", "uncertain"] | str
    )
    grounding_diagnosis: str
    unsupported_claims: list[str]


class AnswerState(TypedDict, total=False):
    """Generated answer and its verification state."""

    generation: str
    answer_status: Literal[
        "answered",
        "insufficient_evidence",
        "unsupported",
        "verification_uncertain",
    ] | str
    grounding: GroundingState
    citations: list[str]


# =============================================================
# Control flow
# =============================================================

class ControlState(TypedDict, total=False):
    """Counters and explicit control decisions."""

    retrieval_attempts: int
    grounding_attempts: int
    next_action: Literal[
        "generate",
        "grade",
        "rewrite_query",
        "verify",
        "abstain",
        "end",
    ] | str


# =============================================================
# Current graph state
# =============================================================

class RAGState(
    RequestState,
    QueryState,
    RetrievalState,
    EvidenceState,
    AnswerState,
    ControlState,
    total=False,
):
    """Current graph state.

    Domain-specific contracts above are the target architecture. The legacy
    flat fields below are retained temporarily so existing nodes continue to
    work while the migration is performed incrementally.

    Observability data such as traces and performance metrics intentionally do
    not belong in this state. They are request/session-level concerns and are
    collected outside LangGraph state.
    """

    # ---------------------------------------------------------
    # Legacy retrieval fields
    # ---------------------------------------------------------

    retrieval_top_score: float
    retrieval_second_score: float
    retrieval_score_gap: float
    retrieval_mean_score: float
    retrieval_top_to_mean_ratio: float
    retrieval_gap_ratio: float
    retrieval_overview_top_score: float | None
    retrieval_content_top_score: float | None
    retrieval_decision: str
    retrieval_evidence_strength: str
    retrieval_decision_reason: str

    # ---------------------------------------------------------
    # Legacy relevance grading
    # ---------------------------------------------------------

    relevance_grade: str

    # ---------------------------------------------------------
    # Legacy verification / retry fields
    # ---------------------------------------------------------

    hallucination_grade: str
    hallucination_retry_count: int
    grounding_diagnosis: str
    grounding_unsupported_claims: list[str]
    correction_attempted: bool
    retry_count: int


# =============================================================
# Public graph schemas
# =============================================================

class GraphInput(TypedDict, total=False):
    """Fields accepted from the API/UI when invoking the graph."""

    question: str
    document_id: str
    messages: Annotated[list, add_messages]


class GraphOutput(TypedDict, total=False):
    """Fields exposed to the caller after graph execution."""

    generation: str
    answer_status: str
    citations: list[str]