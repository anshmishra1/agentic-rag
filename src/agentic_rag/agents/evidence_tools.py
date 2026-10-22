"""Read-only, document-scoped tools for the evidence-research agent."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from langchain_core.documents import Document

from agentic_rag.agents.evidence_contracts import (
    EvidenceToolContext,
    EvidenceToolObservation,
    ResearchToolStep,
)
from agentic_rag.agents.observability import emit_research_event
from agentic_rag.policies.evidence import required_artifact_kinds


class EvidenceToolScopeError(ValueError):
    """A storage adapter returned evidence outside the injected scope."""


class ResearchEvidenceStore(Protocol):
    def get_ready_document_structure(
        self,
        *,
        index_name: str,
        document_id: str,
        limit: int,
    ) -> list[dict[str, Any]]: ...

    def get_ready_related_records(
        self,
        *,
        index_name: str,
        document_id: str,
        record_ids: tuple[str, ...],
        limit: int,
    ) -> list[dict[str, Any]]: ...


Retriever = Callable[..., tuple[list[tuple[Document, float]], dict[str, Any]]]
EventSink = Callable[..., None]


def _default_retriever(**kwargs):
    from agentic_rag.retrieval.v3_reader import retrieve_v3_document

    return retrieve_v3_document(**kwargs)


def _default_store() -> ResearchEvidenceStore:
    from agentic_rag.ingestion.manifest_store import PostgresManifestStore

    return PostgresManifestStore()


def _validate_step(step: ResearchToolStep, expected: str) -> None:
    if step.tool != expected:
        raise ValueError(f"Expected {expected} step, received {step.tool}")


def _validate_candidates(
    context: EvidenceToolContext,
    candidates: Sequence[tuple[Document, float]],
) -> tuple[tuple[Document, float], ...]:
    validated: list[tuple[Document, float]] = []
    for document, score in candidates:
        metadata = document.metadata or {}
        if str(metadata.get("document_id") or "") != context.document_id:
            raise EvidenceToolScopeError(
                "Evidence tool returned a candidate outside document scope"
            )
        if not str(metadata.get("record_id") or "").strip():
            raise EvidenceToolScopeError("Evidence candidate is missing record_id")
        if not str(metadata.get("artifact_id") or "").strip():
            raise EvidenceToolScopeError("Evidence candidate is missing artifact_id")
        if not document.page_content.strip():
            raise EvidenceToolScopeError("Evidence candidate has empty content")
        validated.append((document, float(score)))
    return tuple(validated)


def _row_to_document(
    context: EvidenceToolContext,
    row: dict[str, Any],
) -> Document:
    if str(row.get("document_id") or "") != context.document_id:
        raise EvidenceToolScopeError(
            "Evidence store returned a record outside document scope"
        )
    search_text = str(row.get("search_text") or "").strip()
    record_id = str(row.get("record_id") or "").strip()
    artifact_id = str(row.get("artifact_id") or "").strip()
    if not search_text or not record_id or not artifact_id:
        raise EvidenceToolScopeError("Evidence store returned an incomplete record")

    locator = dict(row.get("locator") or {})
    artifact_metadata = dict(row.get("artifact_metadata") or {})
    metadata = {
        "index_name": context.index_name,
        "source": str(row.get("source_name") or ""),
        "filename": str(row.get("source_name") or ""),
        "document_id": context.document_id,
        "artifact_id": artifact_id,
        "record_id": record_id,
        "artifact_kind": str(row.get("artifact_kind") or ""),
        "representation_kind": str(row.get("representation_kind") or ""),
        "grounding_basis": str(row.get("grounding_basis") or ""),
        "retrieval_schema_id": str(row.get("retrieval_schema_id") or ""),
        "type": "content",
    }
    optional = {
        "parent_artifact_id": row.get("parent_artifact_id"),
        "page_start": locator.get("page_start"),
        "page_end": locator.get("page_end"),
        "page_label": locator.get("page_label"),
        "start_ms": locator.get("start_ms"),
        "end_ms": locator.get("end_ms"),
        "layout_class": artifact_metadata.get("layout_class"),
        "header_level": artifact_metadata.get("header_level"),
    }
    metadata.update({key: value for key, value in optional.items() if value is not None})
    bbox = locator.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        metadata.update(
            dict(zip(
                ("bbox_left", "bbox_top", "bbox_right", "bbox_bottom"),
                bbox,
            ))
        )
    return Document(page_content=search_text, metadata=metadata)


def _emit_failure(
    event_sink: EventSink,
    *,
    tool: str,
    purpose_code: str,
    started: float,
    error: Exception,
) -> None:
    event_sink(
        "tool_failed",
        tool=tool,
        purpose_code=purpose_code,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
        error_type=type(error).__name__,
    )


def search_document(
    *,
    context: EvidenceToolContext,
    step: ResearchToolStep,
    retriever: Retriever = _default_retriever,
    event_sink: EventSink = emit_research_event,
) -> EvidenceToolObservation:
    """Retrieve ordinary V3 candidates from one application-selected document."""

    _validate_step(step, "search_document")
    started = time.perf_counter()
    event_sink(
        "tool_started",
        tool=step.tool,
        purpose_code=step.purpose_code,
        evidence_requirement=step.evidence_requirement,
    )
    try:
        candidates, diagnostics = retriever(
            query=step.query,
            document_id=context.document_id,
            index_name=context.index_name,
        )
        validated = _validate_candidates(context, candidates)
    except Exception as error:
        _emit_failure(
            event_sink,
            tool=step.tool,
            purpose_code=step.purpose_code,
            started=started,
            error=error,
        )
        raise
    event_sink(
        "tool_completed",
        tool=step.tool,
        purpose_code=step.purpose_code,
        evidence_requirement=step.evidence_requirement,
        candidate_count=len(validated),
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )
    return EvidenceToolObservation(
        tool=step.tool,
        purpose_code=step.purpose_code,
        candidates=validated,
        diagnostics=dict(diagnostics),
    )


def search_artifacts(
    *,
    context: EvidenceToolContext,
    step: ResearchToolStep,
    retriever: Retriever = _default_retriever,
    event_sink: EventSink = emit_research_event,
) -> EvidenceToolObservation:
    """Search the closed V3 figure/table/formula artifact lanes."""

    _validate_step(step, "search_artifacts")
    artifact_kinds = required_artifact_kinds(step.evidence_requirement)
    if not artifact_kinds:
        raise ValueError("search_artifacts requires structured evidence")
    started = time.perf_counter()
    event_sink(
        "tool_started",
        tool=step.tool,
        purpose_code=step.purpose_code,
        evidence_requirement=step.evidence_requirement,
    )
    try:
        candidates, diagnostics = retriever(
            query=step.query,
            document_id=context.document_id,
            index_name=context.index_name,
            artifact_kinds=artifact_kinds,
        )
        validated = _validate_candidates(context, candidates)
    except Exception as error:
        _emit_failure(
            event_sink,
            tool=step.tool,
            purpose_code=step.purpose_code,
            started=started,
            error=error,
        )
        raise
    event_sink(
        "tool_completed",
        tool=step.tool,
        purpose_code=step.purpose_code,
        evidence_requirement=step.evidence_requirement,
        candidate_count=len(validated),
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )
    return EvidenceToolObservation(
        tool=step.tool,
        purpose_code=step.purpose_code,
        candidates=validated,
        diagnostics=dict(diagnostics),
    )


def inspect_document_structure(
    *,
    context: EvidenceToolContext,
    step: ResearchToolStep,
    store: ResearchEvidenceStore | None = None,
    limit: int = 30,
    event_sink: EventSink = emit_research_event,
) -> EvidenceToolObservation:
    """Read ready headings and page summaries without vector retrieval."""

    _validate_step(step, "inspect_document_structure")
    started = time.perf_counter()
    event_sink(
        "tool_started",
        tool=step.tool,
        purpose_code=step.purpose_code,
        evidence_requirement=step.evidence_requirement,
    )
    store = store or _default_store()
    try:
        rows = store.get_ready_document_structure(
            index_name=context.index_name,
            document_id=context.document_id,
            limit=limit,
        )
        candidates = _validate_candidates(
            context,
            [(_row_to_document(context, row), 0.0) for row in rows],
        )
    except Exception as error:
        _emit_failure(
            event_sink,
            tool=step.tool,
            purpose_code=step.purpose_code,
            started=started,
            error=error,
        )
        raise
    event_sink(
        "tool_completed",
        tool=step.tool,
        purpose_code=step.purpose_code,
        evidence_requirement=step.evidence_requirement,
        structure_count=len(candidates),
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )
    return EvidenceToolObservation(
        tool=step.tool,
        purpose_code=step.purpose_code,
        candidates=candidates,
        diagnostics={"structure_count": len(candidates)},
    )


def expand_evidence(
    *,
    context: EvidenceToolContext,
    step: ResearchToolStep,
    store: ResearchEvidenceStore | None = None,
    limit: int = 12,
    event_sink: EventSink = emit_research_event,
) -> EvidenceToolObservation:
    """Resolve bounded same-page and parent context for prior evidence IDs."""

    _validate_step(step, "expand_evidence")
    started = time.perf_counter()
    event_sink(
        "tool_started",
        tool=step.tool,
        purpose_code=step.purpose_code,
        evidence_requirement=step.evidence_requirement,
    )
    store = store or _default_store()
    try:
        rows = store.get_ready_related_records(
            index_name=context.index_name,
            document_id=context.document_id,
            record_ids=step.evidence_ids,
            limit=limit,
        )
        candidates = _validate_candidates(
            context,
            [(_row_to_document(context, row), 0.0) for row in rows],
        )
    except Exception as error:
        _emit_failure(
            event_sink,
            tool=step.tool,
            purpose_code=step.purpose_code,
            started=started,
            error=error,
        )
        raise
    event_sink(
        "tool_completed",
        tool=step.tool,
        purpose_code=step.purpose_code,
        evidence_requirement=step.evidence_requirement,
        candidate_count=len(candidates),
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )
    return EvidenceToolObservation(
        tool=step.tool,
        purpose_code=step.purpose_code,
        candidates=candidates,
        diagnostics={"expanded_count": len(candidates)},
    )

