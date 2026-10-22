from __future__ import annotations

from langchain_core.documents import Document
import pytest

from agentic_rag.agents.evidence_contracts import (
    EvidenceToolContext,
    ResearchToolStep,
)
from agentic_rag.agents.evidence_selection import merge_and_select_evidence
from agentic_rag.agents.evidence_tools import (
    EvidenceToolScopeError,
    expand_evidence,
    inspect_document_structure,
    search_artifacts,
    search_document,
)
from agentic_rag.agents.observability import emit_research_event


def _document(
    record_id: str,
    *,
    document_id: str = "doc-1",
    kind: str = "text_block",
    page: str = "1",
    text: str | None = None,
) -> Document:
    return Document(
        page_content=text or f"evidence for {record_id}",
        metadata={
            "document_id": document_id,
            "artifact_id": f"artifact-{record_id}",
            "record_id": record_id,
            "artifact_kind": kind,
            "representation_kind": "native_text",
            "grounding_basis": "direct_text",
            "page_label": page,
            "type": "content",
        },
    )


class _Store:
    def __init__(self) -> None:
        self.structure_calls = []
        self.expansion_calls = []

    def get_ready_document_structure(self, **kwargs):
        self.structure_calls.append(kwargs)
        return [
            {
                "document_id": "doc-1",
                "record_id": "record-heading",
                "artifact_id": "artifact-heading",
                "artifact_kind": "text_block",
                "representation_kind": "native_text",
                "grounding_basis": "direct_text",
                "retrieval_schema_id": "schema-v3",
                "search_text": "3 Results",
                "locator": {"page_start": 4, "page_label": "4"},
                "parent_artifact_id": "page-4",
                "source_name": "paper.pdf",
                "artifact_metadata": {
                    "layout_class": "section-header",
                    "header_level": 1,
                },
                "record_metadata": {},
            }
        ]

    def get_ready_related_records(self, **kwargs):
        self.expansion_calls.append(kwargs)
        return [
            {
                "document_id": "doc-1",
                "record_id": "record-context",
                "artifact_id": "artifact-context",
                "artifact_kind": "text_block",
                "representation_kind": "native_text",
                "grounding_basis": "direct_text",
                "retrieval_schema_id": "schema-v3",
                "search_text": "The caption explains the architecture.",
                "locator": {"page_start": 2, "page_label": "2"},
                "parent_artifact_id": "page-2",
                "source_name": "paper.pdf",
                "artifact_metadata": {},
                "record_metadata": {},
            }
        ]


def _context() -> EvidenceToolContext:
    return EvidenceToolContext(
        document_id="doc-1",
        index_name="agentic-rag-hybrid-v3",
    )


def test_search_tools_keep_document_scope_application_owned() -> None:
    calls = []
    events = []

    def retriever(**kwargs):
        calls.append(kwargs)
        return [(_document("record-1"), 0.03)], {"rrf_count": 1}

    observation = search_document(
        context=_context(),
        step=ResearchToolStep(
            tool="search_document",
            query="attention complexity",
            evidence_requirement="text",
            purpose_code="facet_search",
        ),
        retriever=retriever,
        event_sink=lambda event, **fields: events.append((event, fields)),
    )

    assert calls == [{
        "query": "attention complexity",
        "document_id": "doc-1",
        "index_name": "agentic-rag-hybrid-v3",
    }]
    assert observation.candidates[0][0].metadata["record_id"] == "record-1"
    assert events[-1][1]["candidate_count"] == 1
    assert "attention complexity" not in repr(events)
    assert "doc-1" not in repr(events)


def test_artifact_search_derives_closed_kinds_from_requirement() -> None:
    calls = []

    def retriever(**kwargs):
        calls.append(kwargs)
        return [(_document("figure", kind="figure"), 0.02)], {}

    search_artifacts(
        context=_context(),
        step=ResearchToolStep(
            tool="search_artifacts",
            query="architecture figure",
            evidence_requirement="visual",
            purpose_code="artifact_lookup",
        ),
        retriever=retriever,
        event_sink=lambda *_args, **_kwargs: None,
    )

    assert calls[0]["document_id"] == "doc-1"
    assert calls[0]["index_name"] == "agentic-rag-hybrid-v3"
    assert calls[0]["artifact_kinds"] == frozenset(
        {"figure", "page_image", "standalone_image"}
    )


def test_tool_rejects_candidate_from_another_document() -> None:
    def retriever(**_kwargs):
        return [(_document("leak", document_id="doc-2"), 0.1)], {}

    with pytest.raises(EvidenceToolScopeError, match="document scope"):
        search_document(
            context=_context(),
            step=ResearchToolStep(
                tool="search_document",
                query="question",
                evidence_requirement="text",
                purpose_code="facet_search",
            ),
            retriever=retriever,
            event_sink=lambda *_args, **_kwargs: None,
        )


def test_structure_and_expansion_tools_use_ready_store_scope() -> None:
    store = _Store()
    structure = inspect_document_structure(
        context=_context(),
        step=ResearchToolStep(
            tool="inspect_document_structure",
            purpose_code="structure_lookup",
        ),
        store=store,
        event_sink=lambda *_args, **_kwargs: None,
    )
    expansion = expand_evidence(
        context=_context(),
        step=ResearchToolStep(
            tool="expand_evidence",
            evidence_ids=["record-figure"],
            purpose_code="context_expansion",
        ),
        store=store,
        event_sink=lambda *_args, **_kwargs: None,
    )

    assert store.structure_calls == [{
        "index_name": "agentic-rag-hybrid-v3",
        "document_id": "doc-1",
        "limit": 30,
    }]
    assert store.expansion_calls == [{
        "index_name": "agentic-rag-hybrid-v3",
        "document_id": "doc-1",
        "record_ids": ("record-figure",),
        "limit": 12,
    }]
    assert structure.candidates[0][0].page_content == "3 Results"
    assert structure.candidates[0][0].metadata["layout_class"] == "section-header"
    assert expansion.candidates[0][0].metadata["page_label"] == "2"


def test_global_selection_deduplicates_and_calls_reranker_once() -> None:
    shared = _document("shared", kind="figure", page="2")
    first = search_document(
        context=_context(),
        step=ResearchToolStep(
            tool="search_document",
            query="architecture",
            evidence_requirement="text",
            purpose_code="facet_search",
        ),
        retriever=lambda **_kwargs: (
            [(shared, 0.02), (_document("text", page="3"), 0.01)],
            {},
        ),
        event_sink=lambda *_args, **_kwargs: None,
    )
    second = search_artifacts(
        context=_context(),
        step=ResearchToolStep(
            tool="search_artifacts",
            query="architecture diagram",
            evidence_requirement="visual",
            purpose_code="artifact_lookup",
        ),
        retriever=lambda **_kwargs: (
            [(shared, 0.04), (_document("other-figure", kind="figure"), 0.03)],
            {},
        ),
        event_sink=lambda *_args, **_kwargs: None,
    )
    reranker_calls = []

    def reranker(query, candidate_groups, top_k_by_group):
        reranker_calls.append((query, candidate_groups, top_k_by_group))
        reranked = sorted(
            candidate_groups["content"],
            key=lambda item: item[0].metadata["record_id"],
        )
        return {"content": reranked}

    selection = merge_and_select_evidence(
        query="explain the architecture",
        observations=[first, second],
        evidence_requirement="visual",
        top_k=2,
        required_slots=1,
        companion_slots=0,
        reranker=reranker,
        event_sink=lambda *_args, **_kwargs: None,
    )

    assert len(reranker_calls) == 1
    assert selection.candidate_count == 3
    assert selection.duplicates_removed == 1
    assert selection.requirement_satisfied is True
    assert any(
        document.metadata["artifact_kind"] == "figure"
        for document, _ in selection.selected
    )


def test_research_event_schema_rejects_content_fields(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(
        "agentic_rag.core.logging.log_event",
        lambda event_type, **fields: captured.append((event_type, fields)),
    )
    emit_research_event(
        "tool_completed",
        tool="search_document",
        purpose_code="facet_search",
        candidate_count=2,
        elapsed_ms=3,
    )

    assert captured[0][0] == "evidence_agent"
    assert captured[0][1]["agent_event"] == "tool_completed"

    with pytest.raises(ValueError, match="content-free"):
        emit_research_event(
            "tool_completed",
            tool="search_document",
            query="secret question",
        )
