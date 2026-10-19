"""Exercise query planning nodes without models, retrieval, or live services."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from agentic_rag.core.timing import (
    PerformanceTracker,
    reset_current_tracker,
    set_current_tracker,
)


class ProviderUnavailableError(RuntimeError):
    pass


def _load_nodes(
    monkeypatch,
    *,
    router_content: str = "",
    general_content: str = "",
    fast_error: Exception | None = None,
):
    fast_calls = []
    primary_calls = []

    def fast_invoke(prompt, **kwargs):
        fast_calls.append((prompt, kwargs))
        if fast_error is not None:
            raise fast_error
        return SimpleNamespace(content=router_content)

    def primary_invoke(prompt, **kwargs):
        primary_calls.append((prompt, kwargs))
        return SimpleNamespace(content=general_content)

    stubs = {
        "agentic_rag.llm.provider": {
            "ProviderUnavailableError": ProviderUnavailableError,
            "provider_chain": SimpleNamespace(invoke=primary_invoke),
            "fast_provider_chain": SimpleNamespace(invoke=fast_invoke),
        },
        "agentic_rag.retrieval.vectorstore": {
            "build_query_representation": lambda *args, **kwargs: None,
            "retrieve_hybrid_with_scores": lambda *args, **kwargs: None,
        },
        "agentic_rag.retrieval.reranker": {
            "rerank_many": lambda *args, **kwargs: None,
        },
        "agentic_rag.retrieval.sparse": {
            "load_bm25_json": lambda *args, **kwargs: None,
        },
        "agentic_rag.ingestion.registry": {
            "get_bm25_params": lambda *args, **kwargs: None,
        },
    }
    for name, attributes in stubs.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).resolve().parents[2] / "src/agentic_rag/graph/nodes.py"
    spec = importlib.util.spec_from_file_location("query_nodes_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fast_calls, primary_calls


def _tracked_call(function, state):
    token = set_current_tracker(PerformanceTracker())
    try:
        return function(state)
    finally:
        reset_current_tracker(token)


def test_v3_request_uses_artifact_reader_and_existing_reranker(monkeypatch) -> None:
    nodes, _, _ = _load_nodes(monkeypatch)
    document = Document(
        page_content="diagram evidence",
        metadata={
            "document_id": "doc-1",
            "record_id": "record-1",
            "artifact_id": "artifact-1",
            "artifact_kind": "figure",
            "grounding_basis": "vision_description",
            "page_label": "2",
            "type": "content",
        },
    )
    reader_calls = []
    rerank_calls = []

    def retrieve_v3_document(**kwargs):
        reader_calls.append(kwargs)
        return [(document, 0.02)], {
            "dense_count": 1,
            "sparse_count": 1,
            "rrf_count": 1,
        }

    def rerank_many(query, groups, limits):
        rerank_calls.append((query, groups, limits))
        return {"content": [(document, 0.75)]}

    monkeypatch.setattr(nodes, "retrieve_v3_document", retrieve_v3_document)
    monkeypatch.setattr(nodes, "rerank_many", rerank_many)

    result = _tracked_call(
        nodes.retrieve,
        {
            "question": "Explain the figure",
            "document_id": "doc-1",
            "retrieval_version": "v3",
        },
    )

    assert reader_calls == [{
        "query": "Explain the figure",
        "document_id": "doc-1",
    }]
    assert rerank_calls[0][1] == {"content": [(document, 0.02)]}
    assert result["documents"] == [document]
    assert result["retrieval_scores"] == [0.75]
    assert result["retrieval_overview_top_score"] is None
    assert result["retrieval_content_top_score"] == 0.75


def test_v3_visual_request_keeps_visual_evidence_after_reranking(monkeypatch) -> None:
    nodes, _, _ = _load_nodes(monkeypatch)
    text = Document(
        page_content="surrounding explanation",
        metadata={
            "record_id": "text-1",
            "artifact_id": "artifact-text",
            "artifact_kind": "text_block",
            "page_label": "2",
            "parent_artifact_id": "page-2",
        },
    )
    figure = Document(
        page_content="figure description",
        metadata={
            "record_id": "figure-1",
            "artifact_id": "artifact-figure",
            "artifact_kind": "figure",
            "page_label": "2",
            "parent_artifact_id": "page-2",
        },
    )
    candidates = [(text, 0.02), (figure, 0.01)]
    reader_calls = []

    def retrieve_v3_document(**kwargs):
        reader_calls.append(kwargs)
        return candidates, {"dense_count": 2, "sparse_count": 2, "rrf_count": 2}

    def rerank_many(query, groups, limits):
        assert groups == {"v3_candidates": candidates}
        return {"v3_candidates": [(text, 0.95), (figure, 0.10)]}

    monkeypatch.setattr(nodes, "retrieve_v3_document", retrieve_v3_document)
    monkeypatch.setattr(nodes, "rerank_many", rerank_many)

    result = _tracked_call(
        nodes.retrieve,
        {
            "question": "Explain Figure 2",
            "document_id": "doc-1",
            "retrieval_version": "v3",
            "evidence_requirement": "visual",
        },
    )

    assert reader_calls == [{"query": "Explain Figure 2", "document_id": "doc-1"}]
    assert {doc.metadata["record_id"] for doc in result["documents"]} == {
        "text-1",
        "figure-1",
    }
    assert result["retrieval_evidence_requirement_satisfied"] is True


def test_are_you_sure_becomes_document_verification(monkeypatch) -> None:
    router_content = json.dumps(
        {
            "source": "document",
            "relationship": "verify_previous",
            "response_format": "prose",
            "standalone_query": "Transformer encoder and decoder architecture",
            "needs_clarification": False,
        }
    )
    nodes, fast_calls, _ = _load_nodes(monkeypatch, router_content=router_content)

    result = _tracked_call(
        nodes.contextualize_question,
        {
            "question": "Are you sure?",
            "source_mode": "document",
            "document_id": "doc-1",
            "document_name": "transformer.pdf",
            "messages": [
                HumanMessage(content="Explain the Transformer architecture"),
                AIMessage(content="It uses an encoder and decoder."),
            ],
        },
    )

    assert result["answer_source"] == "document"
    assert result["query_relationship"] == "verify_previous"
    assert result["information_need_source"] == "previous_information_need"
    assert result["retrieval_query"] == "Transformer encoder and decoder architecture"
    assert result["semantic_router_used"] is True
    assert fast_calls[0][1]["max_tokens"] == 256
    assert fast_calls[0][1]["groq_response_format"]["type"] == "json_schema"
    assert "selected_document_name:\n---\ntransformer.pdf" in fast_calls[0][0]


def test_malformed_semantic_plan_fails_closed_to_clarification(monkeypatch) -> None:
    nodes, _, _ = _load_nodes(monkeypatch, router_content="not-json")

    result = _tracked_call(
        nodes.contextualize_question,
        {
            "question": "Use whichever source is best",
            "source_mode": "auto",
            "document_id": "doc-1",
            "messages": [],
        },
    )

    assert result["answer_source"] == "clarify"
    assert result["routing_parse_success"] is False
    assert result["needs_clarification"] is True


def test_auto_router_cannot_bypass_selected_document_topic(monkeypatch) -> None:
    router_content = json.dumps(
        {
            "source": "general",
            "relationship": "standalone",
            "response_format": "prose",
            "standalone_query": None,
            "needs_clarification": False,
            "information_need_source": "current_turn",
            "evidence_requirement": "text",
        }
    )
    nodes, fast_calls, _ = _load_nodes(monkeypatch, router_content=router_content)
    question = (
        "Why is dot-product attention divided by the square root of the key "
        "dimension?"
    )

    result = _tracked_call(
        nodes.contextualize_question,
        {
            "question": question,
            "source_mode": "auto",
            "document_id": "doc-1",
            "document_name": "AttentionIsAllYouNeed.pdf",
            "messages": [],
        },
    )

    assert len(fast_calls) == 1
    assert result["answer_source"] == "document"
    assert result["retrieval_query"] == question


def test_general_generation_skips_documents_and_grounding(monkeypatch) -> None:
    nodes, fast_calls, primary_calls = _load_nodes(
        monkeypatch,
        general_content="Paris is the capital of France.",
    )

    result = _tracked_call(
        nodes.generate_general_answer,
        {
            "question": "What is the capital of France?",
            "response_format": "prose",
            "messages": [],
        },
    )

    assert fast_calls == []
    assert len(primary_calls) == 1
    assert result["answer_source"] == "general"
    assert result["answer_status"] == "general_answer"
    assert result["documents"] == []
    assert result["citations"] == []
    assert result["hallucination_grade"] is None


def test_document_visual_request_preserves_the_current_information_need(
    monkeypatch,
) -> None:
    router_content = json.dumps(
        {
            "source": "document",
            "relationship": "follow_up",
            # The planner may confuse an existing artifact with output format;
            # deterministic enforcement must still preserve the lookup.
            "response_format": "ascii_flowchart",
            "standalone_query": (
                "FlashAttention document diagram explaining architectural changes"
            ),
            "needs_clarification": False,
            # The deterministic guard must reject this unsafe model choice.
            "information_need_source": "previous_information_need",
        }
    )
    nodes, fast_calls, _ = _load_nodes(monkeypatch, router_content=router_content)

    result = _tracked_call(
        nodes.contextualize_question,
        {
            "question": (
                "Is there a diagram in the document that explains the "
                "architectural changes?"
            ),
            "source_mode": "document",
            "document_id": "flash-doc",
            "document_name": "FlashAttention.pdf",
            "messages": [
                HumanMessage(content="Explain the architectural difference."),
                AIMessage(content="It changes attention memory access."),
            ],
        },
    )

    assert result["information_need_source"] == "current_turn"
    assert result["response_format"] == "requested"
    assert result["retrieval_query"] == (
        "FlashAttention document diagram explaining architectural changes"
    )
    assert len(fast_calls) == 1


def test_relevance_grader_uses_bounded_structured_groq_call(monkeypatch) -> None:
    nodes, fast_calls, _ = _load_nodes(
        monkeypatch,
        router_content='{"verdict":"relevant"}',
    )

    result = _tracked_call(
        nodes.grade_documents,
        {
            "question": "How does FlashAttention work?",
            "retrieval_query": "FlashAttention algorithm stages",
            "documents": [
                Document(
                    page_content="FlashAttention computes attention in blocks.",
                    metadata={"filename": "flash.pdf", "type": "content"},
                )
            ],
            "retrieval_scores": [0.017],
        },
    )

    assert result["relevance_grade"] == "relevant"
    assert fast_calls[0][1]["max_tokens"] == 512
    assert fast_calls[0][1]["groq_reasoning_effort"] == "low"
    assert fast_calls[0][1]["groq_response_format"]["type"] == "json_schema"
    assert fast_calls[0][1]["require_nonempty_content"] is True


def test_unavailable_relevance_grader_is_uncertain_not_irrelevant(monkeypatch) -> None:
    nodes, _, _ = _load_nodes(
        monkeypatch,
        fast_error=ProviderUnavailableError("empty provider response"),
    )

    result = _tracked_call(
        nodes.grade_documents,
        {
            "question": "How does FlashAttention work?",
            "documents": [
                Document(page_content="Evidence", metadata={"type": "content"})
            ],
            "retrieval_scores": [0.017],
        },
    )

    assert result["relevance_grade"] == "uncertain"


def test_counterevidence_is_answerable_and_not_rewritten(monkeypatch) -> None:
    nodes, fast_calls, _ = _load_nodes(
        monkeypatch,
        router_content='{"verdict":"contradicted"}',
    )

    result = _tracked_call(
        nodes.grade_documents,
        {
            "question": "Why does DPO require a separate reward model?",
            "documents": [
                Document(
                    page_content=(
                        "DPO optimizes a policy directly without fitting an "
                        "explicit reward model."
                    ),
                    metadata={"filename": "dpo.pdf", "type": "content"},
                )
            ],
            "retrieval_scores": [0.41],
        },
    )

    prompt = fast_calls[0][0]
    assert "correct a false premise" in prompt
    assert result["relevance_grade"] == "contradicted"


def test_generation_explicitly_corrects_a_contradicted_premise(monkeypatch) -> None:
    nodes, _, primary_calls = _load_nodes(
        monkeypatch,
        general_content="DPO does not require a separate reward model [S1].",
    )

    _tracked_call(
        nodes.generate,
        {
            "question": "Why does DPO require a separate reward model?",
            "response_format": "prose",
            "relevance_grade": "contradicted",
            "documents": [
                Document(
                    page_content="DPO avoids fitting an explicit reward model.",
                    metadata={"filename": "dpo.pdf", "page": 3},
                )
            ],
            "messages": [],
        },
    )

    prompt = primary_calls[0][0]
    assert "Do not adopt the question's false premise" in prompt


def test_grounding_retry_uses_bounded_rewriter_and_search_only_terms(monkeypatch) -> None:
    nodes, fast_calls, _ = _load_nodes(
        monkeypatch,
        router_content="FlashAttention algorithm stages and data flow",
    )

    result = _tracked_call(
        nodes.rewrite_query,
        {
            "question": "Explain FlashAttention as a flowchart",
            "retrieval_query": "Explain FlashAttention",
            "grounding_diagnosis": "insufficient_evidence",
            "retry_count": 0,
        },
    )

    prompt, options = fast_calls[0]
    assert "grounding_insufficient_evidence" in prompt
    assert "remove presentation requests" in prompt
    assert options == {
        "stage": "query_rewrite",
        "max_tokens": 512,
        "groq_reasoning_effort": "low",
        "require_nonempty_content": True,
    }
    assert result["retrieval_query"] == "FlashAttention algorithm stages and data flow"
    assert result["rewrite_stalled"] is False


def test_corrective_generation_preserves_math_format_contract(monkeypatch) -> None:
    nodes, _, primary_calls = _load_nodes(
        monkeypatch,
        general_content="Corrected result [S1].",
    )

    result = _tracked_call(
        nodes.correct_generation,
        {
            "question": "Explain the objective mathematically",
            "response_format": "math",
            "documents": [
                Document(
                    page_content="The objective combines reward and a KL penalty.",
                    metadata={"filename": "paper.pdf", "page": 3},
                )
            ],
            "messages": [],
            "generation": "Unsupported equation.",
            "grounding_unsupported_claims": ["The equation is unsupported"],
        },
    )

    prompt = primary_calls[0][0]
    assert "Formatting rules:" in prompt
    assert "valid LaTeX inside $$ delimiters" in prompt
    assert "plain language" in prompt
    assert "Make the smallest complete repair" in prompt
    assert "Do not invent a replacement reason" in prompt
    assert "the supplied excerpts do not establish it" in prompt
    assert result["correction_attempted"] is True


def test_generation_forbids_unproved_motivation_and_exhaustive_claims(
    monkeypatch,
) -> None:
    nodes, _, primary_calls = _load_nodes(
        monkeypatch,
        general_content="LoRA adapts two low-rank matrices [S1].",
    )

    result = _tracked_call(
        nodes.generate,
        {
            "question": "Why are two matrices adapted?",
            "response_format": "prose",
            "documents": [
                Document(
                    page_content="LoRA adapts two low-rank matrices.",
                    metadata={"filename": "lora.pdf", "page": 5},
                )
            ],
            "messages": [],
        },
    )

    prompt = primary_calls[0][0]
    assert "Do not infer author intent, motivation, causality" in prompt
    assert "instead of adopting the false premise" in prompt
    assert "maximality" in prompt
    assert "absence of material from the whole document" in prompt
    assert result["citations"] == ["[S1] lora.pdf, page 5"]
