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


def _load_nodes(monkeypatch, *, router_content: str = "", general_content: str = ""):
    fast_calls = []
    primary_calls = []

    def fast_invoke(prompt, **kwargs):
        fast_calls.append((prompt, kwargs))
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
            "messages": [
                HumanMessage(content="Explain the Transformer architecture"),
                AIMessage(content="It uses an encoder and decoder."),
            ],
        },
    )

    assert result["answer_source"] == "document"
    assert result["query_relationship"] == "verify_previous"
    assert result["retrieval_query"] == "Transformer encoder and decoder architecture"
    assert result["semantic_router_used"] is True
    assert fast_calls[0][1]["max_tokens"] == 256
    assert fast_calls[0][1]["groq_response_format"]["type"] == "json_schema"


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
    assert result["correction_attempted"] is True
