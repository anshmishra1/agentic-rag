"""Exercise the actual grounding node without loading models or calling services."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from langchain_core.documents import Document

from agentic_rag.core.timing import (
    PerformanceTracker,
    reset_current_tracker,
    set_current_tracker,
)


def _load_node(monkeypatch, content: str, metadata: dict):
    calls = []

    def invoke(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return SimpleNamespace(
            content=content,
            response_metadata=metadata,
            additional_kwargs={"reasoning_content": "private reasoning text"},
        )

    stubs = {
        "agentic_rag.llm.provider": {
            "provider_chain": SimpleNamespace(),
            "fast_provider_chain": SimpleNamespace(invoke=invoke),
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
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).resolve().parents[2] / "src/agentic_rag/graph/nodes.py"
    spec = importlib.util.spec_from_file_location("verifier_node_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_hallucination, calls


def _state() -> dict:
    return {
        "question": "What does the course include?",
        "generation": "It includes supervised learning [S1].",
        "documents": [
            Document(
                page_content="The course includes supervised learning.",
                metadata={"filename": "fixture.pdf", "page_label": "2"},
            )
        ],
        "correction_attempted": False,
    }


def test_verifier_node_uses_dedicated_budget_and_low_effort(monkeypatch):
    check, calls = _load_node(
        monkeypatch,
        '{"verdict":"grounded","unsupported_claims":[]}',
        {"finish_reason": "stop"},
    )
    token = set_current_tracker(PerformanceTracker())
    try:
        result = check(_state())
    finally:
        reset_current_tracker(token)

    assert result["answer_status"] == "answered"
    assert len(calls) == 1
    assert "[S1] fixture.pdf, page 2" in calls[0][0]
    assert calls[0][1] == {
        "max_tokens": 1024,
        "groq_reasoning_effort": "low",
    }


def test_empty_verifier_still_fails_closed_and_logs_only_token_counts(
    monkeypatch, capsys
):
    check, calls = _load_node(
        monkeypatch,
        "",
        {
            "finish_reason": "length",
            "token_usage": {
                "completion_tokens": 512,
                "completion_tokens_details": {"reasoning_tokens": 510},
            },
        },
    )
    token = set_current_tracker(PerformanceTracker())
    try:
        result = check(_state())
    finally:
        reset_current_tracker(token)

    printed = capsys.readouterr().out
    assert len(calls) == 1
    assert result["answer_status"] == "verification_uncertain"
    assert result["grounding_parse_success"] is False
    assert "reasoning_tokens': 510" in printed
    assert "private reasoning text" not in printed
