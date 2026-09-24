import ast
from pathlib import Path

from agentic_rag.graph.state import RAGState


EXPECTED_STATE_FIELDS = {
    "question",
    "document_id",
    "messages",
    "retrieval_query",
    "query_intent",
    "query_is_control",
    "contextualization_used",
    "documents",
    "retrieval_scores",
    "retrieval_top_score",
    "retrieval_second_score",
    "retrieval_score_gap",
    "retrieval_mean_score",
    "retrieval_top_to_mean_ratio",
    "retrieval_gap_ratio",
    "retrieval_overview_top_score",
    "retrieval_content_top_score",
    "retrieval_decision",
    "retrieval_evidence_strength",
    "retrieval_decision_reason",
    "relevance_grade",
    "generation",
    "answer_status",
    "hallucination_grade",
    "grounding_diagnosis",
    "grounding_unsupported_claims",
    "grounding_parse_success",
    "verification_exhausted",
    "citations",
    "retry_count",
    "hallucination_retry_count",
    "correction_attempted",
}


def test_rag_state_declares_only_the_flat_runtime_contract() -> None:
    assert set(RAGState.__annotations__) == EXPECTED_STATE_FIELDS


def test_record_turn_does_not_write_trace_to_checkpoint_state() -> None:
    nodes_path = (
        Path(__file__).parents[2]
        / "src"
        / "agentic_rag"
        / "graph"
        / "nodes.py"
    )
    module = ast.parse(nodes_path.read_text(encoding="utf-8"))
    record_turn = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "record_turn"
    )

    trace_literals = [
        node
        for node in ast.walk(record_turn)
        if isinstance(node, ast.Constant) and node.value == "trace"
    ]

    assert trace_literals == []
