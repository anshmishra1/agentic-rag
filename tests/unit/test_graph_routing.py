from agentic_rag.graph.edges import (
    route_after_contextualization,
    route_after_grading,
    route_after_hallucination_check,
    route_after_retrieval_assessment,
)


def test_control_turn_bypasses_retrieval() -> None:
    assert route_after_contextualization({"query_intent": "control"}) == "record_turn"


def test_question_continues_to_retrieval() -> None:
    assert route_after_contextualization({"query_intent": "follow_up"}) == "retrieve"


def test_retrieval_assessment_preserves_known_decisions() -> None:
    assert route_after_retrieval_assessment({"retrieval_decision": "generate"}) == "generate"
    assert route_after_retrieval_assessment({"retrieval_decision": "rewrite_query"}) == "rewrite_query"
    assert route_after_retrieval_assessment({"retrieval_decision": "grade"}) == "grade_documents"


def test_relevant_documents_generate() -> None:
    assert route_after_grading(
        {"relevance_grade": "relevant", "retry_count": 0}
    ) == "generate"


def test_irrelevant_documents_rewrite_until_budget_is_exhausted(
    monkeypatch,
) -> None:
    monkeypatch.setattr("agentic_rag.graph.edges.settings.max_retries", 2)

    assert route_after_grading(
        {"relevance_grade": "irrelevant", "retry_count": 0}
    ) == "rewrite_query"
    assert route_after_grading(
        {"relevance_grade": "irrelevant", "retry_count": 2}
    ) == "generate"


def test_grounded_answer_ends() -> None:
    assert route_after_hallucination_check(
        {"grounding_diagnosis": "grounded"}
    ) == "end"


def test_insufficient_evidence_retries_retrieval(
    monkeypatch,
) -> None:
    monkeypatch.setattr("agentic_rag.graph.edges.settings.max_retries", 2)

    assert route_after_hallucination_check(
        {
            "grounding_diagnosis": "insufficient_evidence",
            "retry_count": 0,
            "correction_attempted": False,
        }
    ) == "rewrite_query"


def test_unsupported_answer_uses_single_correction() -> None:
    assert route_after_hallucination_check(
        {
            "grounding_diagnosis": "unsupported",
            "retry_count": 0,
            "correction_attempted": False,
        }
    ) == "correct_generation"
    assert route_after_hallucination_check(
        {
            "grounding_diagnosis": "unsupported",
            "retry_count": 0,
            "correction_attempted": True,
        }
    ) == "end"
