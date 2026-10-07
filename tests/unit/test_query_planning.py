import json

from langchain_core.messages import AIMessage, HumanMessage

from agentic_rag.policies.query_planning import (
    QueryPlan,
    build_query_plan_prompt,
    direct_query_plan,
    enforce_query_plan,
    groq_query_plan_response_format,
    parse_query_plan,
)


HISTORY = [
    HumanMessage(content="What is the Transformer architecture?"),
    AIMessage(content="It uses encoder and decoder stacks."),
]


def test_closed_control_turn_needs_no_semantic_router() -> None:
    plan = direct_query_plan("Thanks!", HISTORY, "auto")

    assert plan is not None
    assert plan.source == "control"


def test_first_explicit_document_question_goes_directly_to_retrieval() -> None:
    plan = direct_query_plan("Explain self-attention", [], "document")

    assert plan is not None
    assert plan.source == "document"
    assert plan.relationship == "standalone"
    assert plan.standalone_query == "Explain self-attention"


def test_document_conversation_uses_semantic_router() -> None:
    assert direct_query_plan("Are you sure?", HISTORY, "document") is None


def test_explicit_general_mode_skips_semantic_router_and_retrieval() -> None:
    plan = direct_query_plan("What is the capital of France?", HISTORY, "general")

    assert plan is not None
    assert plan.source == "general"
    assert plan.standalone_query == "What is the capital of France?"


def test_structured_verification_plan_is_parsed_and_preserved() -> None:
    content = json.dumps(
        {
            "source": "document",
            "relationship": "verify_previous",
            "response_format": "prose",
            "standalone_query": "Transformer encoder and decoder architecture",
            "needs_clarification": False,
        }
    )

    plan = enforce_query_plan(
        parse_query_plan(f"```json\n{content}\n```"),
        question="Are you sure?",
        history=HISTORY,
        source_mode="document",
        document_selected=True,
    )

    assert plan.relationship == "verify_previous"
    assert plan.standalone_query == "Transformer encoder and decoder architecture"


def test_explicit_source_mode_overrides_semantic_source_choice() -> None:
    plan = QueryPlan(
        source="general",
        relationship="standalone",
        response_format="requested",
        standalone_query=None,
        needs_clarification=False,
    )

    enforced = enforce_query_plan(
        plan,
        question="Explain attention",
        history=HISTORY,
        source_mode="document",
        document_selected=True,
    )

    assert enforced.source == "document"
    assert enforced.standalone_query == "Explain attention"


def test_document_route_without_selected_document_requests_clarification() -> None:
    plan = QueryPlan(
        source="document",
        relationship="standalone",
        response_format="prose",
        standalone_query="Explain the paper",
        needs_clarification=False,
    )

    enforced = enforce_query_plan(
        plan,
        question="Explain the paper",
        history=[],
        source_mode="auto",
        document_selected=False,
    )

    assert enforced.source == "clarify"
    assert enforced.needs_clarification is True


def test_router_prompt_separates_source_relationship_and_format() -> None:
    prompt = build_query_plan_prompt(
        "Show that as a textual flowchart",
        HISTORY,
        "auto",
        document_selected=True,
    )

    assert "source=document" in prompt
    assert "relationship=verify_previous" in prompt
    assert "response_format" in prompt
    assert "Show that as a textual flowchart" in prompt


def test_groq_router_schema_is_strict_and_closed() -> None:
    response_format = groq_query_plan_response_format()
    schema = response_format["json_schema"]["schema"]

    assert response_format["json_schema"]["strict"] is True
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "source",
        "relationship",
        "response_format",
        "standalone_query",
        "needs_clarification",
    }
