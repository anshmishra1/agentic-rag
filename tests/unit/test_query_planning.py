import json

from langchain_core.messages import AIMessage, HumanMessage

from agentic_rag.policies.query_planning import (
    QueryPlan,
    build_query_plan_prompt,
    detect_response_format,
    direct_query_plan,
    enforce_query_plan,
    groq_query_plan_response_format,
    is_document_artifact_lookup,
    is_presentation_only_follow_up,
    parse_query_plan,
    retrieval_query_without_format_request,
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


def test_first_document_flowchart_request_keeps_format_out_of_retrieval() -> None:
    plan = direct_query_plan(
        "Explain FlashAttention as a textual flowchart",
        [],
        "document",
    )

    assert plan is not None
    assert plan.response_format == "ascii_flowchart"
    assert plan.standalone_query == "Explain FlashAttention"


def test_explicit_format_detection_is_limited_to_presentation() -> None:
    assert detect_response_format("Show the objective using LaTeX") == "math"
    assert detect_response_format("Compare the methods in a table") == "table"
    assert detect_response_format("Create a diagram explaining FlashAttention") == "ascii_flowchart"
    assert (
        detect_response_format("Give me a flowchart explaining attention")
        == "ascii_flowchart"
    )
    assert detect_response_format("Is there a diagram in the document?") == "requested"
    assert detect_response_format("Does the paper contain a comparison table?") == "requested"
    assert detect_response_format("Which equation does the paper use?") == "requested"
    assert detect_response_format("Explain the method") == "requested"
    assert (
        retrieval_query_without_format_request(
            "Compare the methods in a table"
        )
        == "Compare the methods"
    )


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


def test_explicit_flowchart_format_overrides_plan_and_cleans_search_query() -> None:
    plan = QueryPlan(
        source="document",
        relationship="standalone",
        response_format="prose",
        standalone_query="Explain FlashAttention as a flowchart",
        needs_clarification=False,
    )

    enforced = enforce_query_plan(
        plan,
        question="Explain FlashAttention as a flowchart",
        history=[],
        source_mode="document",
        document_selected=True,
    )

    assert enforced.response_format == "ascii_flowchart"
    assert enforced.standalone_query == "Explain FlashAttention"


def test_anaphoric_format_followup_reuses_prior_information_need() -> None:
    plan = QueryPlan(
        source="document",
        relationship="standalone",
        response_format="ascii_flowchart",
        standalone_query="Represent that as a textual flowchart",
        needs_clarification=False,
        information_need_source="previous_information_need",
    )

    enforced = enforce_query_plan(
        plan,
        question="Represent that as a textual flowchart",
        history=HISTORY,
        source_mode="document",
        document_selected=True,
    )

    assert enforced.relationship == "follow_up"
    assert enforced.information_need_source == "previous_information_need"
    assert enforced.response_format == "ascii_flowchart"
    assert enforced.standalone_query == "What is the Transformer architecture?"


def test_previous_information_need_requires_a_pure_presentation_request() -> None:
    question = (
        "Is there a diagram in the document that explains the architectural changes?"
    )
    plan = QueryPlan(
        source="document",
        relationship="follow_up",
        response_format="requested",
        standalone_query="FlashAttention document diagram architectural changes",
        needs_clarification=False,
        information_need_source="previous_information_need",
    )

    enforced = enforce_query_plan(
        plan,
        question=question,
        history=HISTORY,
        source_mode="document",
        document_selected=True,
    )

    assert enforced.information_need_source == "current_turn"
    assert enforced.response_format == "requested"
    assert enforced.standalone_query == (
        "FlashAttention document diagram architectural changes"
    )


def test_presentation_only_guard_is_anchored_to_the_complete_request() -> None:
    assert is_presentation_only_follow_up("Put that in a flowchart.")
    assert is_presentation_only_follow_up(
        "Could you represent the previous answer as a textual diagram?"
    )
    assert not is_presentation_only_follow_up(
        "Is there a diagram in the document that explains the architecture?"
    )
    assert not is_presentation_only_follow_up("Can you explain that diagram?")


def test_document_artifact_lookup_cannot_be_forced_into_output_formatting() -> None:
    question = "Is there a diagram in the document that explains the architecture?"
    assert is_document_artifact_lookup(question)
    assert is_document_artifact_lookup(
        "Does the paper include a table comparing the methods?"
    )
    assert not is_document_artifact_lookup(
        "Create a diagram explaining the architecture in the paper."
    )

    plan = QueryPlan(
        source="document",
        relationship="follow_up",
        response_format="ascii_flowchart",
        standalone_query="document diagram explaining the architecture",
        needs_clarification=False,
    )
    enforced = enforce_query_plan(
        plan,
        question=question,
        history=HISTORY,
        source_mode="document",
        document_selected=True,
    )
    assert enforced.response_format == "requested"


def test_legacy_plan_defaults_to_the_current_information_need() -> None:
    plan = parse_query_plan(
        json.dumps(
            {
                "source": "document",
                "relationship": "standalone",
                "response_format": "prose",
                "standalone_query": "Explain attention",
                "needs_clarification": False,
            }
        )
    )

    assert plan.information_need_source == "current_turn"


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
        document_name="Prompt_Engineering_For_LLM_OReilly.pdf",
    )

    assert "source=document" in prompt
    assert "relationship=verify_previous" in prompt
    assert "response_format" in prompt
    assert "Remove presentation instructions" in prompt
    assert "selected_document_name:\n---\nPrompt_Engineering_For_LLM_OReilly.pdf" in prompt
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
        "information_need_source",
    }
