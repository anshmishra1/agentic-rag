import pytest
from pydantic import ValidationError

from agentic_rag.agents.evidence_contracts import (
    EvidenceToolContext,
    ResearchPlan,
    ResearchToolStep,
)


def _step(**overrides) -> ResearchToolStep:
    values = {
        "tool": "search_document",
        "query": "attention complexity",
        "evidence_requirement": "text",
        "purpose_code": "facet_search",
    }
    values.update(overrides)
    return ResearchToolStep(**values)


def test_research_plan_is_small_closed_and_whitespace_normalized() -> None:
    plan = ResearchPlan(
        objective="  Explain the architecture and its limitations. ",
        required_facets=[" architecture ", "limitations"],
        steps=[
            _step(),
            _step(
                tool="search_artifacts",
                query="architecture diagram",
                evidence_requirement="visual",
                purpose_code="artifact_lookup",
            ),
        ],
    )

    assert plan.objective == "Explain the architecture and its limitations."
    assert plan.required_facets == ("architecture", "limitations")
    assert len(plan.steps) == 2

    with pytest.raises(ValidationError):
        ResearchPlan(
            objective="too many steps",
            required_facets=["one"],
            steps=[_step(), _step(), _step(), _step()],
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {
                "tool": "search_document",
                "purpose_code": "facet_search",
            },
            "query",
        ),
        (
            {
                "tool": "search_artifacts",
                "query": "evidence",
                "purpose_code": "artifact_lookup",
                "evidence_requirement": "text",
            },
            "structured evidence",
        ),
        (
            {
                "tool": "expand_evidence",
                "purpose_code": "context_expansion",
            },
            "evidence_ids",
        ),
        (
            {
                "tool": "inspect_document_structure",
                "query": "model supplied scope",
                "purpose_code": "structure_lookup",
            },
            "does not accept a query",
        ),
        (
            {
                "tool": "search_document",
                "query": "question",
                "purpose_code": "artifact_lookup",
            },
            "requires purpose_code=facet_search",
        ),
    ],
)
def test_tool_contract_rejects_incomplete_or_overpowered_steps(
    values: dict,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ResearchToolStep(**values)


def test_application_owned_tool_context_requires_fixed_scope() -> None:
    context = EvidenceToolContext(
        document_id="doc-1",
        index_name="agentic-rag-hybrid-v3",
    )

    assert context.document_id == "doc-1"

    with pytest.raises(ValueError, match="document_id"):
        EvidenceToolContext(document_id=" ", index_name="v3")

