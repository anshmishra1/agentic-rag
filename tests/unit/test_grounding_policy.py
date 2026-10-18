import json
from types import SimpleNamespace

import pytest

from agentic_rag.policies.grounding import (
    build_grounding_verifier_prompt,
    citation_issue_result,
    grounding_result,
    extract_verification_units,
    parse_grounding_response,
    verifier_response_diagnostics,
)


def test_verifier_prompt_keeps_parent_and_child_claim_ownership_separate() -> None:
    answer = (
        "- Each encoder layer contains two sublayers [S5]:\n"
        "  1. Multi-head self-attention [S1].\n"
        "  2. A position-wise feed-forward network [S3]."
    )

    prompt = build_grounding_verifier_prompt(
        question="Explain the encoder.",
        context="[S1] attention\n[S3] feed-forward\n[S5] two sublayers",
        answer=answer,
    )

    assert "parent statement and every child item are separate claim units" in prompt
    assert "Do not merge parent and child text into one claim" in prompt
    assert "Multiple labels on one unit may jointly support it" in prompt
    assert "Do not use an uncited source or general knowledge" in prompt
    assert "[C1] - Each encoder layer contains two sublayers [S5]:" in prompt
    assert "[C2] 1. Multi-head self-attention [S1]." in prompt
    assert "Do not judge whether the answer obeys a requested format" in prompt


def test_verification_units_remove_rendering_syntax_but_keep_semantics() -> None:
    answer = (
        "```text\n"
        "+----------------------+\n"
        "| Load HBM tiles [S1] |\n"
        "          |\n"
        "          v\n"
        "| Compute attention [S2] |\n"
        "```\n\n"
        "| Stage | Purpose |\n"
        "| --- | --- |\n"
        "| SRAM | On-chip work [S3] |\n"
        "\\[\n"
        "O(N^2) [S4]\n"
        "\\]"
    )

    assert extract_verification_units(answer) == [
        "| Load HBM tiles [S1] |",
        "| Compute attention [S2] |",
        "| Stage | Purpose |",
        "| SRAM | On-chip work [S3] |",
        "O(N^2) [S4]",
    ]


@pytest.mark.parametrize(
    ("verdict", "expected_claims"),
    [
        ("grounded", []),
        ("insufficient_evidence", []),
        ("unsupported", ["claim one"]),
    ],
)
def test_parse_supported_verdicts(
    verdict: str,
    expected_claims: list[str],
) -> None:
    raw = json.dumps(
        {
            "verdict": verdict,
            "unsupported_claims": ["claim one"],
        }
    )

    parsed_verdict, claims, parsed = parse_grounding_response(raw)

    assert parsed_verdict == verdict
    assert claims == expected_claims
    assert parsed is True


def test_parse_markdown_fenced_json() -> None:
    raw = """```json
{"verdict": "grounded", "unsupported_claims": []}
```"""

    assert parse_grounding_response(raw) == ("grounded", [], True)


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "",
        '{"verdict": "unknown"}',
        '{"verdict":"unsupported","unsupported_claims":"not-an-array"}',
        "[]",
        "{}",
    ],
)
def test_invalid_response_fails_closed(raw: str) -> None:
    assert parse_grounding_response(raw) == ("unsupported", [], False)


def test_grounded_result_is_verified() -> None:
    result = grounding_result(
        '{"verdict":"grounded","unsupported_claims":[]}',
        correction_attempted=False,
    )

    assert result["hallucination_grade"] == "grounded"
    assert result["answer_status"] == "answered"
    assert result["verification_exhausted"] is False


def test_insufficient_evidence_is_not_treated_as_generation_error() -> None:
    result = grounding_result(
        '{"verdict":"insufficient_evidence","unsupported_claims":[]}',
        correction_attempted=False,
    )

    assert result["grounding_diagnosis"] == "insufficient_evidence"
    assert result["answer_status"] == "insufficient_evidence"
    assert result["verification_exhausted"] is False


def test_unsupported_claims_are_preserved_for_correction() -> None:
    result = grounding_result(
        '{"verdict":"unsupported","unsupported_claims":["invented fact"]}',
        correction_attempted=False,
    )

    assert result["grounding_unsupported_claims"] == ["invented fact"]
    assert result["answer_status"] == "unsupported"
    assert result["verification_exhausted"] is False


def test_malformed_verdict_fails_closed_and_exhausts_after_correction() -> None:
    result = grounding_result(
        "not json",
        correction_attempted=True,
    )

    assert result["grounding_diagnosis"] == "unsupported"
    assert result["grounding_parse_success"] is False
    assert result["answer_status"] == "verification_uncertain"
    assert result["verification_exhausted"] is True


def test_citation_failure_routes_to_correction_then_exhausts() -> None:
    issues = ["List item lacks a source citation: unsupported example"]

    first = citation_issue_result(issues, correction_attempted=False)
    second = citation_issue_result(issues, correction_attempted=True)

    assert first["grounding_diagnosis"] == "unsupported"
    assert first["grounding_unsupported_claims"] == issues
    assert first["verification_exhausted"] is False
    assert second["verification_exhausted"] is True


def test_empty_verifier_diagnostics_exclude_response_text() -> None:
    response = SimpleNamespace(
        content="sensitive answer text",
        additional_kwargs={"reasoning_content": "sensitive reasoning text"},
        response_metadata={
            "finish_reason": "length",
            "token_usage": {
                "completion_tokens": 512,
                "completion_tokens_details": {"reasoning_tokens": 510},
            },
        },
    )

    diagnostics = verifier_response_diagnostics(response)

    assert diagnostics == {
        "finish_reason": "length",
        "completion_tokens": 512,
        "reasoning_tokens": 510,
    }
    assert "sensitive" not in str(diagnostics)
