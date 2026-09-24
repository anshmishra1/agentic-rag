import json

import pytest

from agentic_rag.policies.grounding import grounding_result, parse_grounding_response


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
