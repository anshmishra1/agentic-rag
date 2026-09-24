import json

import pytest

from agentic_rag.policies.grounding import parse_grounding_response


@pytest.mark.parametrize(
    "verdict",
    ["grounded", "insufficient_evidence", "unsupported"],
)
def test_parse_supported_verdicts(verdict: str) -> None:
    raw = json.dumps(
        {
            "verdict": verdict,
            "unsupported_claims": ["claim one"],
        }
    )

    parsed_verdict, claims, parsed = parse_grounding_response(raw)

    assert parsed_verdict == verdict
    assert claims == ["claim one"]
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
        "{}",
    ],
)
def test_invalid_response_fails_closed(raw: str) -> None:
    assert parse_grounding_response(raw) == ("unsupported", [], False)
