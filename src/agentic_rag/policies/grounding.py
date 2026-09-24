"""Grounding-verdict parsing and classification policy.

Kept separate from graph/nodes.py so the decision rule (what counts as a
valid verdict, how to fail safely on a malformed response) can be tested
and changed independently of the node's I/O/tracing responsibilities.
"""
from __future__ import annotations

import json

GROUNDING_VERDICTS = {"grounded", "insufficient_evidence", "unsupported"}

ABSTENTION_RESPONSE = (
    "I don't have enough evidence in the selected document to answer that "
    "reliably. Try rephrasing the question or selecting a document that "
    "covers the requested information."
)


def parse_grounding_response(raw: str) -> tuple[str, list[str], bool]:
    """Parse the grounding verifier's JSON response defensively.

    Returns (verdict, unsupported_claims, parsed_successfully).

    Any parse failure or unrecognized verdict defaults to verdict='unsupported'
    with parsed_successfully=False - an unparseable response is not evidence
    the answer is fine, so we fail closed rather than fail open."""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            lines = lines[1:] if lines else lines
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("grounding response must be a JSON object")

        verdict = str(payload.get("verdict", "")).strip().lower()
        if verdict not in GROUNDING_VERDICTS:
            raise ValueError(f"unrecognized verdict: {verdict!r}")

        claims = payload.get("unsupported_claims") or []
        if not isinstance(claims, list):
            raise ValueError("unsupported_claims must be a JSON array")

        normalized_claims = [
            str(claim).strip()
            for claim in claims
            if str(claim).strip()
        ]
        if verdict != "unsupported":
            normalized_claims = []

        return verdict, normalized_claims, True
    except Exception:
        return "unsupported", [], False


def grounding_result(raw: str, *, correction_attempted: bool) -> dict:
    """Convert a verifier response into graph routing and answer fields.

    ``grounding_diagnosis`` is authoritative. ``hallucination_grade`` remains
    only as a compatibility field until the broader state migration is done.
    """
    verdict, unsupported_claims, parsed_successfully = (
        parse_grounding_response(raw)
    )

    if verdict == "grounded":
        hallucination_grade = "grounded"
        answer_status = "answered"
        verification_exhausted = False
    elif verdict == "insufficient_evidence":
        hallucination_grade = "hallucinated"
        answer_status = "insufficient_evidence"
        verification_exhausted = False
    else:
        hallucination_grade = "hallucinated"
        answer_status = (
            "unsupported"
            if parsed_successfully
            else "verification_uncertain"
        )
        verification_exhausted = correction_attempted

    return {
        "hallucination_grade": hallucination_grade,
        "grounding_diagnosis": verdict,
        "grounding_unsupported_claims": unsupported_claims,
        "grounding_parse_success": parsed_successfully,
        "answer_status": answer_status,
        "verification_exhausted": verification_exhausted,
    }
