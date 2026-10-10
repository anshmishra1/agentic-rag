import json
from pathlib import Path

from agentic_rag.evaluation.answer_reliability import (
    evaluate_acceptance,
    render_acceptance_summary,
    validate_acceptance_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "eval_set" / "answer_reliability_v2_acceptance.json"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _passing_result(case: dict) -> dict:
    query = " ".join(group[0] for group in case.get("required_query_term_groups", []))
    return {
        "case_id": case["case_id"],
        "answer_status": case["allowed_statuses"][0],
        "information_need_source": case["expected_information_need_source"],
        "response_format": case["expected_response_format"],
        "retrieval_query": query,
        "citations_valid": True,
        "retrieval_count": 1,
        "rewrite_count": 0,
        "correction_count": 0,
        "answer": "Supported answer.",
    }


def test_reviewed_manifest_freezes_the_observed_eleven_case_baseline() -> None:
    manifest = _manifest()

    assert validate_acceptance_manifest(manifest) == []
    assert len(manifest["cases"]) == 11
    assert manifest["baseline"]["corrected_answered_cases"] == 5
    assert manifest["baseline"]["answered_cases"] == 8
    assert manifest["baseline"]["diagram_case_latency_seconds"] == 10.54


def test_acceptance_report_passes_complete_sanitized_results() -> None:
    manifest = _manifest()
    results = [_passing_result(case) for case in manifest["cases"]]

    report = evaluate_acceptance(manifest, results)
    summary = render_acceptance_summary(report)

    assert report["total_cases"] == 11
    assert report["passed_cases"] == 11
    assert report["failed_cases"] == 0
    assert "| flash-document-diagram | PASS |" in summary
    assert "Supported answer" not in summary


def test_acceptance_report_exposes_wrong_query_reuse_and_unsupported_claim() -> None:
    manifest = _manifest()
    case = next(
        item for item in manifest["cases"] if item["case_id"] == "flash-document-diagram"
    )
    result = _passing_result(case)
    result.update(
        answer_status="unsupported",
        information_need_source="previous_information_need",
        retrieval_query="Explain the previous architectural difference",
        retrieval_count=3,
        rewrite_count=2,
        answer="No other figures are present.",
    )

    report = evaluate_acceptance(
        {**manifest, "cases": [case]},
        [result],
    )
    failures = report["cases"][0]["failures"]

    assert report["failed_cases"] == 1
    assert any("unexpected answer_status" in failure for failure in failures)
    assert any("information_need_source" in failure for failure in failures)
    assert any("retrieval_query" in failure for failure in failures)
    assert any("retrieval_count=3" in failure for failure in failures)
    assert any("rewrite_count=2" in failure for failure in failures)
    assert any("forbidden answer claim" in failure for failure in failures)
