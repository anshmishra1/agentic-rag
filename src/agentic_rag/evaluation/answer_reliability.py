"""Pure analysis for the v2 answer-reliability acceptance contract."""

from __future__ import annotations

from collections.abc import Sequence


def validate_acceptance_manifest(manifest: dict) -> list[str]:
    """Return human-readable contract errors without contacting runtime services."""

    errors: list[str] = []
    cases = manifest.get("cases")
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(cases, list) or not cases:
        return [*errors, "cases must be a non-empty list"]

    seen: set[str] = set()
    required = {
        "case_id",
        "question",
        "source_mode",
        "expected_information_need_source",
        "expected_response_format",
        "allowed_statuses",
        "citations_required_when_answered",
        "max_retrievals",
        "max_rewrites",
        "max_corrections",
    }
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"cases[{index}] must be an object")
            continue
        missing = sorted(required - case.keys())
        if missing:
            errors.append(f"cases[{index}] missing: {', '.join(missing)}")
        case_id = str(case.get("case_id", "")).strip()
        if not case_id:
            errors.append(f"cases[{index}] has an empty case_id")
        elif case_id in seen:
            errors.append(f"duplicate case_id: {case_id}")
        seen.add(case_id)
    return errors


def _contains_required_query_terms(query: str, groups: Sequence[Sequence[str]]) -> bool:
    normalized = query.casefold()
    return all(any(term.casefold() in normalized for term in group) for group in groups)


def evaluate_acceptance(manifest: dict, results: Sequence[dict]) -> dict:
    """Compare sanitized runtime results with the reviewed acceptance contract."""

    manifest_errors = validate_acceptance_manifest(manifest)
    if manifest_errors:
        raise ValueError("; ".join(manifest_errors))

    by_id = {str(row.get("case_id")): row for row in results}
    evaluated: list[dict] = []

    for case in manifest["cases"]:
        case_id = case["case_id"]
        actual = by_id.get(case_id)
        failures: list[str] = []
        if actual is None:
            evaluated.append(
                {"case_id": case_id, "passed": False, "failures": ["missing result"]}
            )
            continue

        status = str(actual.get("answer_status", ""))
        if status not in case["allowed_statuses"]:
            failures.append(f"unexpected answer_status={status or '<missing>'}")

        information_need_source = str(actual.get("information_need_source", ""))
        if information_need_source != case["expected_information_need_source"]:
            failures.append(
                "information_need_source="
                f"{information_need_source or '<missing>'}"
            )

        response_format = str(actual.get("response_format", ""))
        if response_format != case["expected_response_format"]:
            failures.append(f"response_format={response_format or '<missing>'}")

        query_groups = case.get("required_query_term_groups", [])
        retrieval_query = str(actual.get("retrieval_query", ""))
        if query_groups and not _contains_required_query_terms(
            retrieval_query, query_groups
        ):
            failures.append("retrieval_query lacks required subject terms")

        if (
            status == "answered"
            and case["citations_required_when_answered"]
            and not actual.get("citations_valid", False)
        ):
            failures.append("answered result has invalid citations")

        count_fields = (
            ("retrieval_count", "max_retrievals"),
            ("rewrite_count", "max_rewrites"),
            ("correction_count", "max_corrections"),
        )
        for actual_field, limit_field in count_fields:
            count = int(actual.get(actual_field, 0))
            if count > int(case[limit_field]):
                failures.append(f"{actual_field}={count} exceeds {case[limit_field]}")

        answer = str(actual.get("answer", "")).casefold()
        for phrase in case.get("forbidden_answer_phrases", []):
            if phrase.casefold() in answer:
                failures.append(f"forbidden answer claim: {phrase}")

        evaluated.append(
            {
                "case_id": case_id,
                "passed": not failures,
                "answer_status": status,
                "correction_count": int(actual.get("correction_count", 0)),
                "failures": failures,
            }
        )

    answered = [row for row in evaluated if row.get("answer_status") == "answered"]
    corrected = [row for row in answered if row.get("correction_count", 0) > 0]
    passed = sum(bool(row["passed"]) for row in evaluated)
    return {
        "schema_version": 1,
        "total_cases": len(evaluated),
        "passed_cases": passed,
        "failed_cases": len(evaluated) - passed,
        "answered_cases": len(answered),
        "corrected_answered_cases": len(corrected),
        "correction_rate": (
            round(len(corrected) / len(answered), 4) if answered else None
        ),
        "cases": evaluated,
    }


def render_acceptance_summary(report: dict) -> str:
    """Render a compact review artifact without answer or passage text."""

    correction_rate = report.get("correction_rate")
    correction_display = (
        f"{correction_rate:.1%}" if isinstance(correction_rate, float) else "n/a"
    )
    lines = [
        "# V2 answer-reliability acceptance report",
        "",
        f"- Cases: {report['total_cases']}",
        f"- Passed: {report['passed_cases']}",
        f"- Failed: {report['failed_cases']}",
        f"- Corrected answered cases: {report['corrected_answered_cases']} / "
        f"{report['answered_cases']} ({correction_display})",
        "",
        "| Case | Result | Status | Failures |",
        "|---|---|---|---|",
    ]
    for row in report["cases"]:
        failures = "; ".join(row["failures"]) or "-"
        lines.append(
            f"| {row['case_id']} | {'PASS' if row['passed'] else 'FAIL'} | "
            f"{row.get('answer_status', '-')} | {failures} |"
        )
    return "\n".join(lines) + "\n"
