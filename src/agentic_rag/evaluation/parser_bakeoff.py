"""Pure validation and scoring for the V3 PDF parser bakeoff."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from agentic_rag.ingestion.contracts import ArtifactKind
from agentic_rag.ingestion.parsers.base import ParseRun


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ParserCase:
    case_id: str
    source_group: str
    source_file: str
    sha256: str
    pages: tuple[int, ...]
    purpose: str
    minimum_artifacts: dict[ArtifactKind, int]
    required_text_terms: tuple[str, ...]


def _relative_source_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source_file must be a nonempty relative path")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("source_file must stay within its configured source root")
    return normalized


def load_parser_cases(path: str | Path) -> list[ParserCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("parser corpus schema_version must be 1")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("parser corpus needs at least one case")

    cases: list[ParserCase] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"case {index} must be an object")
        case_id = raw.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"case {index} needs a nonempty case_id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate parser case_id: {case_id}")
        seen_ids.add(case_id)

        source_group = raw.get("source_group")
        if not isinstance(source_group, str) or not source_group.strip():
            raise ValueError(f"case {case_id} needs a source_group")
        digest = raw.get("sha256")
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"case {case_id} needs a lowercase SHA-256 digest")
        pages = raw.get("pages")
        if (
            not isinstance(pages, list)
            or not pages
            or any(not isinstance(page, int) or isinstance(page, bool) or page < 1 for page in pages)
            or pages != sorted(set(pages))
        ):
            raise ValueError(f"case {case_id} pages must be sorted unique positive integers")
        purpose = raw.get("purpose")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError(f"case {case_id} needs a purpose")

        expectations = raw.get("expectations")
        if not isinstance(expectations, dict):
            raise ValueError(f"case {case_id} needs expectations")
        raw_minimums = expectations.get("minimum_artifacts", {})
        if not isinstance(raw_minimums, dict) or not raw_minimums:
            raise ValueError(f"case {case_id} needs minimum artifact expectations")
        minimums: dict[ArtifactKind, int] = {}
        for kind_value, minimum in raw_minimums.items():
            try:
                kind = ArtifactKind(kind_value)
            except ValueError as exc:
                raise ValueError(
                    f"case {case_id} has unsupported artifact kind: {kind_value}"
                ) from exc
            if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
                raise ValueError(f"case {case_id} artifact minimums must be positive integers")
            minimums[kind] = minimum

        terms = expectations.get("required_text_terms", [])
        if (
            not isinstance(terms, list)
            or any(not isinstance(term, str) or not term.strip() for term in terms)
        ):
            raise ValueError(f"case {case_id} required_text_terms must be nonempty strings")
        cases.append(
            ParserCase(
                case_id=case_id,
                source_group=source_group,
                source_file=_relative_source_path(raw.get("source_file")),
                sha256=digest,
                pages=tuple(pages),
                purpose=purpose,
                minimum_artifacts=minimums,
                required_text_terms=tuple(terms),
            )
        )
    return cases


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_parser_corpus(
    cases: list[ParserCase],
    source_roots: dict[str, Path],
) -> dict[str, Any]:
    """Verify exact files without parsing content or loading application settings."""
    checked_files: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[str] = []

    for case in cases:
        root = source_roots.get(case.source_group)
        if root is None:
            errors.append(f"{case.case_id}: missing source root '{case.source_group}'")
            continue
        root = root.resolve()
        source = (root / Path(case.source_file)).resolve()
        if root != source and root not in source.parents:
            errors.append(f"{case.case_id}: source resolves outside configured root")
            continue
        key = (case.source_group, case.source_file)
        if key not in checked_files:
            if not source.is_file():
                checked_files[key] = {"exists": False, "sha256": None}
            else:
                checked_files[key] = {"exists": True, "sha256": _file_sha256(source)}
        result = checked_files[key]
        if not result["exists"]:
            errors.append(f"{case.case_id}: source file is missing")
        elif result["sha256"] != case.sha256:
            errors.append(f"{case.case_id}: source SHA-256 does not match")

    return {
        "case_count": len(cases),
        "unique_file_count": len(checked_files),
        "files": [
            {
                "source_group": group,
                "source_file": source_file,
                **result,
            }
            for (group, source_file), result in sorted(checked_files.items())
        ],
        "errors": errors,
        "valid": not errors,
    }


def evaluate_parser_case(case: ParserCase, run: ParseRun) -> dict[str, Any]:
    if tuple(run.selected_pages) != case.pages:
        raise ValueError("parse run pages do not match the evaluation case")

    counts = Counter(artifact.kind for artifact in run.artifacts)
    artifact_checks = {
        kind.value: {
            "expected_minimum": minimum,
            "observed": counts[kind],
            "met": counts[kind] >= minimum,
        }
        for kind, minimum in sorted(
            case.minimum_artifacts.items(), key=lambda item: item[0].value
        )
    }
    searchable_text = "\n".join(record.search_text for record in run.records).casefold()
    term_checks = {
        term: term.casefold() in searchable_text
        for term in case.required_text_terms
    }
    pages_with_artifacts = {
        artifact.locator.page_start
        for artifact in run.artifacts
        if artifact.locator.page_start is not None
    }
    page_coverage = {
        str(page): page in pages_with_artifacts
        for page in case.pages
    }
    checks = [entry["met"] for entry in artifact_checks.values()]
    checks.extend(term_checks.values())
    checks.extend(page_coverage.values())

    return {
        "case_id": case.case_id,
        "parser": {"name": run.parser_name, "revision": run.parser_revision},
        "purpose": case.purpose,
        "artifact_checks": artifact_checks,
        "required_text_checks": term_checks,
        "page_coverage": page_coverage,
        "expectations_met": sum(bool(value) for value in checks),
        "expectation_count": len(checks),
        "expectation_rate": (
            sum(bool(value) for value in checks) / len(checks)
            if checks
            else 1.0
        ),
        "elapsed_seconds": run.elapsed_seconds,
        "peak_memory_mib": run.peak_memory_mib,
        "warning_count": len(run.warnings),
    }


def compare_repeatability(first: ParseRun, second: ParseRun) -> dict[str, Any]:
    """Compare two runs without comparing or persisting extracted text."""
    identity = (
        first.parser_name,
        first.parser_revision,
        first.document_id,
        first.retrieval_schema_id,
        first.selected_pages,
    )
    other_identity = (
        second.parser_name,
        second.parser_revision,
        second.document_id,
        second.retrieval_schema_id,
        second.selected_pages,
    )
    if identity != other_identity:
        raise ValueError("repeatability comparison requires identical run configuration")

    first_artifacts = {artifact.artifact_id for artifact in first.artifacts}
    second_artifacts = {artifact.artifact_id for artifact in second.artifacts}
    first_records = {record.record_id for record in first.records}
    second_records = {record.record_id for record in second.records}
    return {
        "artifact_ids_stable": first_artifacts == second_artifacts,
        "record_ids_stable": first_records == second_records,
        "artifact_only_in_first": sorted(first_artifacts - second_artifacts),
        "artifact_only_in_second": sorted(second_artifacts - first_artifacts),
        "record_only_in_first": sorted(first_records - second_records),
        "record_only_in_second": sorted(second_records - first_records),
    }
