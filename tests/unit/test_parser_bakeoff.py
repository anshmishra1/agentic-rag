import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from agentic_rag.evaluation.parser_bakeoff import (
    ParserCase,
    audit_parser_corpus,
    compare_repeatability,
    evaluate_parser_case,
    load_parser_cases,
)
from agentic_rag.ingestion.contracts import (
    Artifact,
    ArtifactKind,
    Derivation,
    GroundingBasis,
    RepresentationKind,
    SearchRecord,
    SourceLocator,
)
from agentic_rag.ingestion.parsers.base import ParseRun


def _artifact(*, kind: ArtifactKind, text: str, page: int = 3) -> Artifact:
    return Artifact.from_text(
        document_id="doc-123",
        kind=kind,
        text=text,
        derivation=Derivation(producer="fake-parser", revision="1"),
        locator=SourceLocator(page_start=page, page_end=page, page_label=str(page)),
    )


def _run(*artifacts: Artifact) -> ParseRun:
    records = tuple(
        SearchRecord.from_artifact(
            artifact,
            representation_kind=RepresentationKind.NATIVE_TEXT,
            search_text=artifact.native_text or "",
            grounding_basis=GroundingBasis.DIRECT_TEXT,
            retrieval_schema_id="multimodal_v3:1234567890abcdef",
        )
        for artifact in artifacts
    )
    return ParseRun(
        parser_name="fake-parser",
        parser_revision="1",
        document_id="doc-123",
        retrieval_schema_id="multimodal_v3:1234567890abcdef",
        selected_pages=(3,),
        artifacts=tuple(artifacts),
        records=records,
        elapsed_seconds=0.5,
        peak_memory_mib=32.0,
    )


def test_tracked_parser_manifest_is_valid_and_covers_each_artifact_class() -> None:
    cases = load_parser_cases("eval_set/v3_parser_bakeoff_cases.json")
    kinds = {kind for case in cases for kind in case.minimum_artifacts}

    assert len(cases) == 10
    assert {case.source_group for case in cases} == {"research_articles", "repository"}
    assert {
        ArtifactKind.TEXT_BLOCK,
        ArtifactKind.FIGURE,
        ArtifactKind.TABLE,
        ArtifactKind.FORMULA,
        ArtifactKind.PAGE_IMAGE,
    }.issubset(kinds)


def _write_manifest(tmp_path: Path, cases: list[dict]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"schema_version": 1, "cases": cases}),
        encoding="utf-8",
    )
    return path


def _manifest_case(*, case_id: str, source_file: str = "safe.pdf") -> dict:
    return {
        "case_id": case_id,
        "source_group": "repository",
        "source_file": source_file,
        "sha256": "a" * 64,
        "pages": [1],
        "purpose": "test",
        "expectations": {"minimum_artifacts": {"text_block": 1}},
    }


def test_manifest_rejects_path_escape(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [_manifest_case(case_id="escape", source_file="../secret.pdf")],
    )

    with pytest.raises(ValueError, match="within its configured source root"):
        load_parser_cases(path)


def test_manifest_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    manifest = {
        **_manifest_case(case_id="duplicate"),
        "sha256": "b" * 64,
    }
    path = _write_manifest(
        tmp_path,
        [_manifest_case(case_id="duplicate"), manifest],
    )

    with pytest.raises(ValueError, match="duplicate parser case_id"):
        load_parser_cases(path)


def test_corpus_audit_checks_each_unique_file_once(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"fixed test payload")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    case = ParserCase(
        case_id="case-one",
        source_group="research",
        source_file="paper.pdf",
        sha256=digest,
        pages=(1,),
        purpose="test",
        minimum_artifacts={ArtifactKind.TEXT_BLOCK: 1},
        required_text_terms=(),
    )

    result = audit_parser_corpus(
        [case, replace(case, case_id="case-two")],
        {"research": tmp_path},
    )

    assert result["valid"] is True
    assert result["case_count"] == 2
    assert result["unique_file_count"] == 1


def test_case_evaluation_measures_artifacts_text_and_page_coverage() -> None:
    text = _artifact(
        kind=ArtifactKind.TEXT_BLOCK,
        text="The Transformer encoder and decoder use attention.",
    )
    figure = _artifact(kind=ArtifactKind.FIGURE, text="Figure 1 architecture")
    case = ParserCase(
        case_id="architecture",
        source_group="research",
        source_file="paper.pdf",
        sha256="a" * 64,
        pages=(3,),
        purpose="architecture figure",
        minimum_artifacts={ArtifactKind.TEXT_BLOCK: 1, ArtifactKind.FIGURE: 1},
        required_text_terms=("encoder", "missing term"),
    )

    result = evaluate_parser_case(case, _run(text, figure))

    assert result["artifact_checks"]["figure"]["met"] is True
    assert result["required_text_checks"] == {"encoder": True, "missing term": False}
    assert result["page_coverage"] == {"3": True}
    assert result["expectations_met"] == 4
    assert result["expectation_count"] == 5


def test_repeatability_compares_stable_ids_without_passage_text() -> None:
    artifact = _artifact(kind=ArtifactKind.TEXT_BLOCK, text="Stable passage")
    first = _run(artifact)
    stable = _run(artifact)
    changed = _run(_artifact(kind=ArtifactKind.TEXT_BLOCK, text="Changed passage"))

    assert compare_repeatability(first, stable) == {
        "artifact_ids_stable": True,
        "record_ids_stable": True,
        "artifact_only_in_first": [],
        "artifact_only_in_second": [],
        "record_only_in_first": [],
        "record_only_in_second": [],
    }
    comparison = compare_repeatability(first, changed)
    assert comparison["artifact_ids_stable"] is False
    assert comparison["record_ids_stable"] is False


def test_parse_run_rejects_broken_artifact_lineage() -> None:
    artifact = _artifact(kind=ArtifactKind.TEXT_BLOCK, text="Passage")
    other = _artifact(kind=ArtifactKind.TEXT_BLOCK, text="Other")
    record = SearchRecord.from_artifact(
        other,
        representation_kind=RepresentationKind.NATIVE_TEXT,
        search_text="Other",
        grounding_basis=GroundingBasis.DIRECT_TEXT,
        retrieval_schema_id="multimodal_v3:1234567890abcdef",
    )

    with pytest.raises(ValueError, match="unavailable artifact"):
        ParseRun(
            parser_name="fake-parser",
            parser_revision="1",
            document_id="doc-123",
            retrieval_schema_id="multimodal_v3:1234567890abcdef",
            selected_pages=(3,),
            artifacts=(artifact,),
            records=(record,),
            elapsed_seconds=0,
            peak_memory_mib=0,
        )


def test_parse_run_rejects_artifact_spanning_unselected_page() -> None:
    artifact = Artifact.from_text(
        document_id="doc-123",
        kind=ArtifactKind.TEXT_BLOCK,
        text="Cross-page passage",
        derivation=Derivation(producer="fake-parser", revision="1"),
        locator=SourceLocator(page_start=3, page_end=4),
    )

    with pytest.raises(ValueError, match="outside the requested page selection"):
        ParseRun(
            parser_name="fake-parser",
            parser_revision="1",
            document_id="doc-123",
            retrieval_schema_id="multimodal_v3:1234567890abcdef",
            selected_pages=(3,),
            artifacts=(artifact,),
            records=(),
            elapsed_seconds=0,
            peak_memory_mib=0,
        )
