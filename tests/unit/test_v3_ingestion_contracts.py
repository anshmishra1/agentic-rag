from dataclasses import replace

import pytest
from langchain_core.documents import Document

from agentic_rag.ingestion.contracts import (
    Artifact,
    ArtifactKind,
    Derivation,
    EvidenceRef,
    GroundingBasis,
    RepresentationKind,
    RetrievalSchema,
    SearchRecord,
    SourceLocator,
)
from agentic_rag.ingestion.legacy_adapter import legacy_document_to_v3


def _schema(**overrides) -> RetrievalSchema:
    values = {
        "parser_name": "docling",
        "parser_revision": "2.58.0",
        "parser_options": {"ocr": "conditional", "table_mode": "accurate"},
        "chunking_strategy": "artifact_structure_v3",
        "text_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "text_embedding_revision": "revision-1",
        "text_embedding_dimension": 384,
    }
    values.update(overrides)
    return RetrievalSchema(**values)


def test_retrieval_schema_fingerprint_is_order_independent_and_complete() -> None:
    first = _schema(parser_options={"ocr": "conditional", "table_mode": "accurate"})
    reordered = _schema(parser_options={"table_mode": "accurate", "ocr": "conditional"})

    assert first.fingerprint == reordered.fingerprint
    assert first.schema_id == reordered.schema_id
    assert first.schema_id.startswith("multimodal_v3:")



@pytest.mark.parametrize(
    "change",
    [
        {"schema_family": "multimodal_v3_1"},
        {"parser_name": "pymupdf"},
        {"parser_revision": "2.59.0"},
        {"parser_options": {"ocr": "always", "table_mode": "accurate"}},
        {"chunking_strategy": "artifact_structure_v3_1"},
        {"representation_contract": "artifact_search_record_v2"},
        {"text_embedding_model": "BAAI/bge-m3"},
        {"text_embedding_revision": "revision-2"},
        {"text_embedding_dimension": 1024},
        {"sparse_encoder": "splade_v1"},
        {"vector_id_contract": "artifact_representation_hash_v2"},
        {
            "visual_embedding_model": "google/siglip-base-patch16-224",
            "visual_embedding_revision": "revision-visual",
            "visual_embedding_dimension": 768,
        },
    ],
)
def test_every_index_defining_change_gets_a_new_schema_fingerprint(change) -> None:
    baseline = _schema()

    assert replace(baseline, **change).fingerprint != baseline.fingerprint


def test_visual_embedding_identity_is_all_or_nothing() -> None:
    with pytest.raises(ValueError, match="complete set"):
        _schema(visual_embedding_model="google/siglip-base-patch16-224")


def test_artifact_and_search_record_ids_are_stable_and_lineage_preserving() -> None:
    locator = SourceLocator(page_start=4, page_end=4, page_label="4", bbox=(0.1, 0.2, 0.9, 0.8))
    derivation = Derivation(
        producer="docling",
        revision="2.58.0",
        parameters={"table_mode": "accurate"},
    )
    first = Artifact.from_text(
        document_id="doc-123",
        kind=ArtifactKind.TABLE,
        text="| Model | Score |\n|---|---|\n| Ettin | 0.91 |",
        derivation=derivation,
        locator=locator,
    )
    second = Artifact.from_text(
        document_id="doc-123",
        kind=ArtifactKind.TABLE,
        text="| Model | Score |\n|---|---|\n| Ettin | 0.91 |",
        derivation=derivation,
        locator=locator,
    )

    assert first.artifact_id == second.artifact_id

    record = SearchRecord.from_artifact(
        first,
        representation_kind=RepresentationKind.TABLE_MARKDOWN,
        search_text=first.native_text or "",
        grounding_basis=GroundingBasis.TABLE_EXTRACTION,
        retrieval_schema_id=_schema().schema_id,
    )
    changed_text = SearchRecord.from_artifact(
        first,
        representation_kind=RepresentationKind.TABLE_MARKDOWN,
        search_text="Ettin achieved 0.91.",
        grounding_basis=GroundingBasis.TABLE_EXTRACTION,
        retrieval_schema_id=_schema().schema_id,
    )
    changed_schema = SearchRecord.from_artifact(
        first,
        representation_kind=RepresentationKind.TABLE_MARKDOWN,
        search_text=first.native_text or "",
        grounding_basis=GroundingBasis.TABLE_EXTRACTION,
        retrieval_schema_id=_schema(parser_revision="2.59.0").schema_id,
    )

    assert record.artifact_id == first.artifact_id
    assert record.record_id != changed_text.record_id
    assert record.record_id != changed_schema.record_id


def test_locators_reject_ambiguous_or_invalid_coordinates() -> None:
    with pytest.raises(ValueError, match="requires a source page"):
        SourceLocator(bbox=(0.1, 0.1, 0.9, 0.9))
    with pytest.raises(ValueError, match="normalized and ordered"):
        SourceLocator(page_start=1, bbox=(0.9, 0.1, 0.2, 0.9))
    with pytest.raises(ValueError, match="both start_ms and end_ms"):
        SourceLocator(start_ms=100)
    with pytest.raises(ValueError, match="cannot mix"):
        SourceLocator(page_start=1, start_ms=100, end_ms=200)


def test_evidence_reference_is_serializable_without_content_payloads() -> None:
    evidence = EvidenceRef(
        label="S2",
        document_id="doc-123",
        artifact_id="artifact-123",
        record_id="record-123",
        artifact_kind=ArtifactKind.FIGURE,
        grounding_basis=GroundingBasis.VISION_DESCRIPTION,
        source_name="paper.pdf",
        description="Figure 2, memory hierarchy",
        locator=SourceLocator(page_start=5, page_end=5, page_label="5"),
        asset_url="/documents/doc-123/artifacts/artifact-123",
    )

    payload = evidence.to_dict()

    assert payload["kind"] == "figure"
    assert payload["grounding_basis"] == "vision_description"
    assert payload["locator"]["page_start"] == 5
    assert "text" not in payload


def test_legacy_adapter_preserves_current_chunk_text_and_page_provenance() -> None:
    document = Document(
        page_content="Attention uses query, key, and value projections.",
        metadata={
            "type": "content",
            "filename": "attention.pdf",
            "page": 2,
            "page_label": "3",
            "chunking_strategy": "structure_aware_v2",
            "extraction_mode": "layout",
        },
    )

    artifact, record = legacy_document_to_v3(
        document,
        document_id="doc-attention",
        retrieval_schema_id=_schema().schema_id,
    )

    assert artifact.native_text == document.page_content
    assert artifact.locator.page_start == 3
    assert artifact.locator.page_label == "3"
    assert record.search_text == document.page_content
    assert record.representation_kind == RepresentationKind.NATIVE_TEXT
    assert record.grounding_basis == GroundingBasis.DIRECT_TEXT
    assert record.artifact_id == artifact.artifact_id
