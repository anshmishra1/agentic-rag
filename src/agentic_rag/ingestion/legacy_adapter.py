"""Compatibility adapter from the current V2 document model to V3 contracts."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from agentic_rag.ingestion.contracts import (
    Artifact,
    ArtifactKind,
    Derivation,
    GroundingBasis,
    RepresentationKind,
    SearchRecord,
    SourceLocator,
)


def _page_number(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if value > 0 else None
    if isinstance(value, str):
        match = re.match(r"\s*(\d+)", value)
        if match:
            parsed = int(match.group(1))
            return parsed if parsed > 0 else None
    return None


def _source_locator(metadata: dict[str, Any]) -> SourceLocator:
    page_start = _page_number(metadata.get("page_start"))
    page_end = _page_number(metadata.get("page_end"))
    page_label = metadata.get("page_label")

    if page_start is None:
        zero_based_page = metadata.get("page")
        if isinstance(zero_based_page, int) and not isinstance(zero_based_page, bool):
            page_start = zero_based_page + 1
    if page_end is None:
        page_end = page_start

    start_ms = metadata.get("start_ms")
    end_ms = metadata.get("end_ms")
    return SourceLocator(
        page_start=page_start,
        page_end=page_end,
        page_label=str(page_label) if page_label not in (None, "") else None,
        start_ms=start_ms if isinstance(start_ms, int) else None,
        end_ms=end_ms if isinstance(end_ms, int) else None,
    )


def legacy_document_to_v3(
    document: Document,
    *,
    document_id: str,
    retrieval_schema_id: str,
) -> tuple[Artifact, SearchRecord]:
    """Wrap one existing V2 chunk without changing its searchable content."""

    metadata = dict(document.metadata)
    legacy_type = str(metadata.get("type") or "content")
    if legacy_type == "image":
        artifact_kind = ArtifactKind.STANDALONE_IMAGE
        representation_kind = RepresentationKind.CAPTION
        grounding_basis = GroundingBasis.VISION_DESCRIPTION
    elif legacy_type == "audio":
        artifact_kind = ArtifactKind.AUDIO_SEGMENT
        representation_kind = RepresentationKind.TRANSCRIPT
        grounding_basis = GroundingBasis.AUDIO_TRANSCRIPT
    else:
        artifact_kind = ArtifactKind.TEXT_BLOCK
        representation_kind = (
            RepresentationKind.PAGE_SUMMARY
            if legacy_type == "overview"
            else RepresentationKind.NATIVE_TEXT
        )
        grounding_basis = GroundingBasis.DIRECT_TEXT

    derivation = Derivation(
        producer="legacy_v2_adapter",
        revision="1",
        parameters={
            "chunking_strategy": str(metadata.get("chunking_strategy") or "unknown"),
            "extraction_mode": str(metadata.get("extraction_mode") or "unknown"),
        },
    )
    artifact = Artifact.from_text(
        document_id=document_id,
        kind=artifact_kind,
        text=document.page_content,
        derivation=derivation,
        locator=_source_locator(metadata),
        metadata={
            "legacy_type": legacy_type,
            "original_asset_available": False,
        },
    )
    record = SearchRecord.from_artifact(
        artifact,
        representation_kind=representation_kind,
        search_text=document.page_content,
        grounding_basis=grounding_basis,
        retrieval_schema_id=retrieval_schema_id,
        metadata={
            "filename": str(metadata.get("filename") or metadata.get("source") or ""),
            "legacy_type": legacy_type,
        },
    )
    return artifact, record
