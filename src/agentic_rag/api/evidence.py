"""Translate cited V3 retrieval documents into answer-facing evidence refs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agentic_rag.policies.citations import citation_description
from agentic_rag.policies.generation import select_generation_documents


def _citation_label(description: str) -> str:
    return description.split(" ", 1)[0]


def build_structured_evidence(
    documents: Sequence,
    citations: Sequence[str],
    *,
    retrieval_version: str,
) -> list[dict[str, Any]]:
    """Return cited V3 provenance without passages, binary data, or host paths."""

    if retrieval_version != "v3":
        return []

    used_labels = {_citation_label(citation) for citation in citations}
    evidence: list[dict[str, Any]] = []
    for index, document in enumerate(
        select_generation_documents(list(documents)), start=1
    ):
        label = f"[S{index}]"
        if label not in used_labels:
            continue
        metadata = document.metadata or {}
        artifact_id = str(metadata.get("artifact_id") or "").strip()
        record_id = str(metadata.get("record_id") or "").strip()
        document_id = str(metadata.get("document_id") or "").strip()
        if not artifact_id or not record_id or not document_id:
            continue

        bbox_values = (
            metadata.get("bbox_left"),
            metadata.get("bbox_top"),
            metadata.get("bbox_right"),
            metadata.get("bbox_bottom"),
        )
        bbox = (
            [float(value) for value in bbox_values]
            if all(value is not None for value in bbox_values)
            else None
        )
        description = citation_description(document, index)
        evidence.append(
            {
                "label": label.removeprefix("[").removesuffix("]"),
                "document_id": document_id,
                "artifact_id": artifact_id,
                "record_id": record_id,
                "kind": str(metadata.get("artifact_kind") or "text_block"),
                "grounding_basis": str(
                    metadata.get("grounding_basis") or "direct_text"
                ),
                "source_name": str(
                    metadata.get("filename")
                    or metadata.get("source")
                    or "selected document"
                ),
                "description": description,
                "locator": {
                    "page_start": metadata.get("page_start"),
                    "page_end": metadata.get("page_end"),
                    "page_label": metadata.get("page_label"),
                    "bbox": bbox,
                    "start_ms": metadata.get("start_ms"),
                    "end_ms": metadata.get("end_ms"),
                },
                "asset_url": (
                    f"/v3/documents/{document_id}/artifacts/{artifact_id}/asset"
                ),
            }
        )
    return evidence
