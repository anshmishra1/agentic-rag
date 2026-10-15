"""Shared query-time candidate normalization and deduplication."""

from __future__ import annotations

import hashlib
from typing import Any


def _candidate_identity(document: Any) -> tuple[str, str, str, str]:
    metadata = document.metadata
    normalized_text = " ".join(document.page_content.lower().split())
    content_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:16]
    return (
        str(metadata.get("document_id", "")),
        str(metadata.get("type", "")),
        str(metadata.get("page_label", metadata.get("page", ""))),
        content_hash,
    )


def deduplicate_candidates(candidates: list[tuple]) -> tuple[list[tuple], int]:
    """Keep the first RRF-ranked occurrence of equivalent passage content."""

    seen = set()
    deduplicated = []
    removed = 0
    for candidate in candidates:
        identity = _candidate_identity(candidate[0])
        if identity in seen:
            removed += 1
            continue
        seen.add(identity)
        deduplicated.append(candidate)
    return deduplicated, removed
