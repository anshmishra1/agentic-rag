"""Deterministic citation labels for retrieved generation context."""

from __future__ import annotations

import re
from collections.abc import Sequence


_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


def _source_name(document) -> str:
    metadata = document.metadata
    return str(
        metadata.get("filename")
        or metadata.get("source")
        or "selected document"
    )


def _page_value(document) -> str | None:
    metadata = document.metadata
    page = metadata.get("page_label", metadata.get("page"))
    return str(page) if page not in (None, "") else None


def citation_description(document, index: int) -> str:
    """Return the user-facing description for one labeled source."""
    label = f"[S{index}]"
    source = _source_name(document)
    page = _page_value(document)
    chunk_type = str(document.metadata.get("type") or "content")

    if page is not None:
        return f"{label} {source}, page {page}"
    return f"{label} {source}, {chunk_type}"


def build_citation_context(
    documents: Sequence,
    *,
    max_documents: int,
    max_chars: int,
) -> tuple[str, list[str]]:
    """Build bounded, labeled context and its matching citation catalog."""
    blocks: list[str] = []
    catalog: list[str] = []
    current_length = 0

    for index, document in enumerate(documents[:max_documents], start=1):
        description = citation_description(document, index)
        header = f"{description}\n"
        separator = "\n\n" if blocks else ""
        available = max_chars - current_length - len(separator)

        if available <= len(header):
            break

        content = document.page_content
        block = header + content
        if len(block) > available:
            marker = "\n[...truncated]"
            content_space = available - len(header)
            if content_space > len(marker):
                block = header + content[:content_space - len(marker)] + marker
            else:
                block = header + content[:content_space]

        blocks.append(separator + block)
        catalog.append(description)
        current_length += len(separator) + len(block)

        if len(block) >= available:
            break

    return "".join(blocks), catalog


def extract_used_citations(
    answer: str,
    catalog: Sequence[str],
) -> list[str]:
    """Return valid catalog entries cited by the answer, in first-use order."""
    by_label = {
        entry.split(" ", 1)[0]: entry
        for entry in catalog
    }
    used: list[str] = []
    seen: set[str] = set()

    for match in _CITATION_PATTERN.finditer(answer):
        label = f"[S{match.group(1)}]"
        if label in by_label and label not in seen:
            used.append(by_label[label])
            seen.add(label)

    return used
