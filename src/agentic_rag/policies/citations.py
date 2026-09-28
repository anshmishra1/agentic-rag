"""Deterministic citation labels for retrieved generation context."""

from __future__ import annotations

import re
from collections.abc import Sequence


_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")
_ALTERNATE_CITATION_PATTERN = re.compile(r"【S(\d+)】")


def normalize_citation_markers(answer: str) -> str:
    """Convert the observed alternate source brackets to canonical labels."""
    return _ALTERNATE_CITATION_PATTERN.sub(
        lambda match: f"[S{match.group(1)}]",
        answer,
    )


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


def _labeled_blocks(
    documents: Sequence,
    *,
    max_documents: int,
    max_chars: int,
) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
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

        blocks.append((description, block))
        current_length += len(separator) + len(block)

        if len(block) >= available:
            break

    return blocks


def build_citation_context(
    documents: Sequence,
    *,
    max_documents: int,
    max_chars: int,
) -> tuple[str, list[str]]:
    """Build bounded, labeled context and its matching citation catalog."""
    blocks = _labeled_blocks(
        documents, max_documents=max_documents, max_chars=max_chars
    )
    return "\n\n".join(block for _, block in blocks), [label for label, _ in blocks]


def cited_verification_context(
    documents: Sequence,
    answer: str,
    *,
    max_documents: int,
    max_chars: int,
) -> tuple[str, list[str]]:
    """Limit verification evidence to cited labels and report citation gaps."""
    blocks = _labeled_blocks(
        documents, max_documents=max_documents, max_chars=max_chars
    )
    by_label = {description.split(" ", 1)[0]: block for description, block in blocks}
    cited = {f"[S{match.group(1)}]" for match in _CITATION_PATTERN.finditer(answer)}
    issues: list[str] = []
    if not cited:
        issues.append("The answer has no source citation.")
    for label in sorted(cited - by_label.keys()):
        issues.append(f"The answer cites unavailable source {label}.")

    lines = answer.splitlines()
    for index, line in enumerate(lines):
        if not _BULLET_PATTERN.match(line):
            continue
        item = [line]
        for continuation in lines[index + 1:]:
            if not continuation.strip() or _BULLET_PATTERN.match(continuation):
                break
            item.append(continuation)
        if not _CITATION_PATTERN.search("\n".join(item)):
            issues.append(f"List item lacks a source citation: {line.strip()[:160]}")

    selected = [
        block for description, block in blocks
        if description.split(" ", 1)[0] in cited
    ]
    return "\n\n".join(selected), issues


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
