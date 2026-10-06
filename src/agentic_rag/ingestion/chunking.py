"""Versioned token-window and structure-aware document chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from agentic_rag.config import settings
from agentic_rag.model_identity import canonical_model_name


TOKEN_WINDOW_V1 = "token_window_v1"
STRUCTURE_AWARE_V2 = "structure_aware_v2"
CHUNKING_STRATEGIES = {TOKEN_WINDOW_V1, STRUCTURE_AWARE_V2}

_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*|[A-Z]\.)\s+[A-Z][^.!?]{1,100}$")
_KNOWN_HEADING = re.compile(
    r"^(?:abstract|introduction|background|related work|method(?:ology)?|methods|analysis|"
    r"experiments?|results?|discussion|conclusions?|limitations?|references|"
    r"appendix(?:\s+(?:[A-Z]|\d+)(?:\.\d+)?)?(?::\s+[^.!?]{1,80})?|"
    r"acknowledg(?:e)?ments?|what are agents\?|when and how to use frameworks|"
    r"when \(and when not\) to use agents|when to use .*|examples where .* useful:?|"
    r"building effective agents|"
    r"building blocks?, workflows?, and agents|building block:\s+.*|workflow:\s+.*|"
    r"agents|agents in practice)$",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^(?:[-*•]|\d+[.)])\s+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass(frozen=True)
class _TextUnit:
    text: str
    section_title: str
    page_index: int | None
    page_label: str
    extraction_mode: str


@lru_cache(maxsize=1)
def _get_tokenizer():
    return AutoTokenizer.from_pretrained(
        canonical_model_name(settings.embedding_model),
        revision=settings.embedding_model_revision,
        local_files_only=settings.model_local_files_only,
    )


@lru_cache(maxsize=1)
def _get_token_window_splitter() -> RecursiveCharacterTextSplitter:
    """Return the exact legacy splitter used by the v1 index."""
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _get_tokenizer(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def pdf_extraction_mode(strategy: str) -> str:
    """Keep v1 parsing stable and preserve line layout for v2 structure cues."""
    if strategy == TOKEN_WINDOW_V1:
        return "plain"
    if strategy == STRUCTURE_AWARE_V2:
        return "layout"
    raise ValueError(f"Unsupported chunking strategy: {strategy}")


def _looks_like_heading(line: str) -> bool:
    if not 2 <= len(line) <= 120 or len(line.split()) > 14:
        return False
    if _NUMBERED_HEADING.fullmatch(line) or _KNOWN_HEADING.fullmatch(line):
        return True
    ascii_letters = re.findall(r"[A-Za-z]", line)
    visible = [character for character in line if not character.isspace()]
    return (
        len(ascii_letters) >= 4
        and len(ascii_letters) / len(visible) >= 0.65
        and line.upper() == line
        and not line.endswith(".")
    )


def _clean_lines(text: str) -> list[str]:
    """Collapse layout spacing and repair ordinary line-wrap hyphenation."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        if lines and lines[-1].endswith("-") and line[0].islower():
            lines[-1] = lines[-1][:-1] + line
        else:
            lines.append(line)
    return lines


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]


def _structure_units(docs: list[Document]) -> list[_TextUnit]:
    """Build sentence-level units under headings while retaining page provenance."""
    units: list[_TextUnit] = []
    active_heading = ""

    for fallback_page, document in enumerate(docs):
        metadata = document.metadata
        page_index = metadata.get("page")
        if not isinstance(page_index, int):
            page_index = fallback_page
        page_label = str(metadata.get("page_label", page_index + 1))
        extraction_mode = str(metadata.get("extraction_mode", "plain"))
        body_lines: list[str] = []

        def flush_body() -> None:
            if not body_lines:
                return
            body = " ".join(body_lines)
            body_lines.clear()
            for sentence in _sentences(body):
                units.append(
                    _TextUnit(
                        text=sentence,
                        section_title=active_heading,
                        page_index=page_index,
                        page_label=page_label,
                        extraction_mode=extraction_mode,
                    )
                )

        for line in _clean_lines(document.page_content):
            if _looks_like_heading(line):
                flush_body()
                active_heading = line.rstrip(":").strip()
            elif _BULLET.match(line):
                flush_body()
                units.append(
                    _TextUnit(
                        text=line,
                        section_title=active_heading,
                        page_index=page_index,
                        page_label=page_label,
                        extraction_mode=extraction_mode,
                    )
                )
            else:
                body_lines.append(line)
        flush_body()

    return units


def _token_count(text: str) -> int:
    return len(
        _get_tokenizer().encode(
            text,
            add_special_tokens=False,
            verbose=False,
        )
    )


def _render_chunk(section_title: str, body: str) -> str:
    return f"{section_title}\n\n{body}" if section_title else body


def _chunk_metadata(units: list[_TextUnit], section_title: str) -> dict:
    page_labels = list(dict.fromkeys(unit.page_label for unit in units))
    page_indices = [unit.page_index for unit in units if unit.page_index is not None]
    page_start = page_labels[0] if page_labels else ""
    page_end = page_labels[-1] if page_labels else page_start
    metadata = {
        "type": "content",
        "chunking_strategy": STRUCTURE_AWARE_V2,
        "section_title": section_title,
        "page_label": page_start if page_start == page_end else f"{page_start}-{page_end}",
        "page_start": page_start,
        "page_end": page_end,
        "source_unit_count": len(units),
        "extraction_mode": units[0].extraction_mode if units else "layout",
    }
    if page_indices:
        metadata["page"] = min(page_indices)
    return metadata


def _split_oversized_unit(unit: _TextUnit) -> list[str]:
    """Split a source unit only when it cannot fit beneath the model-safe cap."""
    heading_tokens = _token_count(unit.section_title) if unit.section_title else 0
    body_limit = max(1, settings.chunk_size - heading_tokens - 2)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _get_tokenizer(),
        chunk_size=body_limit,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    return splitter.split_text(unit.text)


def _structure_aware_chunks(docs: list[Document]) -> list[Document]:
    units = _structure_units(docs)
    chunks: list[Document] = []
    bucket: list[_TextUnit] = []
    active_section = ""

    def flush_bucket() -> None:
        if not bucket:
            return
        body = " ".join(unit.text for unit in bucket)
        chunks.append(
            Document(
                page_content=_render_chunk(active_section, body),
                metadata=_chunk_metadata(bucket, active_section),
            )
        )
        bucket.clear()

    for unit in units:
        if bucket and unit.section_title != active_section:
            flush_bucket()
        active_section = unit.section_title

        candidate_units = [*bucket, unit]
        candidate_body = " ".join(item.text for item in candidate_units)
        if _token_count(_render_chunk(active_section, candidate_body)) <= settings.chunk_size:
            bucket.append(unit)
            continue

        flush_bucket()
        active_section = unit.section_title
        if _token_count(_render_chunk(active_section, unit.text)) <= settings.chunk_size:
            bucket.append(unit)
            continue

        for fragment in _split_oversized_unit(unit):
            fragment_unit = _TextUnit(
                text=fragment,
                section_title=active_section,
                page_index=unit.page_index,
                page_label=unit.page_label,
                extraction_mode=unit.extraction_mode,
            )
            chunks.append(
                Document(
                    page_content=_render_chunk(active_section, fragment),
                    metadata=_chunk_metadata([fragment_unit], active_section),
                )
            )

    flush_bucket()
    return _deduplicate_chunks(chunks)


def _deduplicate_chunks(chunks: list[Document]) -> list[Document]:
    """Merge exact extraction duplicates before content-addressed vector IDs collide."""
    unique: dict[str, Document] = {}
    for chunk in chunks:
        existing = unique.get(chunk.page_content)
        if existing is None:
            chunk.metadata["duplicate_occurrences"] = 1
            unique[chunk.page_content] = chunk
            continue

        metadata = existing.metadata
        metadata["duplicate_occurrences"] += 1
        metadata["source_unit_count"] += chunk.metadata.get("source_unit_count", 0)
        metadata["page_end"] = chunk.metadata.get("page_end", metadata["page_end"])
        if metadata["page_start"] != metadata["page_end"]:
            metadata["page_label"] = f"{metadata['page_start']}-{metadata['page_end']}"
    return list(unique.values())


def _enrich_chunks(
    chunks: list[Document],
    *,
    strategy: str,
    document_id: str | None,
    filename: str | None,
) -> list[Document]:
    for chunk in chunks:
        if document_id is not None:
            chunk.metadata["document_id"] = document_id
        if filename is not None:
            chunk.metadata["filename"] = filename
            chunk.metadata["source"] = filename
        chunk.metadata.setdefault("type", "content")
        chunk.metadata["chunking_strategy"] = strategy
        chunk.metadata["token_count"] = _token_count(chunk.page_content)
        page_label = str(chunk.metadata.get("page_label", chunk.metadata.get("page", "")))
        chunk.metadata.setdefault("page_start", page_label)
        chunk.metadata.setdefault("page_end", page_label)
    return chunks


def chunk_documents(
    docs: list[Document],
    *,
    document_id: str | None = None,
    filename: str | None = None,
    strategy: str | None = None,
) -> list[Document]:
    """Chunk documents with an explicit, versioned retrieval strategy."""
    selected = strategy or settings.chunking_strategy
    if selected == TOKEN_WINDOW_V1:
        chunks = _get_token_window_splitter().split_documents(docs)
    elif selected == STRUCTURE_AWARE_V2:
        chunks = _structure_aware_chunks(docs)
    else:
        raise ValueError(f"Unsupported chunking strategy: {selected}")
    return _enrich_chunks(
        chunks,
        strategy=selected,
        document_id=document_id,
        filename=filename,
    )
