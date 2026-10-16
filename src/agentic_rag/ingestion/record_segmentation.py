"""Model-safe segmentation for V3 searchable artifact representations."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from agentic_rag.ingestion.contracts import SearchRecord
from agentic_rag.ingestion.parsers.base import ParseRun
from agentic_rag.model_identity import canonical_model_name


@lru_cache(maxsize=2)
def _tokenizer(model: str, revision: str, local_files_only: bool):
    return AutoTokenizer.from_pretrained(
        canonical_model_name(model),
        revision=revision,
        local_files_only=local_files_only,
    )


def _token_count(tokenizer, text: str) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
            verbose=False,
        )
    )


def segment_parse_run_records(
    run: ParseRun,
    *,
    embedding_model: str,
    embedding_revision: str,
    local_files_only: bool,
    max_tokens: int,
    tokenizer: Any | None = None,
    splitter: Any | None = None,
) -> ParseRun:
    """Split only oversized search proxies; preserve source artifacts intact."""

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    tokenizer = tokenizer or _tokenizer(
        embedding_model,
        embedding_revision,
        local_files_only,
    )
    splitter = splitter or RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=max_tokens,
        chunk_overlap=0,
        separators=["\n\n", "\n", "|", ". ", "; ", ", ", " ", ""],
    )
    segmented: list[SearchRecord] = []
    split_record_count = 0

    for record in run.records:
        if _token_count(tokenizer, record.search_text) <= max_tokens:
            segmented.append(record)
            continue

        parts = [part.strip() for part in splitter.split_text(record.search_text) if part.strip()]
        if not parts:
            raise ValueError("V3 search-record segmentation produced no text")
        split_record_count += 1
        for index, part in enumerate(parts):
            metadata = dict(record.metadata)
            metadata.update(
                {
                    "parent_record_id": record.record_id,
                    "segment_index": index,
                    "segment_count": len(parts),
                }
            )
            segmented.append(
                SearchRecord(
                    document_id=record.document_id,
                    artifact_id=record.artifact_id,
                    representation_kind=record.representation_kind,
                    search_text=part,
                    grounding_basis=record.grounding_basis,
                    retrieval_schema_id=record.retrieval_schema_id,
                    metadata=metadata,
                )
            )

    if not split_record_count:
        return run

    return ParseRun(
        parser_name=run.parser_name,
        parser_revision=run.parser_revision,
        document_id=run.document_id,
        retrieval_schema_id=run.retrieval_schema_id,
        selected_pages=run.selected_pages,
        artifacts=run.artifacts,
        records=tuple(segmented),
        elapsed_seconds=run.elapsed_seconds,
        peak_memory_mib=run.peak_memory_mib,
        warnings=(
            *run.warnings,
            f"segmented_search_records={split_record_count}",
        ),
    )
