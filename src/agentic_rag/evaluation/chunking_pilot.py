"""Prepare a reproducible, human-reviewed chunking pilot from local PDFs.

The tracked manifest contains document identities, questions, and gold chunk
IDs. Full extracted passage text is written only to an ignored review catalog.
No database, vector store, embedding, reranker, or hosted model is used here.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVIEW_STATUSES = {"pending", "verified"}
SPLITS = {"development", "holdout"}


def file_digest(path: Path) -> str:
    """Return a streaming SHA-256 digest without retaining PDF bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_chunk_id(document_id: str, text: str) -> str:
    """Mirror the production content-vector ID contract without importing it.

    Importing ``retrieval.vectorstore`` initializes the embedding model. This
    preparation step only needs the deterministic ID formula used by upsert.
    """
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{document_id}-content-{content_hash}"


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the tracked pilot contract before reading any source PDFs."""
    if manifest.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if not isinstance(manifest.get("pilot_id"), str) or not manifest["pilot_id"].strip():
        raise ValueError("pilot_id must be a nonempty string")
    if not isinstance(manifest.get("index_name"), str) or not manifest["index_name"].strip():
        raise ValueError("index_name must be a nonempty string")

    retrieval_schema = manifest.get("retrieval_schema")
    required_schema = {
        "embedding_model",
        "embedding_model_revision",
        "chunk_size_tokens",
        "chunk_overlap_tokens",
        "chunking_strategy",
        "vector_id_contract",
    }
    if not isinstance(retrieval_schema, dict) or not required_schema.issubset(retrieval_schema):
        raise ValueError("retrieval_schema is incomplete")
    if not isinstance(retrieval_schema["chunk_size_tokens"], int) or retrieval_schema["chunk_size_tokens"] < 1:
        raise ValueError("chunk_size_tokens must be a positive integer")
    if not isinstance(retrieval_schema["chunk_overlap_tokens"], int) or not (
        0 <= retrieval_schema["chunk_overlap_tokens"] < retrieval_schema["chunk_size_tokens"]
    ):
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    documents = manifest.get("documents")
    if not isinstance(documents, list) or len(documents) < 2:
        raise ValueError("documents must contain at least two entries")
    document_keys: set[str] = set()
    filenames: set[str] = set()
    for position, document in enumerate(documents, start=1):
        if not isinstance(document, dict):
            raise ValueError(f"Document {position} must be an object")
        key = document.get("document_key")
        filename = document.get("filename")
        if not isinstance(key, str) or not key or key in document_keys:
            raise ValueError(f"Document {position} needs a unique document_key")
        if not isinstance(filename, str) or Path(filename).name != filename or filename in filenames:
            raise ValueError(f"Document {position} needs a unique safe filename")
        if not isinstance(document.get("sha256"), str) or not SHA256.fullmatch(document["sha256"]):
            raise ValueError(f"Document {position} needs a lowercase SHA-256 digest")
        if not isinstance(document.get("size_bytes"), int) or document["size_bytes"] < 1:
            raise ValueError(f"Document {position} needs a positive size_bytes")
        document_keys.add(key)
        filenames.add(filename)

    questions = manifest.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a nonempty list")
    question_ids: set[str] = set()
    splits_by_document: dict[str, set[str]] = {key: set() for key in document_keys}
    for position, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"Question {position} must be an object")
        question_id = question.get("question_id")
        document_key = question.get("source_document_key")
        status = question.get("review_status")
        if not isinstance(question_id, str) or not question_id or question_id in question_ids:
            raise ValueError(f"Question {position} needs a unique question_id")
        if document_key not in document_keys:
            raise ValueError(f"Question {question_id} references an unknown document")
        if not isinstance(question.get("query"), str) or not question["query"].strip():
            raise ValueError(f"Question {question_id} needs a nonempty query")
        if question.get("split") not in SPLITS:
            raise ValueError(f"Question {question_id} needs a development or holdout split")
        if status not in REVIEW_STATUSES:
            raise ValueError(f"Question {question_id} has an invalid review_status")
        if not isinstance(question.get("source_pages"), list) or any(
            not isinstance(page, str) or not page.strip() for page in question["source_pages"]
        ):
            raise ValueError(f"Question {question_id} needs a source_pages list")
        chunk_ids = question.get("relevant_chunk_ids")
        if not isinstance(chunk_ids, list) or any(not isinstance(value, str) or not value for value in chunk_ids):
            raise ValueError(f"Question {question_id} needs a relevant_chunk_ids list")
        if status == "verified" and (
            not question["source_pages"]
            or not chunk_ids
            or not isinstance(question.get("review_notes"), str)
            or not question["review_notes"].strip()
        ):
            raise ValueError(f"Verified question {question_id} needs pages, chunk IDs, and review notes")
        if status == "pending" and chunk_ids:
            raise ValueError(f"Pending question {question_id} cannot claim gold chunk IDs")
        question_ids.add(question_id)
        splits_by_document[document_key].add(question["split"])

    incomplete = sorted(key for key, splits in splits_by_document.items() if splits != SPLITS)
    if incomplete:
        raise ValueError(f"Every document needs development and holdout questions: {', '.join(incomplete)}")


def validate_runtime_schema(
    manifest: dict[str, Any],
    *,
    embedding_model: str,
    embedding_model_revision: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> None:
    """Reject catalogs produced with settings that differ from the manifest."""
    expected = manifest["retrieval_schema"]
    actual = {
        "embedding_model": embedding_model,
        "embedding_model_revision": embedding_model_revision,
        "chunk_size_tokens": chunk_size_tokens,
        "chunk_overlap_tokens": chunk_overlap_tokens,
    }
    mismatches = [name for name, value in actual.items() if expected.get(name) != value]
    if mismatches:
        raise ValueError(f"Runtime retrieval schema differs from manifest: {', '.join(mismatches)}")


def build_catalog(
    manifest: dict[str, Any],
    pdf_dir: Path,
    *,
    load_pdf: Callable[[Path], list[Any]],
    chunk_documents: Callable[..., list[Any]],
    strategy: str,
) -> dict[str, Any]:
    """Extract current production chunks and full text for local review."""
    validate_manifest(manifest)
    document_reports: list[dict[str, Any]] = []
    catalog_chunks: list[dict[str, Any]] = []

    for document in manifest["documents"]:
        source = pdf_dir / document["filename"]
        if not source.is_file():
            raise FileNotFoundError(f"Pilot PDF not found: {source}")
        actual_size = source.stat().st_size
        actual_digest = file_digest(source)
        if actual_size != document["size_bytes"] or actual_digest != document["sha256"]:
            raise ValueError(f"Pilot PDF identity differs from manifest: {document['filename']}")

        pages = load_pdf(source)
        chunks = chunk_documents(
            pages,
            document_id=document["sha256"],
            filename=document["filename"],
            strategy=strategy,
        )
        empty_pages = [
            str(page.metadata.get("page_label", page.metadata.get("page", position)))
            for position, page in enumerate(pages, start=1)
            if not page.page_content.strip()
        ]
        seen_chunk_ids: set[str] = set()
        for chunk in chunks:
            chunk_id = stable_chunk_id(document["sha256"], chunk.page_content)
            if chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate content chunk in {document['filename']}: {chunk_id}")
            seen_chunk_ids.add(chunk_id)
            catalog_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_key": document["document_key"],
                    "filename": document["filename"],
                    "page_label": str(chunk.metadata.get("page_label", chunk.metadata.get("page", ""))),
                    "page_index": chunk.metadata.get("page"),
                    "page_start": str(chunk.metadata.get("page_start", "")),
                    "page_end": str(chunk.metadata.get("page_end", "")),
                    "section_title": str(chunk.metadata.get("section_title", "")),
                    "chunking_strategy": str(chunk.metadata.get("chunking_strategy", strategy)),
                    "token_count": chunk.metadata.get("token_count"),
                    "duplicate_occurrences": chunk.metadata.get("duplicate_occurrences", 1),
                    "extraction_mode": str(chunk.metadata.get("extraction_mode", "plain")),
                    "character_count": len(chunk.page_content),
                    "text": chunk.page_content,
                }
            )
        document_reports.append(
            {
                **document,
                "page_count": len(pages),
                "empty_pages": empty_pages,
                "content_chunk_count": len(chunks),
            }
        )

    return {
        "schema_version": 1,
        "pilot_id": manifest["pilot_id"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "index_name": manifest["index_name"],
        "retrieval_schema": {
            **manifest["retrieval_schema"],
            "chunking_strategy": strategy,
        },
        "documents": document_reports,
        "questions": manifest["questions"],
        "chunks": catalog_chunks,
    }


def export_reviewed_pairs(manifest: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Export verified source questions in the existing retrieval-pair format."""
    validate_manifest(manifest)
    chunks_by_document: dict[str, set[str]] = {}
    for chunk in catalog.get("chunks", []):
        chunks_by_document.setdefault(chunk["document_key"], set()).add(chunk["chunk_id"])
    documents = {document["document_key"]: document for document in manifest["documents"]}

    pairs: list[dict[str, Any]] = []
    for question in manifest["questions"]:
        if question["review_status"] != "verified":
            raise ValueError(f"Question {question['question_id']} has not been verified")
        available = chunks_by_document.get(question["source_document_key"], set())
        missing = sorted(set(question["relevant_chunk_ids"]) - available)
        if missing:
            raise ValueError(f"Question {question['question_id']} references chunks absent from the catalog")
        document = documents[question["source_document_key"]]
        pairs.append(
            {
                "case_id": question["question_id"],
                "query": question["query"],
                "document_id": document["sha256"],
                "should_match": True,
                "split": question["split"],
                "source_document": document["filename"],
                "target_document": document["filename"],
                "source_pages": "; ".join(question["source_pages"]),
                "relevant_chunk_ids": question["relevant_chunk_ids"],
            }
        )
    return pairs


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
