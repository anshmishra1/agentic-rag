"""Pure Streamlit-facing policies for versioned ingestion and document selection."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def partition_uploads(files: Iterable) -> tuple[list, list]:
    """Route PDFs to V3 and retain V2 for currently supported non-PDF inputs."""

    v3_files = []
    v2_files = []
    for upload in files:
        if Path(upload.name).suffix.casefold() == ".pdf":
            v3_files.append(upload)
        else:
            v2_files.append(upload)
    return v3_files, v2_files


def document_key(document: dict) -> str:
    version = str(document.get("retrieval_version") or "v2")
    return f"{version}:{document['document_id']}"


def document_label(document: dict) -> str:
    version = str(document.get("retrieval_version") or "v2").upper()
    count = int(document.get("record_count", document.get("chunk_count", 0)))
    unit = "records" if version == "V3" else "chunks"
    short_id = str(document["document_id"])[:8]
    return f"{document['filename']} · {version} · {count} {unit} · {short_id}"


def ingestion_summary(result: dict) -> str:
    filename = result["filename"]
    if result.get("retrieval_version") == "v3":
        return (
            f"{filename}: {result['pages_parsed']} pages parsed, "
            f"{result['artifacts_stored']} artifacts stored, and "
            f"{result['search_records_indexed']} search records indexed"
        )
    return f"{filename}: {result['chunks_indexed']} chunks indexed"
