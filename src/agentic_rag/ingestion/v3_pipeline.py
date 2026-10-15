"""Fail-closed coordination for persisted V3 parse runs and vector writes."""

from __future__ import annotations

from agentic_rag.ingestion.contracts import RetrievalSchema
from agentic_rag.ingestion.manifest_store import ManifestStore, ManifestWriteResult
from agentic_rag.ingestion.parsers.base import ParseRun
from agentic_rag.retrieval.v3_writer import (
    SparseEncoder,
    TextEmbedder,
    V3WriteResult,
    VectorIndex,
    write_parse_run,
)


def persist_and_index_parse_run(
    *,
    manifest_store: ManifestStore,
    index: VectorIndex,
    index_name: str,
    source_name: str,
    schema: RetrievalSchema,
    run: ParseRun,
    embedder: TextEmbedder,
    sparse_encoder: SparseEncoder,
    bm25_params: str,
) -> tuple[ManifestWriteResult, V3WriteResult]:
    """Persist a parsed manifest, write vectors, then mark it query-ready.

    A vector or readiness failure leaves the manifest in ``parsed`` state. The
    future V3 retrieval path must expose only ``ready`` documents, so partial
    cross-store work fails closed and can be repaired by an idempotent rerun.
    """

    manifest = manifest_store.replace_parse_run(
        index_name=index_name,
        source_name=source_name,
        schema=schema,
        run=run,
    )
    vectors = write_parse_run(
        index=index,
        index_name=index_name,
        source_name=source_name,
        schema=schema,
        run=run,
        embedder=embedder,
        sparse_encoder=sparse_encoder,
    )
    manifest_store.mark_ready(
        index_name=index_name,
        document_id=run.document_id,
        retrieval_schema_id=schema.schema_id,
        vector_count=vectors.vector_count,
        bm25_params=bm25_params,
    )
    return manifest, vectors
