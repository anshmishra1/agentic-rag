"""Isolated Pinecone writer for multimodal V3 text-proxy records.

The V2 vector store remains untouched. All dependencies are supplied explicitly
so offline tests can verify exact vector payloads, write ordering, and pruning
without initializing the application model or contacting Pinecone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from agentic_rag.ingestion.contracts import RetrievalSchema, SearchRecord
from agentic_rag.ingestion.parsers.base import ParseRun


class TextEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class SparseEncoder(Protocol):
    def encode_documents(self, text: str) -> dict[str, list]: ...


class VectorIndex(Protocol):
    def upsert(self, *, vectors: list[dict[str, Any]]) -> Any: ...

    def list(self, *, prefix: str): ...

    def delete(self, *, ids: list[str]) -> Any: ...


@dataclass(frozen=True)
class V3WriteResult:
    document_id: str
    retrieval_schema_id: str
    written_ids: frozenset[str]
    obsolete_deleted: int

    @property
    def vector_count(self) -> int:
        return len(self.written_ids)


def document_vector_prefix(document_id: str) -> str:
    normalized = document_id.strip()
    if not normalized:
        raise ValueError("document_id must be nonempty")
    return f"{normalized}-v3-record-"


def vector_id_for_record(record: SearchRecord) -> str:
    record_digest = record.record_id.removeprefix("record_")
    return f"{document_vector_prefix(record.document_id)}{record_digest}"


def _validate_sparse(value: dict[str, list]) -> dict[str, list]:
    indices = value.get("indices")
    values = value.get("values")
    if not isinstance(indices, list) or not isinstance(values, list):
        raise ValueError("sparse encoder must return indices and values lists")
    if len(indices) != len(values):
        raise ValueError("sparse indices and values must have equal lengths")
    if any(not isinstance(index, int) or index < 0 for index in indices):
        raise ValueError("sparse indices must be nonnegative integers")
    if any(not isinstance(item, (int, float)) for item in values):
        raise ValueError("sparse values must be numeric")
    return {
        "indices": indices,
        "values": [float(item) for item in values],
    }


def _metadata(
    *,
    index_name: str,
    source_name: str,
    record: SearchRecord,
    artifact,
) -> dict[str, Any]:
    locator = artifact.locator
    metadata: dict[str, Any] = {
        "text": record.search_text,
        "index_name": index_name,
        "source": source_name,
        "filename": source_name,
        "document_id": record.document_id,
        "artifact_id": record.artifact_id,
        "record_id": record.record_id,
        "artifact_kind": artifact.kind.value,
        "representation_kind": record.representation_kind.value,
        "grounding_basis": record.grounding_basis.value,
        "retrieval_schema_id": record.retrieval_schema_id,
        "type": "content",
    }
    optional = {
        "parent_artifact_id": artifact.parent_artifact_id,
        "page_start": locator.page_start,
        "page_end": locator.page_end,
        "page_label": locator.page_label,
        "start_ms": locator.start_ms,
        "end_ms": locator.end_ms,
        "parent_record_id": record.metadata.get("parent_record_id"),
        "segment_index": record.metadata.get("segment_index"),
        "segment_count": record.metadata.get("segment_count"),
    }
    metadata.update(
        {key: value for key, value in optional.items() if value is not None}
    )
    if locator.bbox is not None:
        left, top, right, bottom = locator.bbox
        metadata.update(
            {
                "bbox_left": left,
                "bbox_top": top,
                "bbox_right": right,
                "bbox_bottom": bottom,
            }
        )
    return metadata


def build_vector_payloads(
    *,
    index_name: str,
    source_name: str,
    schema: RetrievalSchema,
    run: ParseRun,
    dense_vectors: Sequence[Sequence[float]],
    sparse_vectors: Sequence[dict[str, list]],
) -> list[dict[str, Any]]:
    index_name = index_name.strip()
    source_name = source_name.strip()
    if not index_name or not source_name:
        raise ValueError("index_name and source_name must be nonempty")
    if run.retrieval_schema_id != schema.schema_id:
        raise ValueError("parse run and retrieval schema identities differ")
    if run.parser_name != schema.parser_name:
        raise ValueError("parse run and retrieval schema parser names differ")
    if run.parser_revision != schema.parser_revision:
        raise ValueError("parse run and retrieval schema parser revisions differ")
    if not run.records:
        raise ValueError("V3 indexing requires at least one search record")
    if len(dense_vectors) != len(run.records):
        raise ValueError("dense vector count must match search-record count")
    if len(sparse_vectors) != len(run.records):
        raise ValueError("sparse vector count must match search-record count")

    artifacts = {artifact.artifact_id: artifact for artifact in run.artifacts}
    payloads: list[dict[str, Any]] = []
    for record, dense, sparse in zip(run.records, dense_vectors, sparse_vectors):
        values = [float(item) for item in dense]
        if len(values) != schema.text_embedding_dimension:
            raise ValueError(
                "dense vector dimension does not match retrieval schema"
            )
        artifact = artifacts[record.artifact_id]
        sparse_values = _validate_sparse(sparse)
        payload = {
            "id": vector_id_for_record(record),
            "values": values,
            "metadata": _metadata(
                index_name=index_name,
                source_name=source_name,
                record=record,
                artifact=artifact,
            ),
        }
        if sparse_values["values"]:
            payload["sparse_values"] = sparse_values
        payloads.append(payload)
    return payloads


def _prune_obsolete(
    *,
    index: VectorIndex,
    document_id: str,
    keep_ids: frozenset[str],
    batch_size: int,
) -> int:
    prefix = document_vector_prefix(document_id)
    if not keep_ids or any(not value.startswith(prefix) for value in keep_ids):
        raise ValueError("keep_ids must contain V3 IDs for one document")
    obsolete = [
        vector_id
        for batch in index.list(prefix=prefix)
        for vector_id in batch
        if vector_id not in keep_ids
    ]
    for start in range(0, len(obsolete), batch_size):
        index.delete(ids=obsolete[start : start + batch_size])
    return len(obsolete)


def write_parse_run(
    *,
    index: VectorIndex,
    index_name: str,
    source_name: str,
    schema: RetrievalSchema,
    run: ParseRun,
    embedder: TextEmbedder,
    sparse_encoder: SparseEncoder,
    batch_size: int = 100,
) -> V3WriteResult:
    """Write one complete V3 document, then prune its obsolete vector IDs.

    Pruning happens only after all current batches have been accepted. A failed
    write therefore cannot erase the previously usable document version.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    texts = [record.search_text for record in run.records]
    if not texts:
        raise ValueError("V3 indexing requires at least one search record")
    dense_vectors = embedder.embed_documents(texts)
    sparse_vectors = [sparse_encoder.encode_documents(text) for text in texts]
    payloads = build_vector_payloads(
        index_name=index_name,
        source_name=source_name,
        schema=schema,
        run=run,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
    )

    for start in range(0, len(payloads), batch_size):
        index.upsert(vectors=payloads[start : start + batch_size])

    written_ids = frozenset(payload["id"] for payload in payloads)
    obsolete_deleted = _prune_obsolete(
        index=index,
        document_id=run.document_id,
        keep_ids=written_ids,
        batch_size=1000,
    )
    return V3WriteResult(
        document_id=run.document_id,
        retrieval_schema_id=schema.schema_id,
        written_ids=written_ids,
        obsolete_deleted=obsolete_deleted,
    )
