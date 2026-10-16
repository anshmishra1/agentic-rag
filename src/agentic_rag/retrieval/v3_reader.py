"""Document-scoped hybrid reader for V3 artifact search records.

The pure retrieval function accepts every external dependency explicitly. The
small application wrapper constructs Pinecone and PostgreSQL adapters lazily so
V2 startup and offline imports do not contact either service.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from langchain_core.documents import Document


class ManifestReader(Protocol):
    def get_ready_manifest(
        self,
        *,
        index_name: str,
        document_id: str,
    ) -> dict[str, Any] | None: ...


class VectorIndex(Protocol):
    def query(self, **kwargs) -> Any: ...


def _matches(response: Any) -> list[Any]:
    if isinstance(response, Mapping):
        return list(response.get("matches", []))
    return list(getattr(response, "matches", []))


def _field(value: Any, name: str, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _match(match: Any) -> tuple[str, Document, float]:
    metadata = dict(_field(match, "metadata", {}) or {})
    text = str(metadata.pop("text", ""))
    return (
        str(_field(match, "id", "")),
        Document(page_content=text, metadata=metadata),
        float(_field(match, "score", 0.0)),
    )


def _weighted_rrf(
    dense_matches: list[tuple[str, Document, float]],
    sparse_matches: list[tuple[str, Document, float]],
    *,
    rrf_k: int,
    alpha: float,
) -> tuple[list[tuple[Document, float]], list[dict[str, Any]]]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")

    fused: dict[str, dict[str, Any]] = {}
    for rank, (vector_id, document, score) in enumerate(dense_matches, start=1):
        fused[vector_id] = {
            "vector_id": vector_id,
            "document": document,
            "dense_rank": rank,
            "dense_score": score,
            "sparse_rank": None,
            "sparse_score": None,
        }
    for rank, (vector_id, document, score) in enumerate(sparse_matches, start=1):
        item = fused.setdefault(
            vector_id,
            {
                "vector_id": vector_id,
                "document": document,
                "dense_rank": None,
                "dense_score": None,
                "sparse_rank": None,
                "sparse_score": None,
            },
        )
        item["sparse_rank"] = rank
        item["sparse_score"] = score

    ranked = []
    for item in fused.values():
        score = 0.0
        if item["dense_rank"] is not None:
            score += alpha / (rrf_k + item["dense_rank"])
        if item["sparse_rank"] is not None:
            score += (1 - alpha) / (rrf_k + item["sparse_rank"])
        item["rrf_score"] = score
        ranked.append(item)
    ranked.sort(key=lambda item: item["rrf_score"], reverse=True)

    return (
        [(item["document"], float(item["rrf_score"])) for item in ranked],
        ranked,
    )


def retrieve_v3_hybrid_with_scores(
    *,
    query: str,
    document_id: str,
    index_name: str,
    index: VectorIndex,
    manifest_store: ManifestReader,
    query_builder: Callable[[str, Any], Any],
    sparse_loader: Callable[[str], Any],
    candidate_k: int,
    rrf_k: int,
    alpha: float,
    artifact_kinds: frozenset[str] | None = None,
) -> tuple[list[tuple[Document, float]], dict[str, Any]]:
    """Retrieve one ready V3 document through dense and sparse text proxies."""

    if not query.strip():
        raise ValueError("query must be nonempty")
    if not document_id.strip():
        raise ValueError("V3 retrieval requires a selected document")
    if candidate_k < 1:
        raise ValueError("candidate_k must be positive")

    manifest = manifest_store.get_ready_manifest(
        index_name=index_name,
        document_id=document_id,
    )
    if manifest is None:
        raise ValueError("Selected V3 document is not ready for retrieval")

    bm25_params = manifest.get("bm25_params")
    sparse_encoder = sparse_loader(bm25_params) if bm25_params else None
    representation = query_builder(query, sparse_encoder)
    dense_query = list(representation.dense)
    filter_terms: list[dict[str, Any]] = [
            {"document_id": {"$eq": document_id}},
            {
                "retrieval_schema_id": {
                    "$eq": manifest["retrieval_schema_id"]
                }
            },
            {"type": {"$eq": "content"}},
    ]
    if artifact_kinds:
        filter_terms.append(
            {"artifact_kind": {"$in": sorted(artifact_kinds)}}
        )
    retrieval_filter = {"$and": filter_terms}

    dense_response = index.query(
        vector=dense_query,
        top_k=candidate_k,
        filter=retrieval_filter,
        include_metadata=True,
    )
    dense_matches = [_match(match) for match in _matches(dense_response)]

    sparse_matches: list[tuple[str, Document, float]] = []
    if sparse_encoder is not None and representation.sparse is not None:
        sparse_response = index.query(
            vector=[0.0] * len(dense_query),
            sparse_vector=representation.sparse,
            top_k=candidate_k,
            filter=retrieval_filter,
            include_metadata=True,
        )
        sparse_matches = [_match(match) for match in _matches(sparse_response)]

    fused, ranked = _weighted_rrf(
        dense_matches,
        sparse_matches,
        rrf_k=rrf_k,
        alpha=alpha,
    )
    fused = fused[:candidate_k]
    return fused, {
        "query": query,
        "document_id": document_id,
        "retrieval_schema_id": manifest["retrieval_schema_id"],
        "dense_count": len(dense_matches),
        "sparse_count": len(sparse_matches),
        "rrf_count": len(fused),
        "artifact_kinds": sorted(artifact_kinds or ()),
        "rrf": [
            {
                "rank": rank,
                "vector_id": item["vector_id"],
                "rrf_score": item["rrf_score"],
                "dense_rank": item["dense_rank"],
                "dense_score": item["dense_score"],
                "sparse_rank": item["sparse_rank"],
                "sparse_score": item["sparse_score"],
                "artifact_id": item["document"].metadata.get("artifact_id"),
                "record_id": item["document"].metadata.get("record_id"),
                "artifact_kind": item["document"].metadata.get("artifact_kind"),
                "grounding_basis": item["document"].metadata.get("grounding_basis"),
                "page": item["document"].metadata.get("page_label"),
            }
            for rank, item in enumerate(ranked[:candidate_k], start=1)
        ],
    }


_client = None
_index = None


def _get_index():
    global _client, _index
    if _index is None:
        from pinecone import Pinecone

        from agentic_rag.config import settings

        _client = Pinecone(api_key=settings.pinecone_api_key)
        _index = _client.Index(settings.v3_index_name)
    return _index


def retrieve_v3_document(
    *,
    query: str,
    document_id: str,
    artifact_kinds: frozenset[str] | None = None,
) -> tuple[list[tuple[Document, float]], dict[str, Any]]:
    """Application wrapper around the explicit, independently testable reader."""

    from agentic_rag.config import settings
    from agentic_rag.ingestion.manifest_store import PostgresManifestStore
    from agentic_rag.retrieval.sparse import load_bm25_json
    from agentic_rag.retrieval.vectorstore import build_query_representation

    return retrieve_v3_hybrid_with_scores(
        query=query,
        document_id=document_id,
        index_name=settings.v3_index_name,
        index=_get_index(),
        manifest_store=PostgresManifestStore(),
        query_builder=build_query_representation,
        sparse_loader=load_bm25_json,
        candidate_k=settings.hybrid_candidate_k,
        rrf_k=settings.rrf_k,
        alpha=settings.hybrid_alpha,
        artifact_kinds=artifact_kinds,
    )
