"""Lazy application composition for the opt-in multimodal V3 ingest path.

The parser, manifest repository, vector index, embedding runtime, and sparse
encoder remain independently testable. External clients are constructed only
when the V3 endpoint is called, so importing the API or using V2 cannot contact
PostgreSQL or Pinecone.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from pinecone import Pinecone

from agentic_rag.config import settings
from agentic_rag.ingestion.asset_store import AssetStore, LocalAssetStore
from agentic_rag.ingestion.contracts import RetrievalSchema
from agentic_rag.ingestion.manifest_store import ManifestStore, PostgresManifestStore
from agentic_rag.ingestion.parser_runtime import (
    build_v3_parser,
    build_v3_retrieval_schema,
)
from agentic_rag.ingestion.parsers.base import ParserAdapter
from agentic_rag.ingestion.v3_pipeline import persist_and_index_parse_run
from agentic_rag.model_identity import canonical_model_name
from agentic_rag.retrieval.sparse import dump_bm25_json, fit_bm25
from agentic_rag.retrieval.v3_writer import TextEmbedder, VectorIndex
from agentic_rag.retrieval.vectorstore import embed_document_texts


@dataclass(frozen=True)
class V3IngestionResult:
    filename: str
    document_id: str
    index_name: str
    retrieval_schema_id: str
    parser_name: str
    parser_revision: str
    pages_parsed: int
    artifacts_stored: int
    search_records_indexed: int
    vectors_indexed: int
    obsolete_vectors_deleted: int
    status: str = "ready"


class _SharedTextEmbedder:
    """Adapt the application's one embedding runtime to the V3 writer."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_document_texts(texts)


def _document_id(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _all_pdf_pages(path: Path) -> tuple[int, ...]:
    import pymupdf

    with pymupdf.open(str(path)) as document:
        if len(document) < 1:
            raise ValueError("PDF contains no pages")
        return tuple(range(1, len(document) + 1))


def validate_v3_index_separation(*, v2_index_name: str, v3_index_name: str) -> str:
    """Reject accidental V3 writes to the active V2 retrieval index."""

    v2 = v2_index_name.strip()
    v3 = v3_index_name.strip()
    if not v2 or not v3:
        raise ValueError("V2 and V3 index names must be nonempty")
    if v2 == v3:
        raise ValueError("V3 requires a separate versioned Pinecone index")
    return v3


class V3IngestionRuntime:
    """Coordinate one real PDF through the selected V3 persistence path."""

    def __init__(
        self,
        *,
        parser: ParserAdapter,
        schema: RetrievalSchema,
        asset_store: AssetStore,
        manifest_store: ManifestStore,
        index: VectorIndex,
        index_name: str,
        embedder: TextEmbedder,
        page_selector: Callable[[Path], tuple[int, ...]] = _all_pdf_pages,
        sparse_factory: Callable[[list[str]], Any] = fit_bm25,
        sparse_serializer: Callable[[Any], str] = dump_bm25_json,
    ) -> None:
        self.parser = parser
        self.schema = schema
        self.asset_store = asset_store
        self.manifest_store = manifest_store
        self.index = index
        self.index_name = index_name.strip()
        self.embedder = embedder
        self.page_selector = page_selector
        self.sparse_factory = sparse_factory
        self.sparse_serializer = sparse_serializer
        if not self.index_name:
            raise ValueError("V3 index name must be nonempty")

    def ingest_pdf(self, path: str | Path, *, source_name: str) -> V3IngestionResult:
        path = Path(path)
        if path.suffix.casefold() != ".pdf":
            raise ValueError("V3 ingestion currently accepts PDF files only")
        if not source_name or not source_name.strip():
            raise ValueError("source_name must be nonempty")

        document_id = _document_id(path)
        selected_pages = self.page_selector(path)
        run = self.parser.parse(
            path,
            document_id=document_id,
            selected_pages=selected_pages,
            schema=self.schema,
            asset_store=self.asset_store,
        )
        texts = [record.search_text for record in run.records]
        if not texts:
            raise ValueError("V3 parser produced no searchable records")
        sparse_encoder = self.sparse_factory(texts)
        bm25_params = self.sparse_serializer(sparse_encoder)

        manifest, vectors = persist_and_index_parse_run(
            manifest_store=self.manifest_store,
            index=self.index,
            index_name=self.index_name,
            source_name=source_name.strip(),
            schema=self.schema,
            run=run,
            embedder=self.embedder,
            sparse_encoder=sparse_encoder,
            bm25_params=bm25_params,
        )
        return V3IngestionResult(
            filename=source_name.strip(),
            document_id=run.document_id,
            index_name=self.index_name,
            retrieval_schema_id=self.schema.schema_id,
            parser_name=run.parser_name,
            parser_revision=run.parser_revision,
            pages_parsed=len(run.selected_pages),
            artifacts_stored=manifest.artifact_count,
            search_records_indexed=manifest.record_count,
            vectors_indexed=vectors.vector_count,
            obsolete_vectors_deleted=vectors.obsolete_deleted,
        )


def build_v3_ingestion_runtime() -> V3IngestionRuntime:
    """Construct external V3 adapters only after the V3 route is requested."""

    index_name = validate_v3_index_separation(
        v2_index_name=settings.pinecone_index_name,
        v3_index_name=settings.v3_index_name,
    )
    parser = build_v3_parser(
        backend=settings.v3_parser_backend,
        render_dpi=settings.v3_parser_render_dpi,
        ocr_dpi=settings.v3_parser_ocr_dpi,
        enable_ocr=settings.v3_parser_ocr_enabled,
    )
    schema = build_v3_retrieval_schema(
        parser,
        embedding_model=canonical_model_name(settings.embedding_model),
        embedding_revision=settings.embedding_model_revision,
        embedding_dimension=settings.embedding_dimension,
    )
    pinecone = Pinecone(api_key=settings.pinecone_api_key)
    return V3IngestionRuntime(
        parser=parser,
        schema=schema,
        asset_store=LocalAssetStore(settings.v3_asset_root),
        manifest_store=PostgresManifestStore(),
        index=pinecone.Index(index_name),
        index_name=index_name,
        embedder=_SharedTextEmbedder(),
    )


_runtime: V3IngestionRuntime | None = None
_runtime_lock = Lock()


def get_v3_ingestion_runtime() -> V3IngestionRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = build_v3_ingestion_runtime()
    return _runtime
