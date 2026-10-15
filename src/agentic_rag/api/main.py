"""FastAPI entrypoint.

Keeps graph logic behind a stable REST interface so the Streamlit UI
(or anything else) never touches LangGraph internals directly.

Postgres checkpointer connection opens once at app startup and closes
once at shutdown via FastAPI lifespan.

Request-scoped observability:
- PerformanceTracker measures execution timing.
- DiagnosticCollector records structured run diagnostics.
- Neither is stored in RAGState or LangGraph checkpoints.
"""

import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from agentic_rag.config import settings
from agentic_rag.api.evidence import build_structured_evidence
from agentic_rag.graph.builder import build_graph
from agentic_rag.ingestion.pipeline import _document_id, ingest_file
from agentic_rag.llm.provider import ProviderUnavailableError
from agentic_rag.ingestion.registry import (
    delete_document_record,
    get_document_metadata,
    list_documents,
)

from agentic_rag.retrieval.reranker import warmup_cross_encoder
from agentic_rag.core.diagnostic import DiagnosticCollector
from agentic_rag.core.logging import configure_logging, get_run_directory,  get_run_id, get_logger
from agentic_rag.core.timing import PerformanceTracker, reset_current_tracker, set_current_tracker
from agentic_rag.retrieval.vectorstore import delete_document_vectors, warmup_embeddings

logger = logging.getLogger(__name__)

configure_logging(settings.debug, log_dir="logs", file_logging=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with PostgresSaver.from_conn_string(settings.postgres_url) as checkpointer:
        checkpointer.setup()
        warmup_embeddings()
        warmup_cross_encoder()
        app.state.rag_graph = build_graph(checkpointer)
        yield


app = FastAPI(
    title="Agentic RAG API",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    document_id: str | None = None
    source_mode: Literal["document", "general", "auto"] = "document"
    retrieval_version: Literal["v2", "v3"] = "v2"


class EvidenceResponse(BaseModel):
    label: str
    document_id: str
    artifact_id: str
    record_id: str
    kind: str
    grounding_basis: str
    source_name: str
    description: str
    locator: dict = Field(default_factory=dict)
    asset_url: str | None = None


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    answer_status: str
    grounding_diagnosis: str | None = None
    verification_exhausted: bool = False
    is_control: bool = False
    citations: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    answer_source: str = "document"
    query_relationship: str = "standalone"
    information_need_source: str = "current_turn"
    response_format: str = "requested"
    retrieval_version: Literal["v2", "v3"] = "v2"
    evidence: list[EvidenceResponse] = Field(default_factory=list)


class IngestResult(BaseModel):
    filename: str
    document_id: str
    index_name: str
    chunking_strategy: str
    chunks_indexed: int
    retrieval_version: Literal["v2"] = "v2"


class V3IngestResult(BaseModel):
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
    status: str
    retrieval_version: Literal["v3"] = "v3"


def _get_v3_ingestion_runtime():
    """Keep V3 parser and external clients lazy and isolated from V2 startup."""

    from agentic_rag.ingestion.v3_runtime import get_v3_ingestion_runtime

    return get_v3_ingestion_runtime()


def _get_v3_manifest_store():
    from agentic_rag.ingestion.manifest_store import PostgresManifestStore

    return PostgresManifestStore()


def _get_v3_asset_store():
    from agentic_rag.ingestion.asset_store import LocalAssetStore

    return LocalAssetStore(settings.v3_asset_root)


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    http_request: Request,
) -> QueryResponse:

    config = {
        "configurable": {
            "thread_id": (
                f"{request.session_id}:{request.retrieval_version}:"
                f"{request.document_id or 'unscoped'}"
            ),
        }
    }

    # ---------------------------------------------------------
    # Fresh observability objects for THIS request only
    # ---------------------------------------------------------

    tracker = PerformanceTracker()
    tracker_token = set_current_tracker(tracker)

    diagnostics = DiagnosticCollector(
        run_id=get_run_id(),
        run_directory=get_run_directory(),
        query=request.question,
        session_id=request.session_id,
)

    diagnostics.record(
        "run_started",
        question=request.question,
        session_id=request.session_id,
        document_id=request.document_id,
        source_mode=request.source_mode,
        retrieval_version=request.retrieval_version,
    )

    result = None
    status = "failed"

    try:
        document_metadata = None
        if request.document_id and request.source_mode != "general":
            if request.retrieval_version == "v3":
                document_metadata = _get_v3_manifest_store().get_ready_manifest(
                    index_name=settings.v3_index_name,
                    document_id=request.document_id,
                )
                if document_metadata is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Selected V3 document is not ready for querying.",
                    )
            else:
                document_metadata = get_document_metadata(request.document_id)
        result = http_request.app.state.rag_graph.invoke(
            {
                "question": request.question,
                "document_id": request.document_id,
                "document_name": (
                    document_metadata.get("filename")
                    or document_metadata.get("source_name")
                    if document_metadata
                    else None
                ),
                "source_mode": request.source_mode,
                "retrieval_version": request.retrieval_version,
            },
            config=config,
        )

        status = "completed"

        is_control = result.get("query_is_control", False)

        diagnostics.record(
            "run_completed",
            grounded=result.get("hallucination_grade") == "grounded",
            answer_status=result.get("answer_status"),
            grounding_diagnosis=result.get("grounding_diagnosis"),
            verification_exhausted=result.get("verification_exhausted", False),
            is_control=is_control,
            answer_source=result.get("answer_source", "document"),
            query_relationship=result.get("query_relationship", "standalone"),
            information_need_source=result.get(
                "information_need_source", "current_turn"
            ),
        )

        contexts = [doc.page_content for doc in result.get("documents", [])]
        citations = result.get("citations", [])
        evidence = build_structured_evidence(
            result.get("documents", []),
            citations,
            retrieval_version=request.retrieval_version,
        )

        return QueryResponse(
            answer=result["generation"],
            grounded=result.get("hallucination_grade") == "grounded",
            answer_status=result.get("answer_status", "verification_uncertain"),
            grounding_diagnosis=result.get("grounding_diagnosis"),
            verification_exhausted=result.get("verification_exhausted", False),
            is_control=is_control,
            citations=citations,
            contexts=contexts,
            answer_source=result.get("answer_source", "document"),
            query_relationship=result.get("query_relationship", "standalone"),
            information_need_source=result.get(
                "information_need_source", "current_turn"
            ),
            response_format=result.get("response_format", "requested"),
            retrieval_version=request.retrieval_version,
            evidence=evidence,
    )

    except Exception as exc:
        diagnostics.record(
            "run_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise

    finally:
        # -----------------------------------------------------
        # Capture THIS request's performance information.
        #
        # summary_data() returns structured data without
        # producing the large console performance block.
        # -----------------------------------------------------

        performance = tracker.summary_data()

        diagnostics.record(
            "performance",
            **performance,
        )

        diagnostics.finalize(
            summary={
                "status": status,
                "question": request.question,
                "session_id": request.session_id,
                "document_id": request.document_id,
                "source_mode": request.source_mode,
                "performance": performance,
            },
        )

        # -----------------------------------------------------
        # Remove this request's tracker from the ContextVar.
        # -----------------------------------------------------

        reset_current_tracker(tracker_token)


@app.post("/ingest", response_model=list[IngestResult])
async def ingest(
    files: list[UploadFile] = File(...),
) -> list[IngestResult]:

    """Accept one or more files and run them through ingestion."""

    results = []

    for upload in files:
        suffix = Path(upload.filename or "").suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as tmp:
            shutil.copyfileobj(upload.file, tmp)
            tmp_path = tmp.name

        try:
            document_id = _document_id(Path(tmp_path))

            chunk_count = ingest_file(
                tmp_path,
                display_name=upload.filename,
            )

            results.append(
                IngestResult(
                    filename=upload.filename or "unknown",
                    document_id=document_id,
                    index_name=settings.pinecone_index_name,
                    chunking_strategy=settings.chunking_strategy,
                    chunks_indexed=chunk_count,
                )
            )

        except ProviderUnavailableError as exc:
            logger.warning("Document ingestion could not generate an overview: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Document ingestion needs an available LLM provider to generate "
                    "its overview. Check provider credits, model access, and connectivity."
                ),
            ) from exc

        except (ValueError, NotImplementedError) as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return results


@app.post("/v3/ingest", response_model=list[V3IngestResult])
async def ingest_v3(
    files: list[UploadFile] = File(...),
) -> list[V3IngestResult]:
    """Run PDFs through the opt-in artifact-aware V3 ingestion path."""

    results: list[V3IngestResult] = []
    runtime = None
    uploads = [
        (upload, Path(upload.filename or "unknown").name)
        for upload in files
    ]
    if any(Path(filename).suffix.casefold() != ".pdf" for _, filename in uploads):
        raise HTTPException(
            status_code=400,
            detail="V3 ingestion currently accepts PDF files only.",
        )

    for upload, filename in uploads:
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload.file, tmp)
            tmp_path = Path(tmp.name)

        try:
            if runtime is None:
                runtime = _get_v3_ingestion_runtime()
            result = await run_in_threadpool(
                runtime.ingest_pdf,
                tmp_path,
                source_name=filename,
            )
            results.append(
                V3IngestResult(
                    filename=result.filename,
                    document_id=result.document_id,
                    index_name=result.index_name,
                    retrieval_schema_id=result.retrieval_schema_id,
                    parser_name=result.parser_name,
                    parser_revision=result.parser_revision,
                    pages_parsed=result.pages_parsed,
                    artifacts_stored=result.artifacts_stored,
                    search_records_indexed=result.search_records_indexed,
                    vectors_indexed=result.vectors_indexed,
                    obsolete_vectors_deleted=result.obsolete_vectors_deleted,
                    status=result.status,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("V3 ingestion runtime is unavailable")
            raise HTTPException(
                status_code=503,
                detail="V3 ingestion runtime is unavailable.",
            ) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

    return results


@app.get("/documents")
def documents() -> list[dict]:
    """Return V2 registry rows and ready V3 manifests through one UI contract."""

    v2_documents = [
        {**document, "retrieval_version": "v2"}
        for document in list_documents()
    ]
    v3_documents = [
        {
            "document_id": manifest["document_id"],
            "filename": manifest["source_name"],
            "chunk_count": manifest["record_count"],
            "record_count": manifest["record_count"],
            "artifact_count": manifest["artifact_count"],
            "vector_count": manifest["vector_count"],
            "index_name": manifest["index_name"],
            "retrieval_schema_id": manifest["retrieval_schema_id"],
            "parser_name": manifest["parser_name"],
            "parser_revision": manifest["parser_revision"],
            "status": manifest["status"],
            "retrieval_version": "v3",
        }
        for manifest in _get_v3_manifest_store().list_ready_manifests(
            index_name=settings.v3_index_name
        )
    ]
    return [*v3_documents, *v2_documents]


@app.get("/v3/documents/{document_id}/artifacts/{artifact_id}/asset")
def v3_artifact_asset(document_id: str, artifact_id: str) -> Response:
    """Return a ready artifact's stored image without exposing its asset key."""

    artifact = _get_v3_manifest_store().get_ready_artifact(
        index_name=settings.v3_index_name,
        document_id=document_id,
        artifact_id=artifact_id,
    )
    if artifact is None or not artifact.get("preview_asset_key"):
        raise HTTPException(status_code=404, detail="Artifact preview not found.")
    try:
        with _get_v3_asset_store().open(artifact["preview_asset_key"]) as source:
            content = source.read()
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="Artifact preview not found.")
    return Response(
        content=content,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )

@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict:
    """Removes a document from both Pinecone and the registry - use this
    before re-ingesting a document whose chunking/embedding pipeline changed,
    since chunk IDs are content-hash-derived and won't overwrite in place
    once chunk boundaries differ."""
    vectors_deleted = delete_document_vectors(document_id)
    rows_deleted = delete_document_record(document_id)

    if vectors_deleted == 0 and rows_deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found.")

    return {
        "document_id": document_id,
        "index_name": settings.pinecone_index_name,
        "vectors_deleted": vectors_deleted,
        "registry_rows_deleted": rows_deleted,
    }

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
