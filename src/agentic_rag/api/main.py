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

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel

from agentic_rag.config import settings
from agentic_rag.graph.builder import build_graph
from agentic_rag.ingestion.pipeline import _document_id, ingest_file
from agentic_rag.ingestion.registry import list_documents

from agentic_rag.retrieval.reranker import warmup_cross_encoder
from agentic_rag.core.diagnostic import DiagnosticCollector
from agentic_rag.core.logging import configure_logging, get_run_directory,  get_run_id, get_logger
from agentic_rag.core.timing import PerformanceTracker, reset_current_tracker, set_current_tracker

logger = logging.getLogger(__name__)

configure_logging(settings.debug, log_dir="logs", file_logging=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with PostgresSaver.from_conn_string(settings.postgres_url) as checkpointer:
        checkpointer.setup()
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


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    verification_exhausted: bool = False


class IngestResult(BaseModel):
    filename: str
    document_id: str
    chunks_indexed: int


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    http_request: Request,
) -> QueryResponse:

    config = {
        "configurable": {
            "thread_id": request.session_id,
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
    )

    result = None
    status = "failed"

    try:
        result = http_request.app.state.rag_graph.invoke(
            {
                "question": request.question,
                "document_id": request.document_id,
            },
            config=config,
        )

        status = "completed"

        diagnostics.record(
            "run_completed",
            grounded=result.get("hallucination_grade") == "grounded",
            verification_exhausted=result.get(
                "verification_exhausted",
                False,
            ),
        )

        return QueryResponse(
            answer=result["generation"],
            grounded=result.get("hallucination_grade") == "grounded",
            verification_exhausted=result.get(
                "verification_exhausted",
                False,
            ),
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
                    chunks_indexed=chunk_count,
                )
            )

        except (ValueError, NotImplementedError) as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return results


@app.get("/documents")
def documents() -> list[dict]:
    """Return metadata for actually ingested documents."""

    return list_documents()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}