"""Exercise the upload HTTP boundary without live models or providers."""

import hashlib
import importlib.util
import sys
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document


class ProviderUnavailableError(RuntimeError):
    pass


def _module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


def _client(
    monkeypatch,
    ingest_file,
    *,
    v3_runtime=None,
    v3_manifest_store=None,
    v3_asset_store=None,
    v2_documents=(),
):
    _module(
        monkeypatch,
        "agentic_rag.config",
        settings=SimpleNamespace(
            debug=False,
            pinecone_index_name="agentic-rag-hybrid-v2",
            chunking_strategy="structure_aware_v2",
            v3_index_name="agentic-rag-hybrid-v3",
            v3_asset_root=Path("/tmp/v3-assets"),
        ),
    )
    _module(monkeypatch, "agentic_rag.graph.builder", build_graph=lambda _: None)
    _module(
        monkeypatch,
        "agentic_rag.ingestion.pipeline",
        _document_id=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
        ingest_file=ingest_file,
    )
    _module(
        monkeypatch,
        "agentic_rag.llm.provider",
        ProviderUnavailableError=ProviderUnavailableError,
    )
    _module(
        monkeypatch,
        "agentic_rag.ingestion.registry",
        list_documents=lambda: list(v2_documents),
        get_document_metadata=lambda _: None,
        delete_document_record=lambda _: None,
    )
    _module(
        monkeypatch,
        "agentic_rag.retrieval.reranker",
        warmup_cross_encoder=lambda: None,
    )
    _module(
        monkeypatch,
        "agentic_rag.retrieval.vectorstore",
        delete_document_vectors=lambda _: None,
        warmup_embeddings=lambda: None,
    )
    _module(
        monkeypatch,
        "agentic_rag.core.logging",
        configure_logging=lambda *args, **kwargs: None,
        get_run_directory=lambda: None,
        get_run_id=lambda: None,
        get_logger=lambda *args, **kwargs: None,
    )

    path = Path(__file__).resolve().parents[2] / "src/agentic_rag/api/main.py"
    spec = importlib.util.spec_from_file_location("ingest_api_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DiagnosticCollector = lambda **kwargs: SimpleNamespace(
        record=lambda *args, **kwargs: None,
        finalize=lambda *args, **kwargs: None,
    )
    if v3_runtime is not None:
        module._get_v3_ingestion_runtime = lambda: v3_runtime
    if v3_manifest_store is not None:
        module._get_v3_manifest_store = lambda: v3_manifest_store
    if v3_asset_store is not None:
        module._get_v3_asset_store = lambda: v3_asset_store
    return TestClient(module.app)


def test_ingest_returns_document_id_and_cleans_upload(monkeypatch):
    uploaded_paths = []

    def ingest_file(path, display_name):
        uploaded_paths.append(Path(path))
        assert Path(path).read_bytes() == b"example PDF bytes"
        assert display_name == "example.pdf"
        return 2

    client = _client(monkeypatch, ingest_file)
    response = client.post(
        "/ingest", files={"files": ("example.pdf", b"example PDF bytes", "application/pdf")}
    )

    assert response.status_code == 200
    assert response.json() == [{
        "filename": "example.pdf",
        "document_id": hashlib.sha256(b"example PDF bytes").hexdigest(),
        "index_name": "agentic-rag-hybrid-v2",
        "chunking_strategy": "structure_aware_v2",
        "chunks_indexed": 2,
        "retrieval_version": "v2",
    }]
    assert len(uploaded_paths) == 1
    assert not uploaded_paths[0].exists()


def test_ingest_reports_provider_failure_and_cleans_upload(monkeypatch):
    uploaded_paths = []

    def ingest_file(path, display_name):
        uploaded_paths.append(Path(path))
        raise ProviderUnavailableError("provider details must stay in server logs")

    client = _client(monkeypatch, ingest_file)
    response = client.post(
        "/ingest", files={"files": ("example.pdf", b"example PDF bytes", "application/pdf")}
    )

    assert response.status_code == 503
    assert "available LLM provider" in response.json()["detail"]
    assert "provider details" not in response.text
    assert len(uploaded_paths) == 1
    assert not uploaded_paths[0].exists()


def test_v3_ingest_is_opt_in_and_cleans_upload(monkeypatch):
    uploaded_paths = []

    class Runtime:
        def ingest_pdf(self, path, *, source_name):
            uploaded_paths.append(Path(path))
            assert Path(path).read_bytes() == b"V3 PDF bytes"
            assert source_name == "paper.pdf"
            return SimpleNamespace(
                filename=source_name,
                document_id="d" * 64,
                index_name="agentic-rag-hybrid-v3",
                retrieval_schema_id="multimodal_v3:1234567890abcdef",
                parser_name="pymupdf_layout",
                parser_revision="runtime-revision",
                pages_parsed=2,
                artifacts_stored=5,
                search_records_indexed=3,
                vectors_indexed=3,
                obsolete_vectors_deleted=0,
                status="ready",
            )

    client = _client(monkeypatch, lambda *_args, **_kwargs: 0, v3_runtime=Runtime())
    response = client.post(
        "/v3/ingest",
        files={"files": ("paper.pdf", b"V3 PDF bytes", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == [{
        "filename": "paper.pdf",
        "document_id": "d" * 64,
        "index_name": "agentic-rag-hybrid-v3",
        "retrieval_schema_id": "multimodal_v3:1234567890abcdef",
        "parser_name": "pymupdf_layout",
        "parser_revision": "runtime-revision",
        "pages_parsed": 2,
        "artifacts_stored": 5,
        "search_records_indexed": 3,
        "vectors_indexed": 3,
        "obsolete_vectors_deleted": 0,
        "status": "ready",
        "retrieval_version": "v3",
    }]
    assert len(uploaded_paths) == 1
    assert not uploaded_paths[0].exists()


def test_v3_ingest_rejects_non_pdf_without_initializing_runtime(monkeypatch):
    client = _client(monkeypatch, lambda *_args, **_kwargs: 0)
    response = client.post(
        "/v3/ingest",
        files={"files": ("notes.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "V3 ingestion currently accepts PDF files only."


def test_v2_ingest_does_not_initialize_v3_runtime(monkeypatch):
    class ForbiddenRuntime:
        def ingest_pdf(self, *_args, **_kwargs):
            raise AssertionError("V3 runtime must not run for /ingest")

    client = _client(monkeypatch, lambda *_args, **_kwargs: 1, v3_runtime=ForbiddenRuntime())
    response = client.post(
        "/ingest",
        files={"files": ("paper.pdf", b"V2 PDF bytes", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()[0]["index_name"] == "agentic-rag-hybrid-v2"


def test_documents_unify_ready_v3_manifests_with_v2_registry(monkeypatch):
    class ManifestStore:
        def list_ready_manifests(self, *, index_name):
            assert index_name == "agentic-rag-hybrid-v3"
            return [{
                "document_id": "v3-doc",
                "source_name": "visual.pdf",
                "record_count": 8,
                "artifact_count": 12,
                "vector_count": 8,
                "index_name": index_name,
                "retrieval_schema_id": "multimodal_v3:schema",
                "parser_name": "pymupdf_layout",
                "parser_revision": "1.28.2",
                "status": "ready",
            }]

    client = _client(
        monkeypatch,
        lambda *_args, **_kwargs: 0,
        v3_manifest_store=ManifestStore(),
        v2_documents=({
            "document_id": "v2-doc",
            "filename": "legacy.pdf",
            "chunk_count": 4,
        },),
    )

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json()[0] == {
        "document_id": "v3-doc",
        "filename": "visual.pdf",
        "chunk_count": 8,
        "record_count": 8,
        "artifact_count": 12,
        "vector_count": 8,
        "index_name": "agentic-rag-hybrid-v3",
        "retrieval_schema_id": "multimodal_v3:schema",
        "parser_name": "pymupdf_layout",
        "parser_revision": "1.28.2",
        "status": "ready",
        "retrieval_version": "v3",
    }
    assert response.json()[1]["retrieval_version"] == "v2"


def test_v3_query_dispatches_to_graph_and_returns_structured_evidence(monkeypatch):
    class ManifestStore:
        def get_ready_manifest(self, **kwargs):
            return {
                "document_id": "doc-1",
                "source_name": "paper.pdf",
                "retrieval_schema_id": "multimodal_v3:schema",
            }

    graph_calls = []
    evidence_document = Document(
        page_content="Figure evidence",
        metadata={
            "document_id": "doc-1",
            "artifact_id": "artifact-1",
            "record_id": "record-1",
            "artifact_kind": "figure",
            "grounding_basis": "vision_description",
            "filename": "paper.pdf",
            "page_start": 2,
            "page_end": 2,
            "page_label": "2",
            "type": "content",
        },
    )

    class Graph:
        def invoke(self, state, *, config):
            graph_calls.append((state, config))
            return {
                "generation": "The figure shows the data flow. [S1]",
                "hallucination_grade": "grounded",
                "answer_status": "answered",
                "citations": ["[S1] paper.pdf, page 2"],
                "documents": [evidence_document],
                "answer_source": "document",
            }

    client = _client(
        monkeypatch,
        lambda *_args, **_kwargs: 0,
        v3_manifest_store=ManifestStore(),
    )
    client.app.state.rag_graph = Graph()

    response = client.post(
        "/query",
        json={
            "question": "Explain the figure",
            "session_id": "session-1",
            "document_id": "doc-1",
            "source_mode": "document",
            "retrieval_version": "v3",
        },
    )

    assert response.status_code == 200
    assert graph_calls[0][0]["retrieval_version"] == "v3"
    assert graph_calls[0][0]["document_name"] == "paper.pdf"
    assert graph_calls[0][1]["configurable"]["thread_id"] == "session-1:v3:doc-1"
    payload = response.json()
    assert payload["retrieval_version"] == "v3"
    assert payload["evidence"][0]["artifact_id"] == "artifact-1"
    assert payload["evidence"][0]["asset_url"].endswith(
        "/artifacts/artifact-1/asset"
    )


def test_v3_asset_route_reads_only_resolved_ready_asset(monkeypatch):
    class ManifestStore:
        def get_ready_artifact(self, **kwargs):
            assert kwargs["document_id"] == "doc-1"
            assert kwargs["artifact_id"] == "artifact-1"
            return {"preview_asset_key": "safe/key"}

    class AssetStore:
        @contextmanager
        def open(self, asset_key):
            assert asset_key == "safe/key"
            yield BytesIO(b"PNG bytes")

    client = _client(
        monkeypatch,
        lambda *_args, **_kwargs: 0,
        v3_manifest_store=ManifestStore(),
        v3_asset_store=AssetStore(),
    )

    response = client.get("/v3/documents/doc-1/artifacts/artifact-1/asset")

    assert response.status_code == 200
    assert response.content == b"PNG bytes"
    assert response.headers["content-type"] == "image/png"


def _pipeline(monkeypatch, invoke, writes):
    def upsert(chunks, encoder):
        writes.append(("vectors", chunks))
        return {"document-content-current", "document-overview-current"}

    _module(
        monkeypatch,
        "agentic_rag.ingestion.chunking",
        chunk_documents=lambda docs, **kwargs: [
            Document(page_content="Indexed fact", metadata=kwargs)
        ],
        pdf_extraction_mode=lambda strategy: "layout",
    )
    _module(
        monkeypatch,
        "agentic_rag.ingestion.loaders",
        load_pdf=lambda path, **kwargs: [Document(page_content="Source fact")],
        load_image=lambda path: [],
        load_audio=lambda path: [],
    )
    _module(
        monkeypatch,
        "agentic_rag.llm.provider",
        provider_chain=SimpleNamespace(invoke=invoke),
    )
    _module(
        monkeypatch,
        "agentic_rag.retrieval.sparse",
        fit_bm25=lambda texts: "encoder",
        dump_bm25_json=lambda encoder: "bm25-json",
    )
    _module(
        monkeypatch,
        "agentic_rag.retrieval.vectorstore",
        upsert_hybrid=upsert,
        warmup_embeddings=lambda: None,
        prune_obsolete_document_vectors=lambda document_id, keep_ids: writes.append(
            ("prune", (document_id, keep_ids))
        ),
    )
    _module(
        monkeypatch,
        "agentic_rag.ingestion.registry",
        ensure_index_chunking_strategy=lambda *args, **kwargs: None,
        record_ingestion=lambda *args, **kwargs: writes.append(("registry", kwargs)),
    )

    path = Path(__file__).resolve().parents[2] / "src/agentic_rag/ingestion/pipeline.py"
    spec = importlib.util.spec_from_file_location("ingest_pipeline_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pipeline_indexes_only_after_overview_succeeds(monkeypatch, tmp_path):
    writes = []
    calls = []

    def invoke(prompt, *, max_tokens):
        calls.append((prompt, max_tokens))
        return SimpleNamespace(content="Document overview")

    pipeline = _pipeline(monkeypatch, invoke, writes)
    source = tmp_path / "example.pdf"
    source.write_bytes(b"synthetic PDF")

    count = pipeline.ingest_file(source, display_name="example.pdf")

    assert count == 1
    assert calls[0][1] == 512
    assert "Source fact" in calls[0][0]
    assert [step for step, _ in writes] == ["vectors", "registry", "prune"]
    assert writes[0][1][-1].metadata["type"] == "overview"
    assert writes[1][1]["document_id"] == hashlib.sha256(b"synthetic PDF").hexdigest()
    assert writes[1][1]["chunking_strategy"] == "structure_aware_v2"
    assert writes[2][1][0] == writes[1][1]["document_id"]
    assert writes[2][1][1] == {"document-content-current", "document-overview-current"}


def test_pipeline_does_not_write_when_overview_provider_fails(monkeypatch, tmp_path):
    writes = []

    def invoke(prompt, *, max_tokens):
        raise ProviderUnavailableError("provider failed")

    pipeline = _pipeline(monkeypatch, invoke, writes)
    source = tmp_path / "example.pdf"
    source.write_bytes(b"synthetic PDF")

    try:
        pipeline.ingest_file(source)
    except ProviderUnavailableError:
        pass
    else:
        raise AssertionError("Provider failure should stop ingestion")

    assert writes == []


def test_pipeline_rejects_mixed_index_before_provider_or_vector_work(monkeypatch, tmp_path):
    writes = []
    provider_calls = []

    def invoke(prompt, *, max_tokens):
        provider_calls.append((prompt, max_tokens))
        return SimpleNamespace(content="Document overview")

    pipeline = _pipeline(monkeypatch, invoke, writes)
    source = tmp_path / "example.pdf"
    source.write_bytes(b"synthetic PDF")

    def reject_strategy(*args, **kwargs):
        raise ValueError("Use a separate versioned index")

    monkeypatch.setattr(
        sys.modules["agentic_rag.ingestion.registry"],
        "ensure_index_chunking_strategy",
        reject_strategy,
    )

    with pytest.raises(ValueError, match="separate versioned index"):
        pipeline.ingest_file(source)

    assert provider_calls == []
    assert writes == []


def test_pipeline_preserves_old_vectors_if_registry_update_fails(monkeypatch, tmp_path):
    writes = []
    pipeline = _pipeline(
        monkeypatch,
        lambda prompt, *, max_tokens: SimpleNamespace(content="Document overview"),
        writes,
    )
    source = tmp_path / "example.pdf"
    source.write_bytes(b"synthetic PDF")

    def fail_registry(*args, **kwargs):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(sys.modules["agentic_rag.ingestion.registry"], "record_ingestion", fail_registry)
    with pytest.raises(RuntimeError, match="registry unavailable"):
        pipeline.ingest_file(source)

    assert [step for step, _ in writes] == ["vectors"]


def test_pipeline_does_not_prune_after_failed_vector_upsert(monkeypatch, tmp_path):
    writes = []
    pipeline = _pipeline(
        monkeypatch,
        lambda prompt, *, max_tokens: SimpleNamespace(content="Document overview"),
        writes,
    )
    source = tmp_path / "example.pdf"
    source.write_bytes(b"synthetic PDF")

    def fail_upsert(chunks, encoder):
        raise RuntimeError("vector write failed")

    monkeypatch.setattr(pipeline, "upsert_hybrid", fail_upsert)
    with pytest.raises(RuntimeError, match="vector write failed"):
        pipeline.ingest_file(source)

    assert writes == []
