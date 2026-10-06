"""Exercise the upload HTTP boundary without live models or providers."""

import hashlib
import importlib.util
import sys
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


def _client(monkeypatch, ingest_file):
    _module(
        monkeypatch,
        "agentic_rag.config",
        settings=SimpleNamespace(
            debug=False,
            pinecone_index_name="agentic-rag-hybrid-v2",
            chunking_strategy="structure_aware_v2",
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
        list_documents=lambda: [],
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
    assert writes[1][1]["chunking_strategy"] == "token_window_v1"
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
