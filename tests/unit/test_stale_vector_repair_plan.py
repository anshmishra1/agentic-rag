"""The maintenance plan must identify current vectors and reject drift."""

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from langchain_core.documents import Document


def _module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


def _repair(monkeypatch):
    _module(monkeypatch, "agentic_rag.config", settings=SimpleNamespace(pinecone_index_name="test-index"))
    _module(monkeypatch, "agentic_rag.ingestion.chunking", chunk_documents=lambda docs, **kwargs: docs)
    _module(monkeypatch, "agentic_rag.ingestion.loaders", load_pdf=lambda path: [])
    _module(monkeypatch, "agentic_rag.retrieval.sparse", load_bm25_json=lambda value: value)
    path = Path(__file__).resolve().parents[2] / "scripts/repair_stale_vectors.py"
    spec = importlib.util.spec_from_file_location("repair_stale_vectors_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_active_overview_must_uniquely_match_registry_sparse_state(monkeypatch):
    repair = _repair(monkeypatch)
    current = "a" * 64 + "-overview-current"
    old = "a" * 64 + "-overview-old"
    vectors = {
        current: SimpleNamespace(metadata={"type": "overview", "text": "current"}, sparse_values={"indices": [1], "values": [1.0]}),
        old: SimpleNamespace(metadata={"type": "overview", "text": "old"}, sparse_values={"indices": [1], "values": [2.0]}),
    }
    index = SimpleNamespace(fetch=lambda *, ids: SimpleNamespace(vectors=vectors))
    encoder = SimpleNamespace(encode_documents=lambda text: {"indices": [1], "values": [1.0]})

    assert repair._active_overview(index, set(vectors), encoder) == current
    vectors[old].sparse_values = {"indices": [1], "values": [1.0]}
    with pytest.raises(ValueError, match="got 2"):
        repair._active_overview(index, set(vectors), encoder)


def test_plan_keeps_current_content_and_matching_overview(monkeypatch, tmp_path):
    repair = _repair(monkeypatch)
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"exact PDF bytes")
    document_id = hashlib.sha256(source.read_bytes()).hexdigest()
    current = f"{document_id}-content-{hashlib.sha256(b'answer passage').hexdigest()[:16]}"
    overview = f"{document_id}-overview-current"
    old = f"{document_id}-content-old"
    monkeypatch.setattr(repair, "load_pdf", lambda path: [Document(page_content="answer passage")])
    monkeypatch.setattr(repair, "_active_overview", lambda index, indexed, encoder: overview)
    index = SimpleNamespace(list=lambda *, prefix: iter([[current, overview, old]]))
    conn = SimpleNamespace(execute=lambda *args: SimpleNamespace(fetchone=lambda: (1, "bm25-state")))

    plan = repair._plan_documents(index, conn, [{"target_document": source.name, "document_id": document_id}], tmp_path)

    assert len(plan) == 1
    assert set(plan[0]["keep_ids"]) == {current, overview}
    assert plan[0]["delete_ids"] == [old]


def test_apply_preflight_rejects_index_change(monkeypatch, tmp_path):
    repair = _repair(monkeypatch)
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"exact PDF bytes")
    document_id = hashlib.sha256(source.read_bytes()).hexdigest()
    current = f"{document_id}-content-current"
    old = f"{document_id}-content-old"
    unexpected = f"{document_id}-content-new"
    index = SimpleNamespace(list=lambda *, prefix: iter([[current, old, unexpected]]))
    conn = SimpleNamespace(execute=lambda *args: SimpleNamespace(fetchone=lambda: (1, "bm25-state")))
    plan = {"index_name": "test-index", "documents": [{
        "filename": source.name,
        "document_id": document_id,
        "registry_bm25_sha256": hashlib.sha256(b"bm25-state").hexdigest(),
        "keep_ids": [current],
        "delete_ids": [old],
    }]}

    with pytest.raises(ValueError, match="Index IDs changed"):
        repair._preflight_apply(index, conn, plan, tmp_path)
