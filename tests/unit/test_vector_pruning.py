"""Protect re-ingestion against deleting current or unrelated vectors."""

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


def _vectorstore(monkeypatch):
    _module(monkeypatch, "torch", cuda=SimpleNamespace(is_available=lambda: False))
    _module(
        monkeypatch,
        "langchain_huggingface",
        HuggingFaceEmbeddings=lambda **kwargs: SimpleNamespace(
            embed_documents=lambda texts: [[0.1] for _ in texts]
        ),
    )
    _module(monkeypatch, "pinecone", Pinecone=lambda **kwargs: None)
    _module(monkeypatch, "agentic_rag.config", settings=SimpleNamespace(embedding_model="stub"))
    path = Path(__file__).resolve().parents[2] / "src/agentic_rag/retrieval/vectorstore.py"
    spec = importlib.util.spec_from_file_location("vectorstore_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_prune_removes_only_obsolete_ids_in_batches(monkeypatch):
    vectorstore = _vectorstore(monkeypatch)
    document_id = "a" * 64
    current = f"{document_id}-content-current"
    obsolete = [f"{document_id}-content-old-{i}" for i in range(2001)]
    deleted = []
    index = SimpleNamespace(
        list=lambda *, prefix: iter([obsolete[:1100] + [current], obsolete[1100:]]),
        delete=lambda *, ids: deleted.append(ids),
    )
    monkeypatch.setattr(vectorstore, "get_pinecone_index", lambda: index)

    assert vectorstore.prune_obsolete_document_vectors(document_id, {current}) == 2001
    assert [len(batch) for batch in deleted] == [1000, 1000, 1]
    assert set().union(*map(set, deleted)) == set(obsolete)
    assert current not in set().union(*map(set, deleted))


def test_prune_rejects_empty_or_cross_document_keep_set_before_index_access(monkeypatch):
    vectorstore = _vectorstore(monkeypatch)
    monkeypatch.setattr(vectorstore, "get_pinecone_index", lambda: pytest.fail("index accessed"))
    document_id = "a" * 64

    with pytest.raises(ValueError, match="keep_ids"):
        vectorstore.prune_obsolete_document_vectors(document_id, set())
    with pytest.raises(ValueError, match="keep_ids"):
        vectorstore.prune_obsolete_document_vectors(document_id, {"b" * 64 + "-content-x"})


def test_upsert_returns_only_ids_written_after_success(monkeypatch):
    vectorstore = _vectorstore(monkeypatch)
    document_id = "a" * 64
    writes = []
    index = SimpleNamespace(upsert=lambda *, vectors: writes.extend(vectors))
    monkeypatch.setattr(vectorstore, "get_pinecone_index", lambda: index)
    chunks = [
        Document(page_content="current passage", metadata={"document_id": document_id, "type": "content"}),
        Document(page_content="current overview", metadata={"document_id": document_id, "type": "overview"}),
    ]
    encoder = SimpleNamespace(encode_documents=lambda text: {"indices": [1], "values": [1.0]})

    written = vectorstore.upsert_hybrid(chunks, encoder)

    assert written == {
        f"{document_id}-{kind}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
        for kind, text in (("content", "current passage"), ("overview", "current overview"))
    }
    assert {vector["id"] for vector in writes} == written
