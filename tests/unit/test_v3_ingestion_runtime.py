from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from agentic_rag.ingestion.contracts import (
    Artifact,
    ArtifactKind,
    Derivation,
    GroundingBasis,
    RepresentationKind,
    RetrievalSchema,
    SearchRecord,
    SourceLocator,
)
from agentic_rag.ingestion.manifest_store import ManifestWriteResult
from agentic_rag.ingestion.parsers.base import ParseRun
from agentic_rag.ingestion.v3_runtime import (
    V3IngestionRuntime,
    validate_v3_index_separation,
)


def _schema() -> RetrievalSchema:
    return RetrievalSchema(
        parser_name="test-parser",
        parser_revision="test-revision",
        chunking_strategy="artifact_structure_v3",
        text_embedding_model="test-embedding",
        text_embedding_revision="embedding-revision",
        text_embedding_dimension=3,
    )


class _Parser:
    name = "test-parser"
    revision = "test-revision"

    def __init__(self) -> None:
        self.calls = []

    def parse(
        self,
        path,
        *,
        document_id,
        selected_pages,
        schema,
        asset_store,
    ):
        self.calls.append((path, document_id, tuple(selected_pages), schema, asset_store))
        text = "artifact-aware evidence"
        artifact = Artifact.from_text(
            document_id=document_id,
            kind=ArtifactKind.TEXT_BLOCK,
            text=text,
            derivation=Derivation(producer=self.name, revision=self.revision),
            locator=SourceLocator(page_start=1, page_end=1, page_label="1"),
        )
        record = SearchRecord.from_artifact(
            artifact,
            representation_kind=RepresentationKind.NATIVE_TEXT,
            search_text=text,
            grounding_basis=GroundingBasis.DIRECT_TEXT,
            retrieval_schema_id=schema.schema_id,
        )
        return ParseRun(
            parser_name=self.name,
            parser_revision=self.revision,
            document_id=document_id,
            retrieval_schema_id=schema.schema_id,
            selected_pages=tuple(selected_pages),
            artifacts=(artifact,),
            records=(record,),
            elapsed_seconds=0.1,
            peak_memory_mib=1.0,
        )


class _ManifestStore:
    def __init__(self, events) -> None:
        self.events = events

    def replace_parse_run(self, *, index_name, source_name, schema, run):
        self.events.append(("parsed", index_name, source_name, run.document_id))
        return ManifestWriteResult(
            index_name=index_name,
            document_id=run.document_id,
            retrieval_schema_id=schema.schema_id,
            artifact_count=len(run.artifacts),
            record_count=len(run.records),
        )

    def mark_ready(self, **values):
        self.events.append(("ready", values["document_id"], values["vector_count"]))


class _Index:
    def __init__(self, events, *, fail=False) -> None:
        self.events = events
        self.fail = fail

    def upsert(self, *, vectors):
        self.events.append(("vectors", len(vectors)))
        if self.fail:
            raise RuntimeError("vector write failed")

    def list(self, *, prefix):
        self.events.append(("list", prefix))
        return []

    def delete(self, *, ids):
        raise AssertionError(f"unexpected deletion: {ids}")


class _Embedder:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _SparseEncoder:
    def encode_documents(self, text):
        return {"indices": [1], "values": [1.0]}


def _runtime(tmp_path, *, fail_index=False):
    events = []
    parser = _Parser()
    schema = _schema()
    sparse_inputs = []

    def sparse_factory(texts):
        sparse_inputs.append(list(texts))
        return _SparseEncoder()

    runtime = V3IngestionRuntime(
        parser=parser,
        schema=schema,
        asset_store=SimpleNamespace(root=tmp_path / "assets"),
        manifest_store=_ManifestStore(events),
        index=_Index(events, fail=fail_index),
        index_name="agentic-rag-hybrid-v3",
        embedder=_Embedder(),
        page_selector=lambda _: (1, 2),
        sparse_factory=sparse_factory,
        sparse_serializer=lambda _: "bm25-json",
    )
    return runtime, parser, sparse_inputs, events


def test_v3_runtime_connects_parser_manifest_vectors_and_readiness(tmp_path) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"synthetic PDF")
    runtime, parser, sparse_inputs, events = _runtime(tmp_path)

    result = runtime.ingest_pdf(source, source_name="paper.pdf")

    expected_document_id = hashlib.sha256(b"synthetic PDF").hexdigest()
    assert parser.calls[0][1:3] == (expected_document_id, (1, 2))
    assert sparse_inputs == [["artifact-aware evidence"]]
    assert [event[0] for event in events] == ["parsed", "vectors", "list", "ready"]
    assert result.document_id == expected_document_id
    assert result.index_name == "agentic-rag-hybrid-v3"
    assert result.pages_parsed == 2
    assert result.artifacts_stored == 1
    assert result.search_records_indexed == 1
    assert result.vectors_indexed == 1
    assert result.status == "ready"


def test_v3_runtime_does_not_mark_manifest_ready_after_vector_failure(tmp_path) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"synthetic PDF")
    runtime, _, _, events = _runtime(tmp_path, fail_index=True)

    with pytest.raises(RuntimeError, match="vector write failed"):
        runtime.ingest_pdf(source, source_name="paper.pdf")

    assert [event[0] for event in events] == ["parsed", "vectors"]


def test_v3_runtime_rejects_non_pdf_before_parser_work(tmp_path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("not a PDF", encoding="utf-8")
    runtime, parser, sparse_inputs, events = _runtime(tmp_path)

    with pytest.raises(ValueError, match="PDF files only"):
        runtime.ingest_pdf(source, source_name="notes.txt")

    assert parser.calls == []
    assert sparse_inputs == []
    assert events == []


def test_v3_index_must_be_separate_from_v2() -> None:
    with pytest.raises(ValueError, match="separate versioned Pinecone index"):
        validate_v3_index_separation(
            v2_index_name="agentic-rag-hybrid-v2",
            v3_index_name="agentic-rag-hybrid-v2",
        )

    assert validate_v3_index_separation(
        v2_index_name="agentic-rag-hybrid-v2",
        v3_index_name="agentic-rag-hybrid-v3",
    ) == "agentic-rag-hybrid-v3"
