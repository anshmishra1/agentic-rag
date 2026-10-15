from __future__ import annotations

from dataclasses import replace
from pathlib import Path
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
    sha256_bytes,
)
from agentic_rag.ingestion.manifest_store import PostgresManifestStore
from agentic_rag.ingestion.parsers.base import ParseRun
from agentic_rag.ingestion.v3_pipeline import persist_and_index_parse_run
from agentic_rag.retrieval.v3_writer import build_vector_payloads, write_parse_run


def _schema(**overrides) -> RetrievalSchema:
    values = {
        "parser_name": "pymupdf_layout",
        "parser_revision": "parser-revision",
        "parser_options": {"ocr": "conditional_rapidocr"},
        "chunking_strategy": "artifact_structure_v3",
        "text_embedding_model": "embedding-model",
        "text_embedding_revision": "embedding-revision",
        "text_embedding_dimension": 3,
    }
    values.update(overrides)
    return RetrievalSchema(**values)


def _run(*, record_count: int = 1) -> tuple[RetrievalSchema, ParseRun]:
    schema = _schema()
    document_id = "d" * 64
    derivation = Derivation(
        producer=schema.parser_name,
        revision=schema.parser_revision,
    )
    page = Artifact(
        document_id=document_id,
        kind=ArtifactKind.PAGE_IMAGE,
        content_sha256=sha256_bytes(b"page image"),
        derivation=derivation,
        locator=SourceLocator(page_start=1, page_end=1, page_label="1"),
        asset_key="d-storage/page-image-digest",
    )
    artifacts = [page]
    records = []
    for number in range(record_count):
        text = f"retrievable passage {number}"
        artifact = Artifact.from_text(
            document_id=document_id,
            kind=ArtifactKind.TEXT_BLOCK,
            text=text,
            derivation=derivation,
            locator=SourceLocator(
                page_start=1,
                page_end=1,
                page_label="1",
                bbox=(0.1, 0.1 + number * 0.1, 0.8, 0.18 + number * 0.1),
            ),
            parent_artifact_id=page.artifact_id,
            metadata={"box_number": number},
        )
        artifacts.append(artifact)
        records.append(
            SearchRecord.from_artifact(
                artifact,
                representation_kind=RepresentationKind.NATIVE_TEXT,
                search_text=text,
                grounding_basis=GroundingBasis.DIRECT_TEXT,
                retrieval_schema_id=schema.schema_id,
            )
        )
    return schema, ParseRun(
        parser_name=schema.parser_name,
        parser_revision=schema.parser_revision,
        document_id=document_id,
        retrieval_schema_id=schema.schema_id,
        selected_pages=(1,),
        artifacts=tuple(artifacts),
        records=tuple(records),
        elapsed_seconds=0.1,
        peak_memory_mib=2.0,
    )


class _Result:
    def __init__(self, *, rows=(), row=None, rowcount=1) -> None:
        self._rows = list(rows)
        self._row = row
        self.rowcount = rowcount

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._row


class _Connection:
    def __init__(
        self,
        *,
        index_schemas=(),
        schema_fingerprint=None,
        update_rowcount=1,
        ready_row=None,
        ready_rows=(),
        artifact_row=None,
    ) -> None:
        self.index_schemas = tuple(index_schemas)
        self.schema_fingerprint = schema_fingerprint
        self.update_rowcount = update_rowcount
        self.ready_row = ready_row
        self.ready_rows = tuple(ready_rows)
        self.artifact_row = artifact_row
        self.executions: list[tuple[str, tuple | None]] = []
        self.exit_error = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_error = exc_type

    def execute(self, statement: str, params: tuple | None = None):
        normalized = " ".join(statement.split())
        self.executions.append((normalized, params))
        if normalized.startswith("SELECT DISTINCT retrieval_schema_id"):
            return _Result(rows=[(value,) for value in self.index_schemas])
        if normalized.startswith("SELECT fingerprint"):
            row = (
                (self.schema_fingerprint,)
                if self.schema_fingerprint is not None
                else None
            )
            return _Result(row=row)
        if normalized.startswith("UPDATE v3_document_manifests"):
            return _Result(rowcount=self.update_rowcount)
        if normalized.startswith("SELECT source_name, retrieval_schema_id"):
            return _Result(row=self.ready_row)
        if normalized.startswith("SELECT document_id, source_name"):
            return _Result(rows=self.ready_rows)
        if normalized.startswith("SELECT artifact.kind, artifact.locator"):
            return _Result(row=self.artifact_row)
        return _Result()


def test_parse_run_rejects_missing_parent_relationship() -> None:
    schema, run = _run()
    child = replace(run.artifacts[1], parent_artifact_id="artifact_missing")

    with pytest.raises(ValueError, match="unavailable parent"):
        replace(run, artifacts=(run.artifacts[0], child))


def test_manifest_replace_persists_schema_lineage_and_records() -> None:
    schema, run = _run(record_count=2)
    connection = _Connection()
    store = PostgresManifestStore(connector=lambda: connection)

    result = store.replace_parse_run(
        index_name="agentic-rag-hybrid-v3",
        source_name="paper.pdf",
        schema=schema,
        run=run,
    )

    statements = [statement for statement, _ in connection.executions]
    assert result.artifact_count == 3
    assert result.record_count == 2
    assert result.status == "parsed"
    assert connection.exit_error is None
    delete_position = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("DELETE FROM v3_document_manifests")
    )
    manifest_position = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("INSERT INTO v3_document_manifests")
    )
    artifact_positions = [
        index
        for index, statement in enumerate(statements)
        if statement.startswith("INSERT INTO v3_artifacts")
    ]
    record_positions = [
        index
        for index, statement in enumerate(statements)
        if statement.startswith("INSERT INTO v3_search_records")
    ]
    assert delete_position < manifest_position < min(artifact_positions)
    assert max(artifact_positions) < min(record_positions)

    child_write = connection.executions[artifact_positions[1]][1]
    assert child_write is not None
    assert child_write[9] == run.artifacts[0].artifact_id
    record_write = connection.executions[record_positions[0]][1]
    assert record_write is not None
    assert record_write[3] == run.records[0].artifact_id
    assert record_write[7] == schema.schema_id


def test_manifest_schema_mismatch_fails_before_replacement() -> None:
    schema, run = _run()
    connection = _Connection(index_schemas=("multimodal_v3:different",))
    store = PostgresManifestStore(connector=lambda: connection)

    with pytest.raises(ValueError, match="separate versioned index"):
        store.replace_parse_run(
            index_name="agentic-rag-hybrid-v3",
            source_name="paper.pdf",
            schema=schema,
            run=run,
        )

    assert connection.exit_error is ValueError
    assert not any(
        statement.startswith("DELETE FROM v3_document_manifests")
        for statement, _ in connection.executions
    )


def test_manifest_ddl_enforces_parent_and_record_lineage() -> None:
    source = (
        Path(__file__).parents[2]
        / "src/agentic_rag/ingestion/manifest_store.py"
    ).read_text(encoding="utf-8")

    assert "DEFERRABLE INITIALLY DEFERRED" in source
    assert "FOREIGN KEY (index_name, parent_artifact_id)" in source
    assert "FOREIGN KEY (index_name, artifact_id, document_id)" in source
    assert "CHECK (native_text IS NOT NULL OR asset_key IS NOT NULL)" in source


def test_mark_ready_requires_exact_parsed_manifest() -> None:
    schema, run = _run()
    missing = _Connection(update_rowcount=0)
    store = PostgresManifestStore(connector=lambda: missing)

    with pytest.raises(ValueError, match="parsed V3 manifest"):
        store.mark_ready(
            index_name="agentic-rag-hybrid-v3",
            document_id=run.document_id,
            retrieval_schema_id=schema.schema_id,
            vector_count=1,
            bm25_params="{}",
        )


def test_only_ready_manifest_is_exposed_to_future_retrieval() -> None:
    schema, run = _run()
    parsed = PostgresManifestStore(connector=lambda: _Connection())
    assert (
        parsed.get_ready_manifest(
            index_name="agentic-rag-hybrid-v3",
            document_id=run.document_id,
        )
        is None
    )

    row = (
        "paper.pdf",
        schema.schema_id,
        schema.parser_name,
        schema.parser_revision,
        [1],
        2,
        1,
        1,
        "{}",
    )
    ready = PostgresManifestStore(
        connector=lambda: _Connection(ready_row=row)
    ).get_ready_manifest(
        index_name="agentic-rag-hybrid-v3",
        document_id=run.document_id,
    )

    assert ready == {
        "index_name": "agentic-rag-hybrid-v3",
        "document_id": run.document_id,
        "source_name": "paper.pdf",
        "retrieval_schema_id": schema.schema_id,
        "parser_name": schema.parser_name,
        "parser_revision": schema.parser_revision,
        "selected_pages": [1],
        "artifact_count": 2,
        "record_count": 1,
        "vector_count": 1,
        "bm25_params": "{}",
        "status": "ready",
    }


def test_ready_manifest_listing_is_index_scoped_and_ui_safe() -> None:
    schema, run = _run()
    row = (
        run.document_id,
        "paper.pdf",
        schema.schema_id,
        schema.parser_name,
        schema.parser_revision,
        [1],
        2,
        1,
        1,
    )
    connection = _Connection(ready_rows=(row,))

    documents = PostgresManifestStore(
        connector=lambda: connection
    ).list_ready_manifests(index_name="agentic-rag-hybrid-v3")

    assert documents == [{
        "index_name": "agentic-rag-hybrid-v3",
        "document_id": run.document_id,
        "source_name": "paper.pdf",
        "retrieval_schema_id": schema.schema_id,
        "parser_name": schema.parser_name,
        "parser_revision": schema.parser_revision,
        "selected_pages": [1],
        "artifact_count": 2,
        "record_count": 1,
        "vector_count": 1,
        "status": "ready",
    }]
    select = next(
        statement
        for statement, _ in connection.executions
        if statement.startswith("SELECT document_id, source_name")
    )
    assert "status = 'ready'" in select


def test_ready_artifact_resolves_parent_page_preview_without_exposing_paths() -> None:
    _, run = _run()
    child = run.artifacts[1]
    connection = _Connection(
        artifact_row=(
            "text_block",
            child.locator.to_dict(),
            None,
            child.native_text,
            run.artifacts[0].artifact_id,
            "d-storage/page-image-digest",
            "paper.pdf",
        )
    )

    artifact = PostgresManifestStore(
        connector=lambda: connection
    ).get_ready_artifact(
        index_name="agentic-rag-hybrid-v3",
        document_id=run.document_id,
        artifact_id=child.artifact_id,
    )

    assert artifact["artifact_id"] == child.artifact_id
    assert artifact["preview_asset_key"] == "d-storage/page-image-digest"
    assert "status = 'ready'" in next(
        statement
        for statement, _ in connection.executions
        if statement.startswith("SELECT artifact.kind, artifact.locator")
    )


class _Embedder:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(number + offset) for offset in range(self.dimension)]
            for number, _ in enumerate(texts)
        ]


class _Sparse:
    def encode_documents(self, text: str) -> dict[str, list]:
        return {"indices": [len(text)], "values": [1.0]}


class _Index:
    def __init__(self, *, existing=(), fail_upsert=False) -> None:
        self.existing = list(existing)
        self.fail_upsert = fail_upsert
        self.events: list[tuple[str, object]] = []
        self.payloads: list[dict] = []

    def upsert(self, *, vectors):
        self.events.append(("upsert", len(vectors)))
        if self.fail_upsert:
            raise RuntimeError("synthetic write failure")
        self.payloads.extend(vectors)

    def list(self, *, prefix):
        self.events.append(("list", prefix))
        return iter([self.existing])

    def delete(self, *, ids):
        self.events.append(("delete", list(ids)))


def test_v3_writer_preserves_provenance_and_prunes_only_after_all_upserts() -> None:
    schema, run = _run(record_count=2)
    stale = f"{run.document_id}-v3-record-stale"
    index = _Index(existing=(stale,))

    result = write_parse_run(
        index=index,
        index_name="agentic-rag-hybrid-v3",
        source_name="paper.pdf",
        schema=schema,
        run=run,
        embedder=_Embedder(),
        sparse_encoder=_Sparse(),
        batch_size=1,
    )

    assert result.vector_count == 2
    assert result.obsolete_deleted == 1
    assert [event[0] for event in index.events] == [
        "upsert",
        "upsert",
        "list",
        "delete",
    ]
    first = index.payloads[0]
    assert first["metadata"]["record_id"] == run.records[0].record_id
    assert first["metadata"]["artifact_id"] == run.records[0].artifact_id
    assert first["metadata"]["parent_artifact_id"] == run.artifacts[0].artifact_id
    assert first["metadata"]["grounding_basis"] == "direct_text"
    assert first["metadata"]["bbox_left"] == pytest.approx(0.1)
    assert "asset_key" not in first["metadata"]


def test_v3_writer_rejects_embedding_dimension_before_index_access() -> None:
    schema, run = _run()
    index = _Index()

    with pytest.raises(ValueError, match="dimension"):
        build_vector_payloads(
            index_name="agentic-rag-hybrid-v3",
            source_name="paper.pdf",
            schema=schema,
            run=run,
            dense_vectors=[[1.0, 2.0]],
            sparse_vectors=[{"indices": [1], "values": [1.0]}],
        )

    assert index.events == []


def test_pipeline_marks_ready_only_after_successful_vector_write() -> None:
    schema, run = _run()
    calls: list[str] = []
    store = SimpleNamespace(
        replace_parse_run=lambda **kwargs: (
            calls.append("persist")
            or SimpleNamespace(status="parsed")
        ),
        mark_ready=lambda **kwargs: calls.append("ready"),
    )

    persist_and_index_parse_run(
        manifest_store=store,
        index=_Index(),
        index_name="agentic-rag-hybrid-v3",
        source_name="paper.pdf",
        schema=schema,
        run=run,
        embedder=_Embedder(),
        sparse_encoder=_Sparse(),
        bm25_params="{}",
    )
    assert calls == ["persist", "ready"]

    calls.clear()
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        persist_and_index_parse_run(
            manifest_store=store,
            index=_Index(fail_upsert=True),
            index_name="agentic-rag-hybrid-v3",
            source_name="paper.pdf",
            schema=schema,
            run=run,
            embedder=_Embedder(),
            sparse_encoder=_Sparse(),
            bm25_params="{}",
        )
    assert calls == ["persist"]


def test_v3_writer_does_not_import_v2_vectorstore_or_global_settings() -> None:
    source = (
        Path(__file__).parents[2]
        / "src/agentic_rag/retrieval/v3_writer.py"
    ).read_text(encoding="utf-8")

    assert "agentic_rag.retrieval.vectorstore" not in source
    assert "agentic_rag.config" not in source
