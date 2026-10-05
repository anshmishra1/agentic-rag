from datetime import datetime

from agentic_rag.ingestion import registry


def _normalize_sql(statement: str) -> str:
    return " ".join(statement.split())


class _Connection:
    def __init__(self, existing_index: str | None = None) -> None:
        self.executions: list[tuple[str, tuple | None]] = []
        self.existing_index = existing_index
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def execute(self, statement: str, params: tuple | None = None):
        self.executions.append((_normalize_sql(statement), params))
        return self

    def fetchone(self):
        return (self.existing_index,)

    def fetchall(self):
        return []


def test_schema_backfills_legacy_rows_before_composite_unique_index(monkeypatch) -> None:
    connection = _Connection()
    monkeypatch.setattr(registry, "_connect", lambda: connection)

    registry.record_ingestion(
        "guide.pdf",
        12,
        document_id="doc-123",
        bm25_params="serialized-bm25",
        index_name="agentic-rag-hybrid-v2",
    )

    statements = [statement for statement, _ in connection.executions]
    backfill_position = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("UPDATE ingested_documents")
    )
    drop_position = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("DROP INDEX")
    )
    deduplicate_position = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("WITH ranked_documents AS")
    )
    unique_index_position = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("CREATE UNIQUE INDEX")
    )

    assert backfill_position < drop_position < deduplicate_position < unique_index_position
    assert "PARTITION BY index_name, document_id" in statements[deduplicate_position]
    assert "(index_name, document_id)" in statements[unique_index_position]
    assert connection.executions[backfill_position][1] == ("agentic-rag-hybrid",)


def test_existing_composite_index_skips_one_time_data_migration(monkeypatch) -> None:
    connection = _Connection(
        existing_index="ingested_documents_index_document_unique"
    )
    monkeypatch.setattr(registry, "_connect", lambda: connection)

    registry.record_ingestion(
        "guide.pdf",
        12,
        document_id="doc-123",
        index_name="agentic-rag-hybrid-v2",
    )

    statements = [statement for statement, _ in connection.executions]

    assert not any(
        statement.startswith("WITH ranked_documents AS")
        for statement in statements
    )
    assert not any(
        statement.startswith("CREATE UNIQUE INDEX")
        for statement in statements
    )
    assert not any(statement.startswith("UPDATE ingested_documents") for statement in statements)
    assert not any(statement.startswith("DROP INDEX") for statement in statements)


def test_record_ingestion_upserts_document_metadata(monkeypatch) -> None:
    connection = _Connection()
    monkeypatch.setattr(registry, "_connect", lambda: connection)

    registry.record_ingestion(
        "renamed-guide.pdf",
        18,
        document_id="doc-123",
        bm25_params="updated-bm25",
        index_name="agentic-rag-hybrid-v2",
    )

    statement, params = connection.executions[-1]

    assert "ON CONFLICT (index_name, document_id)" in statement
    assert "filename = EXCLUDED.filename" in statement
    assert "chunk_count = EXCLUDED.chunk_count" in statement
    assert "ingested_at = EXCLUDED.ingested_at" in statement
    assert "bm25_params = EXCLUDED.bm25_params" in statement
    assert params is not None
    assert params[:4] == (
        "agentic-rag-hybrid-v2",
        "doc-123",
        "renamed-guide.pdf",
        18,
    )
    assert isinstance(params[4], datetime)
    assert params[4].tzinfo is not None
    assert params[5] == "updated-bm25"


def test_reads_and_deletes_are_scoped_to_active_index(monkeypatch) -> None:
    connection = _Connection(
        existing_index="ingested_documents_index_document_unique"
    )
    monkeypatch.setattr(registry, "_connect", lambda: connection)

    registry.get_bm25_params("doc-123", index_name="agentic-rag-hybrid-v2")
    registry.list_documents(index_name="agentic-rag-hybrid-v2")
    registry.delete_document_record(
        "doc-123",
        index_name="agentic-rag-hybrid-v2",
    )

    scoped = [
        (statement, params)
        for statement, params in connection.executions
        if "WHERE index_name = %s" in statement
    ]
    assert len(scoped) == 3
    assert scoped[0][1] == ("agentic-rag-hybrid-v2", "doc-123")
    assert scoped[1][1] == ("agentic-rag-hybrid-v2",)
    assert scoped[2][1] == ("agentic-rag-hybrid-v2", "doc-123")


def test_null_document_id_remains_compatible_with_legacy_rows(monkeypatch) -> None:
    connection = _Connection()
    monkeypatch.setattr(registry, "_connect", lambda: connection)

    registry.record_ingestion("legacy.pdf", 4, index_name="agentic-rag-hybrid-v2")

    _, params = connection.executions[-1]

    assert params is not None
    assert params[:2] == ("agentic-rag-hybrid-v2", None)
