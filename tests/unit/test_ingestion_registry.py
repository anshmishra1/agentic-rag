from datetime import datetime

from agentic_rag.ingestion import registry


def _normalize_sql(statement: str) -> str:
    return " ".join(statement.split())


class _Connection:
    def __init__(self, existing_index: str | None = None) -> None:
        self.executions: list[tuple[str, tuple | None]] = []
        self.existing_index = existing_index

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def execute(self, statement: str, params: tuple | None = None):
        self.executions.append((_normalize_sql(statement), params))
        return self

    def fetchone(self):
        return (self.existing_index,)


def test_schema_deduplicates_before_creating_unique_index(monkeypatch) -> None:
    connection = _Connection()
    monkeypatch.setattr(registry, "_connect", lambda: connection)

    registry.record_ingestion(
        "guide.pdf",
        12,
        document_id="doc-123",
        bm25_params="serialized-bm25",
    )

    statements = [statement for statement, _ in connection.executions]
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

    assert deduplicate_position < unique_index_position
    assert "WHERE document_id IS NOT NULL" in statements[deduplicate_position]
    assert "WHERE document_id IS NOT NULL" in statements[unique_index_position]


def test_existing_unique_index_skips_deduplication(monkeypatch) -> None:
    connection = _Connection(
        existing_index="ingested_documents_document_id_unique"
    )
    monkeypatch.setattr(registry, "_connect", lambda: connection)

    registry.record_ingestion("guide.pdf", 12, document_id="doc-123")

    statements = [statement for statement, _ in connection.executions]

    assert not any(
        statement.startswith("WITH ranked_documents AS")
        for statement in statements
    )
    assert not any(
        statement.startswith("CREATE UNIQUE INDEX")
        for statement in statements
    )


def test_record_ingestion_upserts_document_metadata(monkeypatch) -> None:
    connection = _Connection()
    monkeypatch.setattr(registry, "_connect", lambda: connection)

    registry.record_ingestion(
        "renamed-guide.pdf",
        18,
        document_id="doc-123",
        bm25_params="updated-bm25",
    )

    statement, params = connection.executions[-1]

    assert "ON CONFLICT (document_id) WHERE document_id IS NOT NULL" in statement
    assert "filename = EXCLUDED.filename" in statement
    assert "chunk_count = EXCLUDED.chunk_count" in statement
    assert "ingested_at = EXCLUDED.ingested_at" in statement
    assert "bm25_params = EXCLUDED.bm25_params" in statement
    assert params is not None
    assert params[:3] == ("doc-123", "renamed-guide.pdf", 18)
    assert isinstance(params[3], datetime)
    assert params[3].tzinfo is not None
    assert params[4] == "updated-bm25"


def test_null_document_id_remains_compatible_with_legacy_rows(monkeypatch) -> None:
    connection = _Connection()
    monkeypatch.setattr(registry, "_connect", lambda: connection)

    registry.record_ingestion("legacy.pdf", 4)

    _, params = connection.executions[-1]

    assert params is not None
    assert params[0] is None
