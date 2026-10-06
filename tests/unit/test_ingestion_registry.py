from datetime import datetime

from agentic_rag.ingestion import registry


def _normalize_sql(statement: str) -> str:
    return " ".join(statement.split())


class _Connection:
    def __init__(
        self,
        existing_index: str | None = None,
        *,
        chunking_strategies: tuple[str, ...] = (),
        document_rows: list[tuple] | None = None,
    ) -> None:
        self.executions: list[tuple[str, tuple | None]] = []
        self.existing_index = existing_index
        self.chunking_strategies = chunking_strategies
        self.document_rows = document_rows or []
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
        statement = self.executions[-1][0]
        if statement.startswith("SELECT DISTINCT chunking_strategy"):
            return [(strategy,) for strategy in self.chunking_strategies]
        if statement.startswith("SELECT index_name, document_id"):
            return self.document_rows
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
        if statement.startswith("UPDATE ingested_documents SET index_name")
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
    strategy_backfill = next(
        params
        for statement, params in connection.executions
        if statement.startswith("UPDATE ingested_documents SET chunking_strategy")
    )
    assert strategy_backfill == ("token_window_v1",)


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
    assert not any(
        statement.startswith("UPDATE ingested_documents SET index_name")
        for statement in statements
    )
    assert any(
        statement.startswith("UPDATE ingested_documents SET chunking_strategy")
        for statement in statements
    )
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
    assert "chunking_strategy = EXCLUDED.chunking_strategy" in statement
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
    assert params[6] == "token_window_v1"


def test_index_rejects_a_different_chunking_strategy(monkeypatch) -> None:
    connection = _Connection(
        existing_index="ingested_documents_index_document_unique",
        chunking_strategies=("token_window_v1",),
    )
    monkeypatch.setattr(registry, "_connect", lambda: connection)

    try:
        registry.ensure_index_chunking_strategy(
            "structure_aware_v2",
            index_name="agentic-rag-hybrid-v2",
        )
    except ValueError as exc:
        assert "separate versioned index" in str(exc)
        assert "token_window_v1" in str(exc)
        assert "structure_aware_v2" in str(exc)
    else:
        raise AssertionError("A retrieval index must not mix chunking strategies")


def test_index_accepts_an_empty_or_matching_chunking_strategy(monkeypatch) -> None:
    empty = _Connection(existing_index="ingested_documents_index_document_unique")
    monkeypatch.setattr(registry, "_connect", lambda: empty)
    registry.ensure_index_chunking_strategy(
        "structure_aware_v2",
        index_name="agentic-rag-hybrid-v2",
    )

    matching = _Connection(
        existing_index="ingested_documents_index_document_unique",
        chunking_strategies=("structure_aware_v2",),
    )
    monkeypatch.setattr(registry, "_connect", lambda: matching)
    registry.ensure_index_chunking_strategy(
        "structure_aware_v2",
        index_name="agentic-rag-hybrid-v2",
    )


def test_list_documents_exposes_chunking_strategy(monkeypatch) -> None:
    ingested_at = datetime.now().astimezone()
    connection = _Connection(
        existing_index="ingested_documents_index_document_unique",
        document_rows=[
            (
                "agentic-rag-hybrid-v2",
                "doc-123",
                "guide.pdf",
                18,
                ingested_at,
                "structure_aware_v2",
            )
        ],
    )
    monkeypatch.setattr(registry, "_connect", lambda: connection)

    assert registry.list_documents(index_name="agentic-rag-hybrid-v2") == [
        {
            "index_name": "agentic-rag-hybrid-v2",
            "document_id": "doc-123",
            "filename": "guide.pdf",
            "chunk_count": 18,
            "ingested_at": ingested_at.isoformat(),
            "chunking_strategy": "structure_aware_v2",
        }
    ]


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
