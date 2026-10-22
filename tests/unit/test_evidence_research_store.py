from agentic_rag.ingestion.manifest_store import PostgresManifestStore


_ROW = (
    "record-1",
    "artifact-1",
    "native_text",
    "3 Results",
    "direct_text",
    "schema-v3",
    {},
    "text_block",
    {"page_start": 4, "page_label": "4"},
    "page-4",
    {"layout_class": "section-header", "header_level": 1},
    "paper.pdf",
)


class _Result:
    def __init__(self, rows=()) -> None:
        self.rows = list(rows)

    def fetchall(self):
        return list(self.rows)


class _Connection:
    def __init__(self, *, structure_rows=(), related_rows=()) -> None:
        self.structure_rows = tuple(structure_rows)
        self.related_rows = tuple(related_rows)
        self.executions = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, statement, params=None):
        normalized = " ".join(statement.split())
        self.executions.append((normalized, params))
        if normalized.startswith("SELECT record.record_id, record.artifact_id"):
            return _Result(self.structure_rows)
        if normalized.startswith("WITH anchors AS"):
            return _Result(self.related_rows)
        return _Result()


def test_structure_reader_exposes_only_ready_scoped_records() -> None:
    connection = _Connection(structure_rows=(_ROW,))
    records = PostgresManifestStore(
        connector=lambda: connection
    ).get_ready_document_structure(
        index_name="agentic-rag-hybrid-v3",
        document_id="doc-1",
        limit=30,
    )

    assert records[0]["record_id"] == "record-1"
    statement, params = next(
        execution
        for execution in connection.executions
        if execution[0].startswith("SELECT record.record_id, record.artifact_id")
    )
    assert "manifest.status = 'ready'" in statement
    assert "page_summary" in statement
    assert "section-header" in statement
    assert params == ("agentic-rag-hybrid-v3", "doc-1", 30)


def test_related_reader_expands_only_ready_scoped_anchors() -> None:
    connection = _Connection(related_rows=(_ROW,))
    records = PostgresManifestStore(
        connector=lambda: connection
    ).get_ready_related_records(
        index_name="agentic-rag-hybrid-v3",
        document_id="doc-1",
        record_ids=("record-1",),
        limit=12,
    )

    assert records[0]["document_id"] == "doc-1"
    statement, params = next(
        execution
        for execution in connection.executions
        if execution[0].startswith("WITH anchors AS")
    )
    assert statement.count("manifest.status = 'ready'") == 2
    assert "EXISTS ( SELECT 1 FROM anchors" in statement
    assert "artifact.parent_artifact_id = anchors.parent_artifact_id" in statement
    assert params[0:2] == ("agentic-rag-hybrid-v3", "doc-1")
    assert params[-1] == 12

