"""Application registry for ingested documents.

The registry is separate from the vector store. It answers application-level
questions such as which documents have been ingested, provides the same
document_id used by Pinecone retrieval, and - since the hybrid-search change -
stores each document's fitted BM25 parameters so query time can reconstruct
the exact same encoder used at ingestion time.
"""

from datetime import datetime, timezone

import psycopg

from agentic_rag.config import settings
from agentic_rag.indexing import LEGACY_INDEX_NAME


LEGACY_CHUNKING_STRATEGY = "token_window_v1"


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS ingested_documents (
    id SERIAL PRIMARY KEY,
    document_id TEXT,
    filename TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL
);
"""

_MIGRATE_DOCUMENT_ID = """
ALTER TABLE ingested_documents
ADD COLUMN IF NOT EXISTS document_id TEXT;
"""

_MIGRATE_BM25_PARAMS = """
ALTER TABLE ingested_documents
ADD COLUMN IF NOT EXISTS bm25_params TEXT;
"""

_MIGRATE_INDEX_NAME = """
ALTER TABLE ingested_documents
ADD COLUMN IF NOT EXISTS index_name TEXT;
"""

_MIGRATE_CHUNKING_STRATEGY = """
ALTER TABLE ingested_documents
ADD COLUMN IF NOT EXISTS chunking_strategy TEXT;
"""

_BACKFILL_LEGACY_INDEX_NAME = """
UPDATE ingested_documents
SET index_name = %s
WHERE index_name IS NULL
  AND document_id IS NOT NULL;
"""

_BACKFILL_LEGACY_CHUNKING_STRATEGY = """
UPDATE ingested_documents
SET chunking_strategy = %s
WHERE chunking_strategy IS NULL;
"""

_GET_INDEX_CHUNKING_STRATEGIES = """
SELECT DISTINCT chunking_strategy
FROM ingested_documents
WHERE index_name = %s
  AND chunking_strategy IS NOT NULL;
"""

_DEDUPLICATE_DOCUMENT_IDS = """
WITH ranked_documents AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY index_name, document_id
            ORDER BY ingested_at DESC, id DESC
        ) AS row_number
    FROM ingested_documents
    WHERE document_id IS NOT NULL
      AND index_name IS NOT NULL
)
DELETE FROM ingested_documents
WHERE id IN (
    SELECT id
    FROM ranked_documents
    WHERE row_number > 1
);
"""

_DROP_LEGACY_DOCUMENT_ID_UNIQUE_INDEX = """
DROP INDEX IF EXISTS ingested_documents_document_id_unique;
"""

_CREATE_INDEX_DOCUMENT_UNIQUE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS ingested_documents_index_document_unique
ON ingested_documents (index_name, document_id)
WHERE document_id IS NOT NULL
  AND index_name IS NOT NULL;
"""

_GET_INDEX_DOCUMENT_UNIQUE_INDEX = """
SELECT to_regclass('ingested_documents_index_document_unique');
"""


def _connect():
    return psycopg.connect(settings.postgres_url, autocommit=True)


def _ensure_table(conn) -> None:
    conn.execute(_CREATE_TABLE)
    conn.execute(_MIGRATE_DOCUMENT_ID)
    conn.execute(_MIGRATE_BM25_PARAMS)
    conn.execute(_MIGRATE_INDEX_NAME)
    conn.execute(_MIGRATE_CHUNKING_STRATEGY)
    conn.execute(_BACKFILL_LEGACY_CHUNKING_STRATEGY, (LEGACY_CHUNKING_STRATEGY,))

    index_row = conn.execute(_GET_INDEX_DOCUMENT_UNIQUE_INDEX).fetchone()
    if not index_row or index_row[0] is None:
        conn.execute(_BACKFILL_LEGACY_INDEX_NAME, (LEGACY_INDEX_NAME,))
        conn.execute(_DROP_LEGACY_DOCUMENT_ID_UNIQUE_INDEX)
        conn.execute(_DEDUPLICATE_DOCUMENT_IDS)
        conn.execute(_CREATE_INDEX_DOCUMENT_UNIQUE_INDEX)


def _active_index_name(index_name: str | None = None) -> str:
    active = index_name or settings.pinecone_index_name
    if not active or not active.strip():
        raise ValueError("PINECONE_INDEX_NAME cannot be empty")
    return active.strip()


def _active_chunking_strategy(chunking_strategy: str | None = None) -> str:
    active = chunking_strategy or settings.chunking_strategy
    if not active or not active.strip():
        raise ValueError("CHUNKING_STRATEGY cannot be empty")
    return active.strip()


def _assert_index_chunking_strategy(conn, index_name: str, chunking_strategy: str) -> None:
    rows = conn.execute(
        _GET_INDEX_CHUNKING_STRATEGIES,
        (index_name,),
    ).fetchall()
    existing = {row[0] for row in rows if row and row[0]}
    if existing and existing != {chunking_strategy}:
        configured = ", ".join(sorted(existing))
        raise ValueError(
            f"Index '{index_name}' already contains chunking strategy "
            f"'{configured}', but the application is configured for "
            f"'{chunking_strategy}'. Use a separate versioned index."
        )


def ensure_index_chunking_strategy(
    chunking_strategy: str | None = None,
    index_name: str | None = None,
) -> None:
    """Reject an ingestion that would mix retrieval schemas in one index."""
    active_index = _active_index_name(index_name)
    active_strategy = _active_chunking_strategy(chunking_strategy)
    with _connect() as conn:
        _ensure_table(conn)
        _assert_index_chunking_strategy(conn, active_index, active_strategy)


def record_ingestion(
    filename: str,
    chunk_count: int,
    document_id: str | None = None,
    bm25_params: str | None = None,
    index_name: str | None = None,
    chunking_strategy: str | None = None,
) -> None:
    active_index = _active_index_name(index_name)
    active_strategy = _active_chunking_strategy(chunking_strategy)
    with _connect() as conn:
        _ensure_table(conn)
        _assert_index_chunking_strategy(conn, active_index, active_strategy)
        conn.execute(
            """
            INSERT INTO ingested_documents
                (index_name, document_id, filename, chunk_count, ingested_at,
                 bm25_params, chunking_strategy)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (index_name, document_id)
                WHERE document_id IS NOT NULL AND index_name IS NOT NULL
            DO UPDATE SET
                filename = EXCLUDED.filename,
                chunk_count = EXCLUDED.chunk_count,
                ingested_at = EXCLUDED.ingested_at,
                bm25_params = EXCLUDED.bm25_params,
                chunking_strategy = EXCLUDED.chunking_strategy
            """,
            (
                active_index,
                document_id,
                filename,
                chunk_count,
                datetime.now(timezone.utc),
                bm25_params,
                active_strategy,
            ),
        )


def get_bm25_params(document_id: str, index_name: str | None = None) -> str | None:
    """Return BM25 state for one document in one retrieval index."""
    active_index = _active_index_name(index_name)
    with _connect() as conn:
        _ensure_table(conn)
        row = conn.execute(
            """
            SELECT bm25_params
            FROM ingested_documents
            WHERE index_name = %s AND document_id = %s
            ORDER BY ingested_at DESC
            LIMIT 1
            """,
            (active_index, document_id),
        ).fetchone()
    return row[0] if row and row[0] else None


def get_document_metadata(
    document_id: str,
    index_name: str | None = None,
) -> dict | None:
    """Return authoritative routing metadata for one indexed document."""

    active_index = _active_index_name(index_name)
    with _connect() as conn:
        _ensure_table(conn)
        row = conn.execute(
            """
            SELECT filename, chunking_strategy
            FROM ingested_documents
            WHERE index_name = %s AND document_id = %s
            ORDER BY ingested_at DESC
            LIMIT 1
            """,
            (active_index, document_id),
        ).fetchone()

    if not row:
        return None
    return {
        "filename": row[0],
        "chunking_strategy": row[1],
    }


def list_documents(index_name: str | None = None) -> list[dict]:
    active_index = _active_index_name(index_name)
    with _connect() as conn:
        _ensure_table(conn)
        rows = conn.execute(
            """
            SELECT index_name, document_id, filename, chunk_count, ingested_at,
                   chunking_strategy
            FROM ingested_documents
            WHERE index_name = %s
            ORDER BY ingested_at DESC
            """,
            (active_index,),
        ).fetchall()

    return [
        {
            "index_name": r[0],
            "document_id": r[1],
            "filename": r[2],
            "chunk_count": r[3],
            "ingested_at": r[4].isoformat(),
            "chunking_strategy": r[5],
        }
        for r in rows
    ]

def delete_document_record(document_id: str, index_name: str | None = None) -> int:
    """Removes a document's row(s) from the registry. Returns rows deleted.

    Doesn't touch Pinecone - see retrieval.vectorstore.delete_document_vectors
    for that half. Callers needing a full removal should call both.
    """
    active_index = _active_index_name(index_name)
    with _connect() as conn:
        _ensure_table(conn)
        result = conn.execute(
            """
            DELETE FROM ingested_documents
            WHERE index_name = %s AND document_id = %s
            """,
            (active_index, document_id),
        )
        return result.rowcount
