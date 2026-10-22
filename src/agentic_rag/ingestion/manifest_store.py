"""Durable PostgreSQL manifest storage for multimodal V3 ingestion.

Binary assets stay in ``AssetStore``. This module persists only identities,
lineage, locators, searchable text, and schema metadata. A document manifest is
replaced inside one PostgreSQL transaction, so a failure during that database
replacement restores the previous manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import psycopg

from agentic_rag.config import settings
from agentic_rag.ingestion.contracts import RetrievalSchema
from agentic_rag.ingestion.parsers.base import ParseRun


_CREATE_RETRIEVAL_SCHEMAS = """
CREATE TABLE IF NOT EXISTS v3_retrieval_schemas (
    schema_id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL UNIQUE,
    schema_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_MANIFESTS = """
CREATE TABLE IF NOT EXISTS v3_document_manifests (
    index_name TEXT NOT NULL,
    document_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    retrieval_schema_id TEXT NOT NULL
        REFERENCES v3_retrieval_schemas(schema_id),
    parser_name TEXT NOT NULL,
    parser_revision TEXT NOT NULL,
    selected_pages JSONB NOT NULL,
    warnings JSONB NOT NULL,
    artifact_count INTEGER NOT NULL CHECK (artifact_count >= 0),
    record_count INTEGER NOT NULL CHECK (record_count >= 0),
    vector_count INTEGER NOT NULL DEFAULT 0 CHECK (vector_count >= 0),
    bm25_params TEXT,
    status TEXT NOT NULL CHECK (status IN ('parsed', 'ready')),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (index_name, document_id)
);
"""

_CREATE_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS v3_artifacts (
    index_name TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    derivation JSONB NOT NULL,
    locator JSONB NOT NULL,
    asset_key TEXT,
    native_text TEXT,
    parent_artifact_id TEXT,
    metadata JSONB NOT NULL,
    CHECK (native_text IS NOT NULL OR asset_key IS NOT NULL),
    PRIMARY KEY (index_name, artifact_id),
    UNIQUE (index_name, artifact_id, document_id),
    FOREIGN KEY (index_name, document_id)
        REFERENCES v3_document_manifests(index_name, document_id)
        ON DELETE CASCADE,
    FOREIGN KEY (index_name, parent_artifact_id)
        REFERENCES v3_artifacts(index_name, artifact_id)
        ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED
);
"""

_CREATE_SEARCH_RECORDS = """
CREATE TABLE IF NOT EXISTS v3_search_records (
    index_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    representation_kind TEXT NOT NULL,
    search_text TEXT NOT NULL,
    grounding_basis TEXT NOT NULL,
    retrieval_schema_id TEXT NOT NULL
        REFERENCES v3_retrieval_schemas(schema_id),
    metadata JSONB NOT NULL,
    PRIMARY KEY (index_name, record_id),
    FOREIGN KEY (index_name, artifact_id, document_id)
        REFERENCES v3_artifacts(index_name, artifact_id, document_id)
        ON DELETE CASCADE
);
"""

_CREATE_RECORD_DOCUMENT_INDEX = """
CREATE INDEX IF NOT EXISTS v3_search_records_index_document
ON v3_search_records (index_name, document_id);
"""

_SELECT_INDEX_SCHEMAS = """
SELECT DISTINCT retrieval_schema_id
FROM v3_document_manifests
WHERE index_name = %s;
"""

_SELECT_SCHEMA_FINGERPRINT = """
SELECT fingerprint
FROM v3_retrieval_schemas
WHERE schema_id = %s;
"""


@dataclass(frozen=True)
class ManifestWriteResult:
    index_name: str
    document_id: str
    retrieval_schema_id: str
    artifact_count: int
    record_count: int
    status: str = "parsed"


class ManifestStore(Protocol):
    def replace_parse_run(
        self,
        *,
        index_name: str,
        source_name: str,
        schema: RetrievalSchema,
        run: ParseRun,
    ) -> ManifestWriteResult: ...

    def mark_ready(
        self,
        *,
        index_name: str,
        document_id: str,
        retrieval_schema_id: str,
        vector_count: int,
        bm25_params: str,
    ) -> None: ...

    def get_ready_manifest(
        self,
        *,
        index_name: str,
        document_id: str,
    ) -> dict[str, Any] | None: ...

    def list_ready_manifests(self, *, index_name: str) -> list[dict[str, Any]]: ...

    def get_ready_artifact(
        self,
        *,
        index_name: str,
        document_id: str,
        artifact_id: str,
    ) -> dict[str, Any] | None: ...

    def get_ready_document_structure(
        self,
        *,
        index_name: str,
        document_id: str,
        limit: int,
    ) -> list[dict[str, Any]]: ...

    def get_ready_related_records(
        self,
        *,
        index_name: str,
        document_id: str,
        record_ids: tuple[str, ...],
        limit: int,
    ) -> list[dict[str, Any]]: ...


def _connect():
    return psycopg.connect(settings.postgres_url)


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _nonempty(name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be nonempty")
    return normalized


def _bounded_limit(value: int, *, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("limit must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")
    return value


def _evidence_record(
    *,
    index_name: str,
    document_id: str,
    row,
) -> dict[str, Any]:
    return {
        "index_name": index_name,
        "document_id": document_id,
        "record_id": row[0],
        "artifact_id": row[1],
        "representation_kind": row[2],
        "search_text": row[3],
        "grounding_basis": row[4],
        "retrieval_schema_id": row[5],
        "record_metadata": row[6],
        "artifact_kind": row[7],
        "locator": row[8],
        "parent_artifact_id": row[9],
        "artifact_metadata": row[10],
        "source_name": row[11],
    }


def _ensure_tables(connection) -> None:
    for statement in (
        _CREATE_RETRIEVAL_SCHEMAS,
        _CREATE_MANIFESTS,
        _CREATE_ARTIFACTS,
        _CREATE_SEARCH_RECORDS,
        _CREATE_RECORD_DOCUMENT_INDEX,
    ):
        connection.execute(statement)


def _validate_run(schema: RetrievalSchema, run: ParseRun) -> None:
    if run.retrieval_schema_id != schema.schema_id:
        raise ValueError("parse run and retrieval schema identities differ")
    if run.parser_name != schema.parser_name:
        raise ValueError("parse run and retrieval schema parser names differ")
    if run.parser_revision != schema.parser_revision:
        raise ValueError("parse run and retrieval schema parser revisions differ")


def _assert_index_schema(connection, index_name: str, schema: RetrievalSchema) -> None:
    existing = {
        row[0]
        for row in connection.execute(_SELECT_INDEX_SCHEMAS, (index_name,)).fetchall()
        if row and row[0]
    }
    if existing and existing != {schema.schema_id}:
        configured = ", ".join(sorted(existing))
        raise ValueError(
            f"V3 index '{index_name}' already contains retrieval schema "
            f"'{configured}', but this run uses '{schema.schema_id}'. "
            "Use a separate versioned index."
        )

    row = connection.execute(
        _SELECT_SCHEMA_FINGERPRINT,
        (schema.schema_id,),
    ).fetchone()
    if row and row[0] != schema.fingerprint:
        raise ValueError("retrieval schema ID collision detected")


class PostgresManifestStore:
    """Transactional V3 manifest repository backed by PostgreSQL."""

    def __init__(self, connector=None) -> None:
        self._connector = connector or _connect

    def replace_parse_run(
        self,
        *,
        index_name: str,
        source_name: str,
        schema: RetrievalSchema,
        run: ParseRun,
    ) -> ManifestWriteResult:
        index_name = _nonempty("index_name", index_name)
        source_name = _nonempty("source_name", source_name)
        _validate_run(schema, run)

        with self._connector() as connection:
            _ensure_tables(connection)
            _assert_index_schema(connection, index_name, schema)
            connection.execute(
                """
                INSERT INTO v3_retrieval_schemas
                    (schema_id, fingerprint, schema_json)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (schema_id) DO NOTHING
                """,
                (schema.schema_id, schema.fingerprint, _json(schema.to_dict())),
            )

            # Deleting the old manifest cascades through artifacts and records.
            # The surrounding transaction restores it if any replacement write
            # fails, so a partial re-ingestion is never exposed as ready.
            connection.execute(
                """
                DELETE FROM v3_document_manifests
                WHERE index_name = %s AND document_id = %s
                """,
                (index_name, run.document_id),
            )
            connection.execute(
                """
                INSERT INTO v3_document_manifests
                    (index_name, document_id, source_name, retrieval_schema_id,
                     parser_name, parser_revision, selected_pages, warnings,
                     artifact_count, record_count, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                        %s, %s, 'parsed')
                """,
                (
                    index_name,
                    run.document_id,
                    source_name,
                    schema.schema_id,
                    run.parser_name,
                    run.parser_revision,
                    _json(list(run.selected_pages)),
                    _json(list(run.warnings)),
                    len(run.artifacts),
                    len(run.records),
                ),
            )

            for artifact in run.artifacts:
                payload = artifact.to_dict()
                connection.execute(
                    """
                    INSERT INTO v3_artifacts
                        (index_name, artifact_id, document_id, kind,
                         content_sha256, derivation, locator, asset_key,
                         native_text, parent_artifact_id, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                            %s, %s, %s, %s::jsonb)
                    """,
                    (
                        index_name,
                        artifact.artifact_id,
                        artifact.document_id,
                        artifact.kind.value,
                        artifact.content_sha256,
                        _json(payload["derivation"]),
                        _json(payload["locator"]),
                        artifact.asset_key,
                        artifact.native_text,
                        artifact.parent_artifact_id,
                        _json(payload["metadata"]),
                    ),
                )

            for record in run.records:
                payload = record.to_dict()
                connection.execute(
                    """
                    INSERT INTO v3_search_records
                        (index_name, record_id, document_id, artifact_id,
                         representation_kind, search_text, grounding_basis,
                         retrieval_schema_id, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        index_name,
                        record.record_id,
                        record.document_id,
                        record.artifact_id,
                        record.representation_kind.value,
                        record.search_text,
                        record.grounding_basis.value,
                        record.retrieval_schema_id,
                        _json(payload["metadata"]),
                    ),
                )

        return ManifestWriteResult(
            index_name=index_name,
            document_id=run.document_id,
            retrieval_schema_id=schema.schema_id,
            artifact_count=len(run.artifacts),
            record_count=len(run.records),
        )

    def mark_ready(
        self,
        *,
        index_name: str,
        document_id: str,
        retrieval_schema_id: str,
        vector_count: int,
        bm25_params: str,
    ) -> None:
        index_name = _nonempty("index_name", index_name)
        document_id = _nonempty("document_id", document_id)
        retrieval_schema_id = _nonempty(
            "retrieval_schema_id", retrieval_schema_id
        )
        if vector_count < 1:
            raise ValueError("vector_count must be positive")
        _nonempty("bm25_params", bm25_params)

        with self._connector() as connection:
            _ensure_tables(connection)
            result = connection.execute(
                """
                UPDATE v3_document_manifests
                SET status = 'ready', vector_count = %s, bm25_params = %s,
                    updated_at = NOW()
                WHERE index_name = %s
                  AND document_id = %s
                  AND retrieval_schema_id = %s
                  AND status = 'parsed'
                """,
                (
                    vector_count,
                    bm25_params,
                    index_name,
                    document_id,
                    retrieval_schema_id,
                ),
            )
            if result.rowcount != 1:
                raise ValueError("matching parsed V3 manifest was not found")

    def delete_document(self, *, index_name: str, document_id: str) -> int:
        index_name = _nonempty("index_name", index_name)
        document_id = _nonempty("document_id", document_id)
        with self._connector() as connection:
            _ensure_tables(connection)
            result = connection.execute(
                """
                DELETE FROM v3_document_manifests
                WHERE index_name = %s AND document_id = %s
                """,
                (index_name, document_id),
            )
            return result.rowcount

    def get_ready_manifest(
        self,
        *,
        index_name: str,
        document_id: str,
    ) -> dict[str, Any] | None:
        index_name = _nonempty("index_name", index_name)
        document_id = _nonempty("document_id", document_id)
        with self._connector() as connection:
            _ensure_tables(connection)
            row = connection.execute(
                """
                SELECT source_name, retrieval_schema_id, parser_name,
                       parser_revision, selected_pages, artifact_count,
                       record_count, vector_count, bm25_params
                FROM v3_document_manifests
                WHERE index_name = %s
                  AND document_id = %s
                  AND status = 'ready'
                """,
                (index_name, document_id),
            ).fetchone()
        if not row:
            return None
        return {
            "index_name": index_name,
            "document_id": document_id,
            "source_name": row[0],
            "retrieval_schema_id": row[1],
            "parser_name": row[2],
            "parser_revision": row[3],
            "selected_pages": row[4],
            "artifact_count": row[5],
            "record_count": row[6],
            "vector_count": row[7],
            "bm25_params": row[8],
            "status": "ready",
        }

    def list_ready_manifests(self, *, index_name: str) -> list[dict[str, Any]]:
        """List only complete V3 documents exposed to retrieval and the UI."""

        index_name = _nonempty("index_name", index_name)
        with self._connector() as connection:
            _ensure_tables(connection)
            rows = connection.execute(
                """
                SELECT document_id, source_name, retrieval_schema_id,
                       parser_name, parser_revision, selected_pages,
                       artifact_count, record_count, vector_count
                FROM v3_document_manifests
                WHERE index_name = %s AND status = 'ready'
                ORDER BY source_name, document_id
                """,
                (index_name,),
            ).fetchall()
        return [
            {
                "index_name": index_name,
                "document_id": row[0],
                "source_name": row[1],
                "retrieval_schema_id": row[2],
                "parser_name": row[3],
                "parser_revision": row[4],
                "selected_pages": row[5],
                "artifact_count": row[6],
                "record_count": row[7],
                "vector_count": row[8],
                "status": "ready",
            }
            for row in rows
        ]

    def get_ready_artifact(
        self,
        *,
        index_name: str,
        document_id: str,
        artifact_id: str,
    ) -> dict[str, Any] | None:
        """Resolve an artifact and its nearest stored preview for a ready document."""

        index_name = _nonempty("index_name", index_name)
        document_id = _nonempty("document_id", document_id)
        artifact_id = _nonempty("artifact_id", artifact_id)
        with self._connector() as connection:
            _ensure_tables(connection)
            row = connection.execute(
                """
                SELECT artifact.kind, artifact.locator, artifact.asset_key,
                       artifact.native_text, artifact.parent_artifact_id,
                       parent.asset_key, manifest.source_name
                FROM v3_artifacts AS artifact
                JOIN v3_document_manifests AS manifest
                  ON manifest.index_name = artifact.index_name
                 AND manifest.document_id = artifact.document_id
                LEFT JOIN v3_artifacts AS parent
                  ON parent.index_name = artifact.index_name
                 AND parent.artifact_id = artifact.parent_artifact_id
                WHERE artifact.index_name = %s
                  AND artifact.document_id = %s
                  AND artifact.artifact_id = %s
                  AND manifest.status = 'ready'
                """,
                (index_name, document_id, artifact_id),
            ).fetchone()
        if not row:
            return None
        return {
            "index_name": index_name,
            "document_id": document_id,
            "artifact_id": artifact_id,
            "kind": row[0],
            "locator": row[1],
            "asset_key": row[2],
            "native_text": row[3],
            "parent_artifact_id": row[4],
            "preview_asset_key": row[2] or row[5],
            "source_name": row[6],
        }

    def get_ready_document_structure(
        self,
        *,
        index_name: str,
        document_id: str,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Return bounded headings and page summaries for one ready document."""

        index_name = _nonempty("index_name", index_name)
        document_id = _nonempty("document_id", document_id)
        limit = _bounded_limit(limit, maximum=100)
        with self._connector() as connection:
            _ensure_tables(connection)
            rows = connection.execute(
                """
                SELECT record.record_id, record.artifact_id,
                       record.representation_kind, record.search_text,
                       record.grounding_basis, record.retrieval_schema_id,
                       record.metadata, artifact.kind, artifact.locator,
                       artifact.parent_artifact_id, artifact.metadata,
                       manifest.source_name
                FROM v3_search_records AS record
                JOIN v3_artifacts AS artifact
                  ON artifact.index_name = record.index_name
                 AND artifact.artifact_id = record.artifact_id
                 AND artifact.document_id = record.document_id
                JOIN v3_document_manifests AS manifest
                  ON manifest.index_name = record.index_name
                 AND manifest.document_id = record.document_id
                WHERE record.index_name = %s
                  AND record.document_id = %s
                  AND manifest.status = 'ready'
                  AND (
                    record.representation_kind = 'page_summary'
                    OR artifact.metadata->>'layout_class'
                       IN ('title', 'section-header')
                  )
                ORDER BY
                  COALESCE(
                    NULLIF(artifact.locator->>'page_start', '')::INTEGER,
                    2147483647
                  ),
                  COALESCE(
                    NULLIF(artifact.metadata->>'header_level', '')::INTEGER,
                    2147483647
                  ),
                  record.record_id
                LIMIT %s
                """,
                (index_name, document_id, limit),
            ).fetchall()
        return [
            _evidence_record(
                index_name=index_name,
                document_id=document_id,
                row=row,
            )
            for row in rows
        ]

    def get_ready_related_records(
        self,
        *,
        index_name: str,
        document_id: str,
        record_ids: tuple[str, ...],
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Expand anchors to bounded same-page or same-parent evidence."""

        index_name = _nonempty("index_name", index_name)
        document_id = _nonempty("document_id", document_id)
        limit = _bounded_limit(limit, maximum=50)
        normalized_ids = tuple(
            dict.fromkeys(_nonempty("record_id", value) for value in record_ids)
        )
        if not normalized_ids:
            raise ValueError("record_ids must contain at least one value")
        if len(normalized_ids) > 5:
            raise ValueError("record_ids cannot contain more than five values")

        with self._connector() as connection:
            _ensure_tables(connection)
            rows = connection.execute(
                """
                WITH anchors AS (
                    SELECT record.record_id, artifact.artifact_id,
                           artifact.parent_artifact_id,
                           artifact.locator->>'page_label' AS page_label
                    FROM v3_search_records AS record
                    JOIN v3_artifacts AS artifact
                      ON artifact.index_name = record.index_name
                     AND artifact.artifact_id = record.artifact_id
                     AND artifact.document_id = record.document_id
                    JOIN v3_document_manifests AS manifest
                      ON manifest.index_name = record.index_name
                     AND manifest.document_id = record.document_id
                    WHERE record.index_name = %s
                      AND record.document_id = %s
                      AND record.record_id = ANY(%s::text[])
                      AND manifest.status = 'ready'
                )
                SELECT record.record_id, record.artifact_id,
                       record.representation_kind, record.search_text,
                       record.grounding_basis, record.retrieval_schema_id,
                       record.metadata, artifact.kind, artifact.locator,
                       artifact.parent_artifact_id, artifact.metadata,
                       manifest.source_name
                FROM v3_search_records AS record
                JOIN v3_artifacts AS artifact
                  ON artifact.index_name = record.index_name
                 AND artifact.artifact_id = record.artifact_id
                 AND artifact.document_id = record.document_id
                JOIN v3_document_manifests AS manifest
                  ON manifest.index_name = record.index_name
                 AND manifest.document_id = record.document_id
                WHERE record.index_name = %s
                  AND record.document_id = %s
                  AND manifest.status = 'ready'
                  AND EXISTS (
                    SELECT 1
                    FROM anchors
                    WHERE record.record_id = anchors.record_id
                       OR (
                         anchors.parent_artifact_id IS NOT NULL
                         AND artifact.parent_artifact_id = anchors.parent_artifact_id
                       )
                       OR (
                         anchors.page_label IS NOT NULL
                         AND artifact.locator->>'page_label' = anchors.page_label
                       )
                  )
                ORDER BY
                  CASE WHEN record.record_id = ANY(%s::text[]) THEN 0 ELSE 1 END,
                  COALESCE(
                    NULLIF(artifact.locator->>'page_start', '')::INTEGER,
                    2147483647
                  ),
                  record.record_id
                LIMIT %s
                """,
                (
                    index_name,
                    document_id,
                    list(normalized_ids),
                    index_name,
                    document_id,
                    list(normalized_ids),
                    limit,
                ),
            ).fetchall()
        return [
            _evidence_record(
                index_name=index_name,
                document_id=document_id,
                row=row,
            )
            for row in rows
        ]
