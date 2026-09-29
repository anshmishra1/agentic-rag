"""Plan, then explicitly apply, cleanup of stale vectors for exact local PDFs.

The default mode only reads Pinecone and PostgreSQL. It writes a JSON plan
without passage text. Apply requires both the reviewed plan and its SHA-256;
it rechecks the index and BM25 registry state before deleting any IDs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import psycopg
from pinecone import Pinecone

from agentic_rag.config import settings
from agentic_rag.ingestion.chunking import chunk_documents
from agentic_rag.ingestion.loaders import load_pdf
from agentic_rag.retrieval.sparse import load_bm25_json


def _indexed_ids(index, document_id: str) -> set[str]:
    return {
        vector_id
        for batch in index.list(prefix=f"{document_id}-")
        for vector_id in batch
    }


def _registry_state(conn, document_id: str) -> tuple[int, str]:
    row = conn.execute(
        "SELECT chunk_count, bm25_params FROM ingested_documents WHERE document_id = %s",
        (document_id,),
    ).fetchone()
    if not row or not row[1]:
        raise ValueError(f"Missing BM25 registry state for {document_id}")
    return int(row[0]), row[1]


def _sparse_distance(expected: dict, actual: dict) -> float:
    left = dict(zip(expected["indices"], expected["values"]))
    right = dict(zip(actual["indices"], actual["values"]))
    return sum(abs(left.get(term, 0.0) - right.get(term, 0.0)) for term in left.keys() | right.keys())


def _active_overview(index, indexed: set[str], encoder) -> str:
    overview_ids = sorted(vector_id for vector_id in indexed if "-overview-" in vector_id)
    if not overview_ids:
        raise ValueError("No indexed overview candidates")
    vectors = index.fetch(ids=overview_ids).vectors
    matching = []
    for vector_id in overview_ids:
        vector = vectors.get(vector_id)
        if vector is None or (vector.metadata or {}).get("type") != "overview":
            continue
        text = str(vector.metadata.get("text", ""))
        if not text or not vector.sparse_values:
            continue
        expected = encoder.encode_documents(text)
        if _sparse_distance(expected, vector.sparse_values) <= 0.0001:
            matching.append(vector_id)
    if len(matching) != 1:
        raise ValueError(f"Expected one overview matching registry BM25 state, got {len(matching)}")
    return matching[0]


def _plan_documents(index, conn, pairs: list[dict], pdf_dir: Path) -> list[dict]:
    documents = sorted({(row["target_document"], row["document_id"]) for row in pairs})
    plan = []
    for filename, document_id in documents:
        if Path(filename).name != filename:
            raise ValueError(f"Unsafe PDF filename: {filename}")
        source = pdf_dir / filename
        if hashlib.sha256(source.read_bytes()).hexdigest() != document_id:
            raise ValueError(f"PDF hash differs from reviewed document ID: {filename}")
        chunks = chunk_documents(load_pdf(source), document_id=document_id, filename=filename)
        chunk_count, bm25_json = _registry_state(conn, document_id)
        if chunk_count != len(chunks):
            raise ValueError(f"Registry chunk count differs from current PDF parsing: {filename}")
        current_content = {
            f"{document_id}-content-{hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()[:16]}"
            for chunk in chunks
        }
        indexed = _indexed_ids(index, document_id)
        if not current_content.issubset(indexed):
            raise ValueError(f"Current content vectors missing from index: {filename}")
        active_overview = _active_overview(index, indexed, load_bm25_json(bm25_json))
        keep = current_content | {active_overview}
        plan.append({
            "filename": filename,
            "document_id": document_id,
            "registry_bm25_sha256": hashlib.sha256(bm25_json.encode("utf-8")).hexdigest(),
            "raw_chunk_count": len(chunks),
            "current_content_count": len(current_content),
            "indexed_count": len(indexed),
            "keep_ids": sorted(keep),
            "delete_ids": sorted(indexed - keep),
        })
    return plan


def _preflight_apply(index, conn, plan: dict, pdf_dir: Path) -> None:
    if plan.get("index_name") != settings.pinecone_index_name:
        raise ValueError("Pinecone index changed since plan creation")
    for row in plan["documents"]:
        filename = row["filename"]
        document_id = row["document_id"]
        if Path(filename).name != filename:
            raise ValueError("Unsafe filename in plan")
        if hashlib.sha256((pdf_dir / filename).read_bytes()).hexdigest() != document_id:
            raise ValueError(f"PDF changed since plan creation: {filename}")
        _, bm25_json = _registry_state(conn, document_id)
        if hashlib.sha256(bm25_json.encode("utf-8")).hexdigest() != row["registry_bm25_sha256"]:
            raise ValueError(f"BM25 state changed since plan creation: {filename}")
        indexed = _indexed_ids(index, document_id)
        keep, delete = set(row["keep_ids"]), set(row["delete_ids"])
        if not keep or keep & delete or indexed != keep | delete:
            raise ValueError(f"Index IDs changed since plan creation: {filename}")
        if any(not vector_id.startswith(f"{document_id}-") for vector_id in indexed):
            raise ValueError(f"Cross-document ID in plan: {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args()

    index = Pinecone(api_key=settings.pinecone_api_key).Index(settings.pinecone_index_name)
    with psycopg.connect(settings.postgres_url) as conn:
        pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
        if args.apply:
            if not args.expected_plan_sha256:
                parser.error("--apply requires --expected-plan-sha256 from the reviewed dry run")
            data = args.plan.read_bytes()
            if hashlib.sha256(data).hexdigest() != args.expected_plan_sha256:
                raise ValueError("Plan digest differs from reviewed dry run")
            plan = json.loads(data)
            if plan.get("schema_version") != 1:
                raise ValueError("Unsupported cleanup plan schema")
            _preflight_apply(index, conn, plan, args.pdf_dir)
            if _plan_documents(index, conn, pairs, args.pdf_dir) != plan["documents"]:
                raise ValueError("Recomputed cleanup plan differs from reviewed dry run")
            for row in plan["documents"]:
                for start in range(0, len(row["delete_ids"]), 1000):
                    index.delete(ids=row["delete_ids"][start:start + 1000])
            print(f"Deleted {sum(len(row['delete_ids']) for row in plan['documents'])} stale IDs")
            return

        plan = {
            "schema_version": 1,
            "index_name": settings.pinecone_index_name,
            "documents": _plan_documents(index, conn, pairs, args.pdf_dir),
        }
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        digest = hashlib.sha256(args.plan.read_bytes()).hexdigest()
        print(json.dumps({
            "plan_sha256": digest,
            "documents": [
                {"filename": row["filename"], "keep": len(row["keep_ids"]), "delete": len(row["delete_ids"])}
                for row in plan["documents"]
            ],
        }))


if __name__ == "__main__":
    main()
