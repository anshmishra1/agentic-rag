"""Rebuild local PDF chunks for a saved retrieval score snapshot, offline.

Run in the pinned API environment with model downloads disabled. The output
contains source passage text and belongs in ignored logs/, not in Git.
"""

import argparse
import hashlib
import json
from pathlib import Path

from agentic_rag.ingestion.chunking import chunk_documents
from agentic_rag.ingestion.loaders import load_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.scores.read_text(encoding="utf-8"))
    documents = {(row["target_document"], row["document_id"]) for row in rows}
    indexed_chunks = {}
    source_counts = {}
    for filename, document_id in sorted(documents):
        source = args.pdf_dir / filename
        actual_id = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_id != document_id:
            raise ValueError(f"PDF hash differs from score snapshot: {filename}")
        chunks = chunk_documents(load_pdf(source), document_id=document_id, filename=filename)
        source_counts[filename] = len(chunks)
        for chunk in chunks:
            content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:16]
            chunk_id = f"{document_id}-content-{content_hash}"
            indexed_chunks[chunk_id] = chunk

    results = []
    missing = 0
    matched = 0
    for row in rows:
        candidates = []
        for rank, candidate in enumerate(row["ranked_candidates"], 1):
            if candidate["type"] != "content":
                continue
            chunk = indexed_chunks.get(candidate["chunk_id"])
            if chunk is None:
                missing += 1
            else:
                matched += 1
            candidates.append({
                "rank": rank,
                "chunk_id": candidate["chunk_id"],
                "score": candidate["score"],
                "saved_page": candidate["page"],
                "reconstructed_page": chunk.metadata.get("page_label", chunk.metadata.get("page")) if chunk else None,
                "text": chunk.page_content if chunk else None,
            })
        results.append({
            "case_id": row["case_id"],
            "query": row["query"],
            "target_document": row["target_document"],
            "should_match": row["should_match"],
            "split": row["split"],
            "content_candidates": candidates,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source_chunk_counts": source_counts, "matched": matched, "missing": missing, "cases": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reconstructed {matched} ranked content passages; {missing} unmatched. Output: {args.output}")


if __name__ == "__main__":
    main()
