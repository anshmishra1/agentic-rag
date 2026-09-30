"""Compare RRF and cross-encoder ranks for reviewed PDF passage IDs.

Live mode reads PostgreSQL BM25 state and queries Pinecone, then runs the
local cross-encoder. It does not ingest, call an LLM, or write index data.
Output contains IDs and ranks but no query or passage text.
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path


def first_support_rank(order: list[str], support_ids: set[str]) -> int | None:
    return next((rank for rank, vector_id in enumerate(order, 1) if vector_id in support_ids), None)


def summarize(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["judgment"] == "support_found"]
    if not eligible:
        raise ValueError("No reviewed support labels")
    summary = {"reviewed_positive_cases": len(eligible)}
    for stage, field in (("rrf", "rrf_ids"), ("cross_encoder", "reranked_ids")):
        ranks = [first_support_rank(row[field], set(row["support_ids"])) for row in eligible]
        summary[stage] = {
            "support_in_candidates": sum(rank is not None for rank in ranks),
            "hit_at_1": sum(rank is not None and rank <= 1 for rank in ranks),
            "hit_at_3": sum(rank is not None and rank <= 3 for rank in ranks),
            "hit_at_5": sum(rank is not None and rank <= 5 for rank in ranks),
            "mean_reciprocal_rank": sum(1 / rank for rank in ranks if rank is not None) / len(ranks),
        }
    return summary


def collect(pairs: list[dict], reviews: list[dict]) -> list[dict]:
    from agentic_rag.config import settings
    from agentic_rag.ingestion.registry import get_bm25_params
    from agentic_rag.retrieval.reranker import rerank
    from agentic_rag.retrieval.sparse import load_bm25_json
    from agentic_rag.retrieval.vectorstore import (
        _stable_chunk_id,
        build_query_representation,
        retrieve_hybrid_with_scores,
    )

    by_case = {row["case_id"]: row for row in pairs}
    if len(by_case) != len(pairs):
        raise ValueError("Duplicate pair case IDs")
    output = []
    for review in reviews:
        case_id = review["case_id"]
        pair = by_case[case_id]
        document_id = pair["document_id"]
        support_ids = {item["chunk_id"] for item in review["confirmed_support"]}
        if review["judgment"] == "support_found" and not support_ids:
            raise ValueError(f"Missing reviewed support IDs: {case_id}")
        params = get_bm25_params(document_id)
        if not params:
            raise ValueError(f"Missing BM25 state: {case_id}")
        encoder = load_bm25_json(params)
        document_filter = {"$and": [
            {"document_id": {"$eq": document_id}},
            {"type": {"$eq": "content"}},
        ]}
        query = pair["query"]
        with redirect_stdout(io.StringIO()):
            representation = build_query_representation(query, encoder)
            candidates, _ = retrieve_hybrid_with_scores(
                query, encoder, settings.hybrid_candidate_k,
                filter=document_filter, query_representation=representation,
            )
            reranked = rerank(query, candidates, len(candidates))

        def ids(items):
            return [
                _stable_chunk_id(document_id, "content", doc.page_content)
                for doc, _ in items
            ]

        rrf_ids = ids(candidates)
        reranked_ids = ids(reranked)
        if set(rrf_ids) != set(reranked_ids):
            raise ValueError(f"Candidate set changed during rerank: {case_id}")
        output.append({
            "case_id": case_id,
            "split": review["split"],
            "judgment": review["judgment"],
            "document_id": document_id,
            "support_ids": sorted(support_ids),
            "rrf_ids": rrf_ids,
            "reranked_ids": reranked_ids,
            "rrf_support_rank": first_support_rank(rrf_ids, support_ids),
            "reranked_support_rank": first_support_rank(reranked_ids, support_ids),
        })
        print(f"{case_id}: RRF={output[-1]['rrf_support_rank']} CE={output[-1]['reranked_support_rank']}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true", help="read PostgreSQL/Pinecone and run the local model")
    args = parser.parse_args()
    if not args.live:
        parser.error("Live collection requires explicit --live")
    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    rows = collect(pairs, reviews)
    payload = {"cases": rows, "selected_diagnostic_summary": summarize(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["selected_diagnostic_summary"], indent=2))


if __name__ == "__main__":
    main()
