"""Run the V3 retrieval and local reranker without calling an LLM provider.

The selected documents must already have ready V3 manifests. Output contains
IDs, locators, ranks, scores, and timings but no passage or answer text.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_rag.evaluation.v3_retrieval import (
    assess_case,
    build_candidate_rows,
    resolve_retrieval_query,
    select_retrieval_cases,
    summarize,
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_suite(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def collect_case(
    case: dict[str, Any],
    *,
    document_id: str,
    cases_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from agentic_rag.config import settings
    from agentic_rag.retrieval.candidates import deduplicate_candidates
    from agentic_rag.retrieval.reranker import rerank_many
    from agentic_rag.retrieval.v3_reader import retrieve_v3_document

    query, query_source_case_id = resolve_retrieval_query(
        case,
        cases_by_id=cases_by_id,
    )
    started = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        retrieval_started = time.perf_counter()
        candidates, diagnostics = retrieve_v3_document(
            query=query,
            document_id=document_id,
        )
        candidates, duplicates_removed = deduplicate_candidates(candidates)
        retrieval_seconds = time.perf_counter() - retrieval_started
        rerank_started = time.perf_counter()
        selected = rerank_many(
            query,
            {"content": candidates},
            {"content": settings.rerank_top_k_content},
        ).get("content", [])
        rerank_seconds = time.perf_counter() - rerank_started
    candidate_rows = build_candidate_rows(
        candidates=candidates,
        retrieval_diagnostics=diagnostics,
        selected=selected,
    )
    return {
        "case_id": case["case_id"],
        "document_key": case["document_key"],
        "document_id": document_id,
        "execution_tier": case["execution_tier"],
        "split": case["split"],
        "interaction_style": case["interaction_style"],
        "query_resolution": {
            "information_need_source": case["expected"]["information_need_source"],
            "source_case_id": query_source_case_id,
        },
        "status": "ok",
        "retrieval_schema_id": diagnostics["retrieval_schema_id"],
        "counts": {
            "dense": diagnostics["dense_count"],
            "sparse": diagnostics["sparse_count"],
            "rrf": diagnostics["rrf_count"],
            "deduplicated": len(candidates),
            "duplicates_removed": duplicates_removed,
            "selected": len(selected),
        },
        "timings": {
            "retrieval": retrieval_seconds,
            "reranking": rerank_seconds,
            "total": time.perf_counter() - started,
        },
        "assessment": assess_case(case, candidate_rows),
        "candidates": candidate_rows,
    }


def collect(
    suite: dict[str, Any],
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    documents = {document["document_key"]: document for document in suite["documents"]}
    cases_by_id = {case["case_id"]: case for case in suite["cases"]}
    results = []
    for case in cases:
        document_id = documents[case["document_key"]]["sha256"]
        try:
            result = collect_case(
                case,
                document_id=document_id,
                cases_by_id=cases_by_id,
            )
            assessment = result["assessment"]
            print(
                f"{case['case_id']}: candidates={result['counts']['rrf']} "
                f"selected={result['counts']['selected']} "
                f"coarse_hit={assessment['candidate_coarse_evidence_hit']} "
                f"exact_labelled={assessment['exact_support_labelled']} "
                f"rerank_rank={assessment['best_rerank_rank']}"
            )
        except Exception as exc:
            result = {
                "case_id": case["case_id"],
                "document_key": case["document_key"],
                "document_id": document_id,
                "execution_tier": case["execution_tier"],
                "split": case["split"],
                "interaction_style": case["interaction_style"],
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"{case['case_id']}: ERROR {type(exc).__name__}: {exc}")
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("eval_set/v3_end_to_end_questions.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tier", choices=("e2e_core", "e2e_extended"))
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument(
        "--live",
        action="store_true",
        help="read PostgreSQL/Pinecone and run the local reranker",
    )
    args = parser.parse_args()
    if not args.live:
        parser.error("Retrieval collection requires explicit --live")

    suite = load_suite(args.suite)
    cases = select_retrieval_cases(
        suite,
        execution_tier=args.tier,
        case_ids=set(args.case_ids) if args.case_ids else None,
    )
    if not cases:
        parser.error("No retrieval cases matched the requested filters")

    results = collect(suite, cases)
    from agentic_rag.config import settings
    from agentic_rag.model_identity import canonical_model_name

    payload = {
        "schema_version": 1,
        "suite_id": suite["suite_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "retrieval_only_no_llm",
        "configuration": {
            "index_name": settings.v3_index_name,
            "candidate_k": settings.hybrid_candidate_k,
            "rrf_k": settings.rrf_k,
            "hybrid_alpha": settings.hybrid_alpha,
            "rerank_top_k": settings.rerank_top_k_content,
            "reranker_model": canonical_model_name(settings.cross_encoder_model),
            "reranker_revision": settings.cross_encoder_model_revision,
            "reranker_device": settings.cross_encoder_device,
        },
        "summary": summarize(results),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    if payload["summary"]["cases_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
