"""Pure report assembly and metrics for the V3 retrieval-only evaluation."""

from __future__ import annotations

from collections import Counter
from math import ceil
from statistics import fmean
from typing import Any


def select_retrieval_cases(
    suite: dict[str, Any],
    *,
    execution_tier: str | None = None,
    case_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select cases that should exercise retrieval, preserving suite order."""

    selected = []
    for case in suite["cases"]:
        if not case["expected"]["should_retrieve"]:
            continue
        if execution_tier and case["execution_tier"] != execution_tier:
            continue
        if case_ids is not None and case["case_id"] not in case_ids:
            continue
        selected.append(case)
    if case_ids is not None:
        missing = case_ids - {case["case_id"] for case in selected}
        if missing:
            raise ValueError(
                "Unknown or non-retrieval case IDs: " + ", ".join(sorted(missing))
            )
    return selected


def build_candidate_rows(
    *,
    candidates: list[tuple[Any, float]],
    retrieval_diagnostics: dict[str, Any],
    selected: list[tuple[Any, float]],
) -> list[dict[str, Any]]:
    """Join RRF diagnostics to cross-encoder ranks without passage text."""

    rrf_by_record = {
        str(row.get("record_id") or row["vector_id"]): row
        for row in retrieval_diagnostics["rrf"]
    }
    ranked = sorted(
        candidates,
        key=lambda item: float(item[0].metadata.get("_cross_encoder_score", -1.0)),
        reverse=True,
    )
    rerank_rank = {
        str(document.metadata.get("record_id")): rank
        for rank, (document, _) in enumerate(ranked, start=1)
    }
    selected_rank = {
        str(document.metadata.get("record_id")): rank
        for rank, (document, _) in enumerate(selected, start=1)
    }

    rows = []
    for document, _ in candidates:
        metadata = document.metadata
        record_id = str(metadata.get("record_id") or "")
        if not record_id or record_id not in rrf_by_record:
            raise ValueError("V3 candidate is missing matching record diagnostics")
        diagnostic = rrf_by_record[record_id]
        rows.append(
            {
                "record_id": record_id,
                "artifact_id": metadata.get("artifact_id"),
                "artifact_kind": metadata.get("artifact_kind"),
                "grounding_basis": metadata.get("grounding_basis"),
                "page": metadata.get("page_label"),
                "dense_rank": diagnostic.get("dense_rank"),
                "dense_score": diagnostic.get("dense_score"),
                "sparse_rank": diagnostic.get("sparse_rank"),
                "sparse_score": diagnostic.get("sparse_score"),
                "rrf_rank": diagnostic["rank"],
                "rrf_score": diagnostic["rrf_score"],
                "rerank_rank": rerank_rank[record_id],
                "rerank_score": metadata.get("_cross_encoder_score"),
                "rerank_logit": metadata.get("_cross_encoder_logit"),
                "selected_rank": selected_rank.get(record_id),
            }
        )
    return sorted(rows, key=lambda row: row["rrf_rank"])


def assess_case(case: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Score page, artifact, and grounding labels for one retrieval case."""

    expected = case["expected"]
    pages = {str(page) for page in expected["source_pages"]}
    artifact_types = set(expected["artifact_types"])
    grounding_bases = set(expected["grounding_bases"])
    support_labelled = bool(pages or artifact_types or grounding_bases)

    def page_hit(row: dict[str, Any]) -> bool:
        return not pages or str(row.get("page")) in pages

    def type_hit(row: dict[str, Any]) -> bool:
        return not artifact_types or row.get("artifact_kind") in artifact_types

    def basis_hit(row: dict[str, Any]) -> bool:
        return not grounding_bases or row.get("grounding_basis") in grounding_bases

    def evidence_hit(row: dict[str, Any]) -> bool:
        return support_labelled and page_hit(row) and type_hit(row) and basis_hit(row)

    matching = [row for row in candidates if evidence_hit(row)]
    best_rrf = min((row["rrf_rank"] for row in matching), default=None)
    best_rerank = min((row["rerank_rank"] for row in matching), default=None)
    best_selected = min(
        (row["selected_rank"] for row in matching if row["selected_rank"] is not None),
        default=None,
    )
    return {
        "support_labelled": support_labelled,
        "candidate_page_hit": support_labelled and any(page_hit(row) for row in candidates),
        "candidate_artifact_type_hit": support_labelled
        and any(type_hit(row) for row in candidates),
        "candidate_grounding_basis_hit": support_labelled
        and any(basis_hit(row) for row in candidates),
        "candidate_evidence_hit": bool(matching),
        "best_rrf_rank": best_rrf,
        "best_rerank_rank": best_rerank,
        "best_selected_rank": best_selected,
        "rerank_hit_at_1": best_rerank is not None and best_rerank <= 1,
        "rerank_hit_at_3": best_rerank is not None and best_rerank <= 3,
        "rerank_hit_at_5": best_rerank is not None and best_rerank <= 5,
        "production_selection_hit": best_selected is not None,
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    percentile_index = max(0, min(len(ordered) - 1, ceil(0.95 * len(ordered)) - 1))
    return {
        "mean": fmean(ordered),
        "p95": ordered[percentile_index],
        "maximum": ordered[-1],
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate only reviewed support cases into quality metrics."""

    successful = [row for row in results if row.get("status") == "ok"]
    labelled = [row for row in successful if row["assessment"]["support_labelled"]]
    errors = [row for row in results if row.get("status") != "ok"]
    summary: dict[str, Any] = {
        "cases_requested": len(results),
        "cases_completed": len(successful),
        "cases_failed": len(errors),
        "reviewed_support_cases": len(labelled),
        "unanswerable_or_unlabelled_cases": len(successful) - len(labelled),
        "failures_by_type": dict(sorted(Counter(row.get("error_type") for row in errors).items())),
    }
    if labelled:
        summary["quality"] = {
            field: sum(bool(row["assessment"][field]) for row in labelled) / len(labelled)
            for field in (
                "candidate_page_hit",
                "candidate_artifact_type_hit",
                "candidate_grounding_basis_hit",
                "candidate_evidence_hit",
                "rerank_hit_at_1",
                "rerank_hit_at_3",
                "rerank_hit_at_5",
                "production_selection_hit",
            )
        }
        summary["quality"]["mean_reciprocal_rerank"] = sum(
            1 / rank if (rank := row["assessment"]["best_rerank_rank"]) else 0
            for row in labelled
        ) / len(labelled)
    if successful:
        summary["latency_seconds"] = {
            stage: _latency_summary([float(row["timings"][stage]) for row in successful])
            for stage in ("retrieval", "reranking", "total")
        }
    return summary
