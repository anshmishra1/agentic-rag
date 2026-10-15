from types import SimpleNamespace

import pytest

from agentic_rag.evaluation.v3_retrieval import (
    assess_case,
    build_candidate_rows,
    select_retrieval_cases,
    summarize,
)


def _document(record_id, *, page, kind, basis, score, logit):
    return SimpleNamespace(
        metadata={
            "record_id": record_id,
            "artifact_id": f"artifact-{record_id}",
            "page_label": page,
            "artifact_kind": kind,
            "grounding_basis": basis,
            "_cross_encoder_score": score,
            "_cross_encoder_logit": logit,
        }
    )


def _case(*, pages=None, kinds=None, bases=None):
    return {
        "expected": {
            "source_pages": pages or [],
            "artifact_types": kinds or [],
            "grounding_bases": bases or [],
        }
    }


def test_select_retrieval_cases_excludes_router_only_cases() -> None:
    suite = {
        "cases": [
            {"case_id": "core", "execution_tier": "e2e_core", "expected": {"should_retrieve": True}},
            {"case_id": "extended", "execution_tier": "e2e_extended", "expected": {"should_retrieve": True}},
            {"case_id": "router", "execution_tier": "e2e_core", "expected": {"should_retrieve": False}},
        ]
    }

    assert [case["case_id"] for case in select_retrieval_cases(suite)] == ["core", "extended"]
    assert [
        case["case_id"]
        for case in select_retrieval_cases(suite, execution_tier="e2e_core")
    ] == ["core"]
    with pytest.raises(ValueError, match="Unknown or non-retrieval"):
        select_retrieval_cases(suite, case_ids={"router"})


def test_candidate_rows_preserve_all_ranks_and_scores_without_text() -> None:
    first = _document("r1", page="3", kind="figure", basis="vision_description", score=0.4, logit=-0.4)
    second = _document("r2", page="3", kind="text_block", basis="direct_text", score=0.9, logit=2.2)
    diagnostics = {
        "rrf": [
            {"record_id": "r1", "vector_id": "v1", "rank": 1, "rrf_score": 0.02, "dense_rank": 1, "dense_score": 0.8, "sparse_rank": 2, "sparse_score": 3.0},
            {"record_id": "r2", "vector_id": "v2", "rank": 2, "rrf_score": 0.01, "dense_rank": 2, "dense_score": 0.7, "sparse_rank": 1, "sparse_score": 4.0},
        ]
    }

    rows = build_candidate_rows(
        candidates=[(first, 0.02), (second, 0.01)],
        retrieval_diagnostics=diagnostics,
        selected=[(second, 0.9)],
    )

    assert rows[0]["rrf_rank"] == 1
    assert rows[0]["rerank_rank"] == 2
    assert rows[0]["selected_rank"] is None
    assert rows[1]["dense_score"] == pytest.approx(0.7)
    assert rows[1]["sparse_score"] == pytest.approx(4.0)
    assert rows[1]["rerank_rank"] == 1
    assert rows[1]["selected_rank"] == 1
    assert all("text" not in row for row in rows)


def test_assessment_requires_page_type_and_grounding_on_the_same_record() -> None:
    candidates = [
        {"page": "3", "artifact_kind": "text_block", "grounding_basis": "direct_text", "rrf_rank": 1, "rerank_rank": 2, "selected_rank": 2},
        {"page": "4", "artifact_kind": "figure", "grounding_basis": "vision_description", "rrf_rank": 2, "rerank_rank": 1, "selected_rank": 1},
    ]

    result = assess_case(
        _case(pages=["3"], kinds=["figure"], bases=["vision_description"]),
        candidates,
    )

    assert result["candidate_page_hit"] is True
    assert result["candidate_artifact_type_hit"] is True
    assert result["candidate_grounding_basis_hit"] is True
    assert result["candidate_evidence_hit"] is False
    assert result["best_rerank_rank"] is None


def test_unanswerable_case_is_not_misreported_as_retrieval_failure() -> None:
    result = assess_case(_case(), [])

    assert result["support_labelled"] is False
    assert result["candidate_evidence_hit"] is False
    assert result["rerank_hit_at_5"] is False


def test_summary_excludes_unlabelled_cases_from_quality_denominator() -> None:
    labelled = assess_case(
        _case(pages=["3"], kinds=["figure"], bases=["vision_description"]),
        [{"page": "3", "artifact_kind": "figure", "grounding_basis": "vision_description", "rrf_rank": 2, "rerank_rank": 1, "selected_rank": 1}],
    )
    unlabelled = assess_case(_case(), [])
    rows = [
        {"status": "ok", "assessment": labelled, "timings": {"retrieval": 1.0, "reranking": 2.0, "total": 3.0}},
        {"status": "ok", "assessment": unlabelled, "timings": {"retrieval": 3.0, "reranking": 4.0, "total": 7.0}},
        {"status": "error", "error_type": "ValueError"},
    ]

    result = summarize(rows)

    assert result["reviewed_support_cases"] == 1
    assert result["unanswerable_or_unlabelled_cases"] == 1
    assert result["quality"]["candidate_evidence_hit"] == 1.0
    assert result["quality"]["mean_reciprocal_rerank"] == 1.0
    assert result["latency_seconds"]["total"]["mean"] == 5.0
    assert result["failures_by_type"] == {"ValueError": 1}
