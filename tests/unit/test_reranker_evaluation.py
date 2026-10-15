from scripts.evaluate_reranker import first_support_rank, summarize


def test_support_rank_and_summary_compare_same_candidate_set():
    rows = [
        {
            "judgment": "support_found", "support_ids": ["good"],
            "rrf_ids": ["other", "good"], "reranked_ids": ["good", "other"],
            "reranked_scores": [0.9, 0.1], "rerank_seconds": 2.0,
        },
        {
            "judgment": "support_found", "support_ids": ["missing"],
            "rrf_ids": ["other"], "reranked_ids": ["other"],
            "reranked_scores": [0.2], "rerank_seconds": 4.0,
        },
        {
            "judgment": "unresolved", "support_ids": [],
            "rrf_ids": ["other"], "reranked_ids": ["other"],
        },
    ]
    assert first_support_rank(["x", "good"], {"good"}) == 2
    assert first_support_rank(["x"], {"good"}) is None
    result = summarize(rows)
    assert result["reviewed_positive_cases"] == 2
    assert result["rrf"]["hit_at_1"] == 0
    assert result["cross_encoder"]["hit_at_1"] == 1
    assert result["rrf"]["mean_reciprocal_rank"] == 0.25
    assert result["cross_encoder"]["mean_reciprocal_rank"] == 0.5
    assert result["reranker_latency_seconds"] == {"mean": 3.0, "maximum": 4.0}
    assert result["reranker_score_distribution"]["minimum"] == 0.1
    assert result["reranker_score_distribution"]["maximum"] == 0.9
