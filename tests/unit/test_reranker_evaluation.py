from scripts.evaluate_reranker import first_support_rank, summarize


def test_support_rank_and_summary_compare_same_candidate_set():
    rows = [
        {
            "judgment": "support_found", "support_ids": ["good"],
            "rrf_ids": ["other", "good"], "reranked_ids": ["good", "other"],
        },
        {
            "judgment": "support_found", "support_ids": ["missing"],
            "rrf_ids": ["other"], "reranked_ids": ["other"],
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
