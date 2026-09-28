from agentic_rag.policies import retrieval


def test_repeated_rewrite_rejects_empty_and_equivalent_queries() -> None:
    assert retrieval.is_repeated_rewrite("   ", "What is alignment?", "alignment")
    assert retrieval.is_repeated_rewrite(
        "WHAT is alignment!", "What is alignment?", "alignment in LLMs"
    )
    assert retrieval.is_repeated_rewrite(
        "Alignment in LLMs.", "What is alignment?", "alignment in LLMs"
    )
    assert not retrieval.is_repeated_rewrite(
        "How do authors define alignment?", "What is alignment?", "alignment in LLMs"
    )


def test_rewrite_prompt_uses_failure_reason_without_dropping_original_question() -> None:
    prompt = retrieval.build_rewrite_prompt(
        "Define alignment and explain why it matters",
        "alignment definition",
        "top_score_below_floor",
        "irrelevant",
    )

    assert "Define alignment and explain why it matters" in prompt
    assert "alignment definition" in prompt
    assert "top_score_below_floor" in prompt
    assert "retrieved passages were judged irrelevant" in prompt
    assert "Do not repeat" in prompt


def _configure_thresholds(monkeypatch) -> None:
    monkeypatch.setattr(retrieval.settings, "max_retries", 2)
    monkeypatch.setattr(retrieval.settings, "retrieval_min_top_score", 0.30)
    monkeypatch.setattr(retrieval.settings, "retrieval_top_to_mean_ratio", 1.10)
    monkeypatch.setattr(retrieval.settings, "retrieval_gap_ratio", 0.02)
    monkeypatch.setattr(retrieval.settings, "retrieval_overview_margin", 0.02)


def test_no_evidence_rewrites_query(monkeypatch) -> None:
    _configure_thresholds(monkeypatch)

    result = retrieval.assess_retrieval_confidence(
        metrics={},
        top_doc_type=None,
        overview_top_score=None,
        content_top_score=None,
        retry_count=0,
    )

    assert result == {
        "decision": "rewrite_query",
        "evidence_strength": "weak",
        "reason": "no_usable_retrieval_evidence",
    }


def test_low_score_is_graded_after_retry_budget(monkeypatch) -> None:
    _configure_thresholds(monkeypatch)

    result = retrieval.assess_retrieval_confidence(
        metrics={
            "top_score": 0.10,
            "second_score": 0.05,
            "mean_score": 0.07,
            "top_to_mean_ratio": 1.43,
            "gap_ratio": 0.50,
        },
        top_doc_type="content",
        overview_top_score=None,
        content_top_score=0.10,
        retry_count=2,
    )

    assert result["decision"] == "grade"
    assert result["reason"] == "top_score_below_floor_retries_exhausted"


def test_well_separated_candidates_generate(monkeypatch) -> None:
    _configure_thresholds(monkeypatch)

    result = retrieval.assess_retrieval_confidence(
        metrics={
            "top_score": 0.80,
            "second_score": 0.40,
            "mean_score": 0.50,
            "top_to_mean_ratio": 1.60,
            "gap_ratio": 0.50,
        },
        top_doc_type="content",
        overview_top_score=0.60,
        content_top_score=0.80,
        retry_count=0,
    )

    assert result == {
        "decision": "generate",
        "evidence_strength": "strong",
        "reason": "strong_cross_encoder_score_distribution",
    }


def test_overview_with_clear_margin_generates(monkeypatch) -> None:
    _configure_thresholds(monkeypatch)

    result = retrieval.assess_retrieval_confidence(
        metrics={
            "top_score": 0.62,
            "second_score": 0.60,
            "mean_score": 0.60,
            "top_to_mean_ratio": 1.033,
            "gap_ratio": 0.032,
        },
        top_doc_type="overview",
        overview_top_score=0.62,
        content_top_score=0.60,
        retry_count=0,
    )

    assert result == {
        "decision": "generate",
        "evidence_strength": "strong",
        "reason": "overview_dominant_with_clear_margin",
    }


def test_close_top_candidates_require_grading(monkeypatch) -> None:
    _configure_thresholds(monkeypatch)

    result = retrieval.assess_retrieval_confidence(
        metrics={
            "top_score": 0.80,
            "second_score": 0.79,
            "mean_score": 0.70,
            "top_to_mean_ratio": 1.14,
            "gap_ratio": 0.0125,
        },
        top_doc_type="content",
        overview_top_score=None,
        content_top_score=0.80,
        retry_count=0,
    )

    assert result == {
        "decision": "grade",
        "evidence_strength": "ambiguous",
        "reason": "top_candidates_too_close",
    }
