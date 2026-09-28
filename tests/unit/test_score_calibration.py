import pytest

from agentic_rag.evaluation.score_calibration import analyze_labeled_scores, analyze_scored_rows
from agentic_rag.policies.calibrate_retrieval import collect_live_scores, summarize_retrieval_result


def test_small_sample_does_not_recommend_threshold():
    report = analyze_labeled_scores(
        [{"should_match": True, "top_score": 0.8}, {"should_match": False, "top_score": 0.2}]
    )
    assert report["candidate_min_top_score"] is None
    assert report["reason"] == "insufficient_labeled_examples"


def test_overlapping_scores_do_not_recommend_threshold():
    rows = [{"should_match": True, "top_score": score} for score in (0.5, 0.6, 0.7, 0.8, 0.9)]
    rows += [{"should_match": False, "top_score": score} for score in (0.1, 0.2, 0.3, 0.4, 0.6)]
    report = analyze_labeled_scores(rows)
    assert report["candidate_min_top_score"] is None
    assert report["reason"] == "score_distributions_overlap"


def test_separated_scores_offer_provisional_floor():
    rows = [{"should_match": True, "top_score": score} for score in (0.7, 0.75, 0.8, 0.85, 0.9)]
    rows += [{"should_match": False, "top_score": score} for score in (0.1, 0.2, 0.3, 0.4, 0.5)]
    report = analyze_labeled_scores(rows)
    assert report["candidate_min_top_score"] == pytest.approx(0.6)
    assert report["reason"] == "separated_sample_requires_holdout_validation"


def test_invalid_scores_fail_closed():
    with pytest.raises(ValueError, match="finite"):
        analyze_labeled_scores([{"should_match": True, "top_score": float("nan")}])
    with pytest.raises(ValueError, match="boolean"):
        analyze_labeled_scores([{"should_match": 1, "top_score": 0.5}])


def test_live_collection_validates_before_loading_runtime():
    with pytest.raises(ValueError, match="64-character document_id"):
        collect_live_scores([{"query": "Q", "document_id": "bad", "should_match": True}])


def test_retrieval_snapshot_keeps_policy_inputs_without_passage_text():
    from types import SimpleNamespace

    result = {
        "documents": [SimpleNamespace(page_content="private passage", metadata={"type": "content", "page": 3})],
        "retrieval_scores": [0.7],
        "retrieval_top_score": 0.7,
        "retrieval_second_score": 0.0,
        "retrieval_mean_score": 0.7,
        "retrieval_top_to_mean_ratio": 1.0,
        "retrieval_gap_ratio": 0.0,
        "retrieval_overview_top_score": None,
        "retrieval_content_top_score": 0.7,
    }
    snapshot = summarize_retrieval_result(result, "d" * 64, lambda doc, kind, text: f"{doc}-{kind}-{len(text)}")
    assert snapshot["ranked_candidates"] == [
        {"chunk_id": f"{'d' * 64}-content-15", "score": 0.7, "type": "content", "page": 3}
    ]
    assert snapshot["top_to_mean_ratio"] == 1.0
    assert "private passage" not in str(snapshot)


def test_holdout_scores_do_not_set_candidate_floor():
    rows = [{"split": "development", "should_match": True, "top_score": score} for score in (0.7, 0.75, 0.8, 0.85, 0.9)]
    rows += [{"split": "development", "should_match": False, "top_score": score} for score in (0.1, 0.2, 0.3, 0.4, 0.5)]
    rows += [
        {"split": "holdout", "should_match": True, "top_score": 0.4},
        {"split": "holdout", "should_match": False, "top_score": 0.8},
    ]
    report = analyze_scored_rows(rows)
    assert report["development"]["candidate_min_top_score"] == pytest.approx(0.6)
    assert report["holdout_at_candidate_floor"]["false_positive"] == 1
    assert report["holdout_at_candidate_floor"]["false_negative"] == 1
