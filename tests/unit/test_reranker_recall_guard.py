from langchain_core.documents import Document

from agentic_rag.retrieval import reranker


def test_rrf_reserve_keeps_second_ranked_support_within_five_slots():
    items = [
        {"original_rank": rank, "cross_encoder_score": float(16 - ce_rank)}
        for ce_rank, rank in enumerate([7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6], 1)
    ]
    selected = reranker._select_with_rrf_reserve(items, top_k=5, reserve=2)
    assert len(selected) == 5
    assert {item["original_rank"] for item in selected} == {7, 8, 9, 1, 2}
    assert [item["cross_encoder_score"] for item in selected] == sorted(
        (item["cross_encoder_score"] for item in selected), reverse=True
    )
    assert reranker._select_with_rrf_reserve(items, top_k=5, reserve=0) == items[:5]
    assert len(reranker._select_with_rrf_reserve(items, top_k=1, reserve=2)) == 1


def test_runtime_content_reserve_does_not_change_overview_group(monkeypatch):
    class FakeModel:
        def predict(self, pairs, **kwargs):
            return [float(int(text)) for _, text in pairs]

    monkeypatch.setattr(reranker, "_get_cross_encoder", lambda: FakeModel())
    monkeypatch.setattr(reranker.settings, "rerank_content_rrf_reserve", 2)
    candidates = [(Document(page_content=str(rank), metadata={"type": "content"}), 1 / rank)
                  for rank in range(1, 7)]
    overview = [(Document(page_content=str(rank), metadata={"type": "overview"}), 1 / rank)
                for rank in range(1, 4)]
    result = reranker.rerank_many(
        "query", {"content": candidates, "overview": overview},
        {"content": 3, "overview": 2},
    )
    assert [doc.page_content for doc, _ in result["content"]] == ["6", "2", "1"]
    assert [doc.page_content for doc, _ in result["overview"]] == ["3", "2"]
