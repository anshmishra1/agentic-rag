from dataclasses import dataclass

from agentic_rag.policies import generation


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict | None = None


def test_short_refusal_is_detected() -> None:
    assert generation.is_refusal_answer(
        "I don't know based on the provided context."
    )


def test_substantive_answer_with_hedge_is_not_treated_as_refusal() -> None:
    answer = "I'm not sure, but " + ("this is a factual claim. " * 20)
    assert not generation.is_refusal_answer(answer)


def test_generation_limits_documents_history_and_context(
    monkeypatch,
) -> None:
    monkeypatch.setattr(generation.settings, "max_generation_context_documents", 2)
    monkeypatch.setattr(generation.settings, "max_generation_context_chars", 12)
    monkeypatch.setattr(generation.settings, "max_history_messages_for_generation", 2)

    documents = [
        FakeDocument("first"),
        FakeDocument("second"),
        FakeDocument("third"),
    ]
    history = ["one", "two", "three"]

    context, limited_history = generation.apply_generation_limits(
        documents,
        history,
    )

    assert context == "first\n\nsecon\n[...truncated]"
    assert limited_history == ["two", "three"]


def test_generation_selection_preserves_content_before_one_overview(
    monkeypatch,
) -> None:
    monkeypatch.setattr(generation.settings, "max_generation_context_documents", 5)
    monkeypatch.setattr(generation.settings, "max_generation_context_overviews", 1)

    documents = [
        FakeDocument("overview-high", {"type": "overview"}),
        FakeDocument("content-1", {"type": "content"}),
        FakeDocument("overview-second", {"type": "overview"}),
        FakeDocument("content-2", {"type": "content"}),
        FakeDocument("content-3", {"type": "content"}),
        FakeDocument("content-4", {"type": "content"}),
        FakeDocument("content-5", {"type": "content"}),
    ]

    selected = generation.select_generation_documents(documents)

    assert [document.page_content for document in selected] == [
        "content-1",
        "content-2",
        "content-3",
        "content-4",
        "content-5",
        "overview-high",
    ]


def test_citation_context_uses_same_content_and_overview_selection(
    monkeypatch,
) -> None:
    monkeypatch.setattr(generation.settings, "max_generation_context_documents", 2)
    monkeypatch.setattr(generation.settings, "max_generation_context_overviews", 1)
    monkeypatch.setattr(generation.settings, "max_generation_context_chars", 1000)

    documents = [
        FakeDocument("overview", {"type": "overview", "filename": "paper.pdf"}),
        FakeDocument("content-1", {"type": "content", "filename": "paper.pdf", "page": 1}),
        FakeDocument("content-2", {"type": "content", "filename": "paper.pdf", "page": 2}),
        FakeDocument("content-3", {"type": "content", "filename": "paper.pdf", "page": 3}),
    ]

    context, _, catalog = generation.apply_citation_generation_limits(documents, [])

    assert "content-1" in context
    assert "content-2" in context
    assert "content-3" not in context
    assert context.index("content-2") < context.index("overview")
    assert catalog == [
        "[S1] paper.pdf, page 1",
        "[S2] paper.pdf, page 2",
        "[S3] paper.pdf, overview",
    ]
