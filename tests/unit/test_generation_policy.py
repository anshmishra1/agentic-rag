from dataclasses import dataclass

from agentic_rag.policies import generation


@dataclass
class FakeDocument:
    page_content: str


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
