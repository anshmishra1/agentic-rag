from dataclasses import dataclass, field

from agentic_rag.policies.citations import (
    build_citation_context,
    citation_description,
    extract_used_citations,
)


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)


def test_citation_description_uses_filename_and_page() -> None:
    document = FakeDocument(
        "evidence",
        {"filename": "guide.pdf", "page_label": "3", "type": "content"},
    )

    assert citation_description(document, 1) == "[S1] guide.pdf, page 3"


def test_citation_description_falls_back_to_chunk_type() -> None:
    document = FakeDocument(
        "summary",
        {"source": "guide.pdf", "type": "overview"},
    )

    assert citation_description(document, 2) == "[S2] guide.pdf, overview"


def test_context_labels_are_deterministic_and_bounded() -> None:
    documents = [
        FakeDocument("first evidence", {"filename": "guide.pdf", "page": 1}),
        FakeDocument("second evidence", {"filename": "guide.pdf", "page": 2}),
        FakeDocument("third evidence", {"filename": "guide.pdf", "page": 3}),
    ]

    context, catalog = build_citation_context(
        documents,
        max_documents=2,
        max_chars=500,
    )

    assert catalog == [
        "[S1] guide.pdf, page 1",
        "[S2] guide.pdf, page 2",
    ]
    assert "[S1] guide.pdf, page 1\nfirst evidence" in context
    assert "[S2] guide.pdf, page 2\nsecond evidence" in context
    assert "third evidence" not in context
    assert len(context) <= 500


def test_context_does_not_publish_a_label_without_evidence() -> None:
    document = FakeDocument("evidence", {"filename": "guide.pdf", "page": 1})

    context, catalog = build_citation_context(
        [document],
        max_documents=1,
        max_chars=10,
    )

    assert context == ""
    assert catalog == []


def test_only_valid_answer_citations_are_returned() -> None:
    catalog = [
        "[S1] guide.pdf, page 1",
        "[S2] guide.pdf, page 2",
    ]

    citations = extract_used_citations(
        "The first claim [S2] supports the second [S1]. Ignore [S9] and [S2].",
        catalog,
    )

    assert citations == [
        "[S2] guide.pdf, page 2",
        "[S1] guide.pdf, page 1",
    ]
