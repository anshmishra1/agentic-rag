from dataclasses import dataclass, field

from agentic_rag.policies.citations import (
    build_citation_context,
    cited_verification_context,
    citation_description,
    extract_used_citations,
    normalize_citation_markers,
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


def test_alternate_source_brackets_are_normalized_before_extraction() -> None:
    catalog = ["[S1] guide.pdf, page 1", "[S2] guide.pdf, page 2"]
    answer = "First claim 【S2】. Second claim [S1]. Repeat 【S2】; ignore 【S9】."

    normalized = normalize_citation_markers(answer)

    assert normalized == "First claim [S2]. Second claim [S1]. Repeat [S2]; ignore [S9]."
    assert extract_used_citations(normalized, catalog) == [
        "[S2] guide.pdf, page 2",
        "[S1] guide.pdf, page 1",
    ]


def test_verification_only_sees_sources_the_answer_cites() -> None:
    documents = [
        FakeDocument("Representation models do not generate text; classification is an example.", {"filename": "book.pdf", "page": 9}),
        FakeDocument("Clustering and semantic search are discussed elsewhere.", {"filename": "book.pdf", "page": 45}),
    ]

    context, issues = cited_verification_context(
        documents,
        "Representation models perform classification [S1].",
        max_documents=2,
        max_chars=500,
    )

    assert issues == []
    assert "[S1] book.pdf, page 9" in context
    assert "Clustering and semantic search" not in context
    assert "[S2]" not in context

    second_context, second_issues = cited_verification_context(
        documents,
        "The later passage discusses clustering [S2].",
        max_documents=2,
        max_chars=500,
    )
    assert second_issues == []
    assert second_context.startswith("[S2] book.pdf, page 45\n")
    assert "[S1]" not in second_context


def test_uncited_bullets_fail_even_with_a_citation_at_answer_end() -> None:
    documents = [
        FakeDocument("Representation models do not generate text; classification is an example.", {"filename": "book.pdf", "page": 9}),
        FakeDocument("Clustering and semantic search are discussed elsewhere.", {"filename": "book.pdf", "page": 45}),
    ]
    answer = (
        "* Representation models are used for classification, clustering, and semantic search.\n\n"
        "* Generative models produce text.\n\n"
        "The models differ in whether they generate text [S1]."
    )

    context, issues = cited_verification_context(
        documents, answer, max_documents=2, max_chars=500
    )

    assert len(issues) == 2
    assert all("List item lacks a source citation" in issue for issue in issues)
    assert "Clustering and semantic search" not in context


def test_missing_and_unknown_citations_fail_closed() -> None:
    documents = [FakeDocument("Evidence", {"filename": "book.pdf", "page": 1})]

    _, missing = cited_verification_context(
        documents, "An uncited claim.", max_documents=1, max_chars=200
    )
    _, unknown = cited_verification_context(
        documents, "An invented citation [S9].", max_documents=1, max_chars=200
    )

    assert missing == ["The answer has no source citation."]
    assert unknown == ["The answer cites unavailable source [S9]."]


def test_citation_on_wrapped_list_item_is_accepted() -> None:
    documents = [FakeDocument("Supported statement", {"filename": "book.pdf", "page": 1})]

    _, issues = cited_verification_context(
        documents,
        "* A supported statement\n  continues here [S1].",
        max_documents=1,
        max_chars=200,
    )

    assert issues == []
