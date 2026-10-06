from langchain_core.documents import Document
import pytest

from agentic_rag.ingestion import chunking


def _tokens(text: str) -> int:
    return len(chunking._get_tokenizer().encode(text, add_special_tokens=False))


def test_pdf_extraction_mode_is_versioned_with_chunking_strategy():
    assert chunking.pdf_extraction_mode(chunking.TOKEN_WINDOW_V1) == "plain"
    assert chunking.pdf_extraction_mode(chunking.STRUCTURE_AWARE_V2) == "layout"
    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        chunking.pdf_extraction_mode("unknown")


def test_appendix_reference_sentence_is_not_treated_as_a_heading():
    assert chunking._looks_like_heading("Appendix E") is True
    assert chunking._looks_like_heading("Appendix E: Additional Results") is True
    assert (
        chunking._looks_like_heading(
            "Appendix E includes plots of the validation perplexity."
        )
        is False
    )


def test_structure_aware_chunks_preserve_sections_sentences_and_page_spans(monkeypatch):
    monkeypatch.setattr(chunking.settings, "chunk_size", 32)
    documents = [
        Document(
            page_content=(
                "1 Introduction\n"
                "The first sentence establishes the topic.\n"
                "The second sentence adds enough detail to approach the token limit.\n"
                "The third sentence remains intact when a new chunk is required.\n"
            ),
            metadata={"page": 0, "page_label": "1", "extraction_mode": "layout"},
        ),
        Document(
            page_content=(
                "The fourth sentence continues the same section on another page.\n"
                "2 Method\n"
                "The method sentence belongs under a different heading.\n"
            ),
            metadata={"page": 1, "page_label": "2", "extraction_mode": "layout"},
        ),
    ]

    chunks = chunking.chunk_documents(
        documents,
        document_id="doc-id",
        filename="paper.pdf",
        strategy=chunking.STRUCTURE_AWARE_V2,
    )

    assert len(chunks) >= 3
    assert all(_tokens(document.page_content) <= 32 for document in chunks)
    assert all(document.metadata["chunking_strategy"] == chunking.STRUCTURE_AWARE_V2 for document in chunks)
    assert all(document.metadata["filename"] == "paper.pdf" for document in chunks)
    assert all(document.metadata["document_id"] == "doc-id" for document in chunks)
    assert any(document.metadata["page_label"] == "1-2" for document in chunks)
    assert any(document.page_content.startswith("2 Method\n\n") for document in chunks)
    assert "The third sentence remains intact" in " ".join(document.page_content for document in chunks)


def test_structure_aware_chunks_split_oversized_units_without_fixed_overlap(monkeypatch):
    monkeypatch.setattr(chunking.settings, "chunk_size", 24)
    text = "3 Results\n" + " ".join(f"token{index}" for index in range(90))
    chunks = chunking.chunk_documents(
        [Document(page_content=text, metadata={"page": 2, "page_label": "3", "extraction_mode": "layout"})],
        strategy=chunking.STRUCTURE_AWARE_V2,
    )

    assert len(chunks) > 1
    assert all(_tokens(document.page_content) <= 24 for document in chunks)
    assert all(document.metadata["section_title"] == "3 Results" for document in chunks)
    bodies = [document.page_content.split("\n\n", 1)[1] for document in chunks]
    assert len(" ".join(bodies).split()) == 90


def test_token_window_v1_keeps_plain_content_and_marks_schema(monkeypatch):
    monkeypatch.setattr(chunking.settings, "chunk_size", 240)
    monkeypatch.setattr(chunking.settings, "chunk_overlap", 40)
    chunking._get_token_window_splitter.cache_clear()
    source = "A short legacy passage that remains in one chunk."

    chunks = chunking.chunk_documents(
        [Document(page_content=source, metadata={"page": 0, "page_label": "1"})],
        strategy=chunking.TOKEN_WINDOW_V1,
    )

    assert [document.page_content for document in chunks] == [source]
    assert chunks[0].metadata["chunking_strategy"] == chunking.TOKEN_WINDOW_V1
    assert chunks[0].metadata["page_start"] == chunks[0].metadata["page_end"] == "1"


def test_structure_aware_chunks_merge_exact_duplicates_with_page_provenance(monkeypatch):
    monkeypatch.setattr(chunking.settings, "chunk_size", 240)
    chunks = [
        Document(
            page_content="Repeated table note.",
            metadata={
                "page": 3,
                "page_label": "4",
                "page_start": "4",
                "page_end": "4",
                "source_unit_count": 1,
            },
        ),
        Document(
            page_content="Repeated table note.",
            metadata={
                "page": 4,
                "page_label": "5",
                "page_start": "5",
                "page_end": "5",
                "source_unit_count": 1,
            },
        ),
    ]

    chunks = chunking._deduplicate_chunks(chunks)

    assert len(chunks) == 1
    assert chunks[0].metadata["page_label"] == "4-5"
    assert chunks[0].metadata["duplicate_occurrences"] == 2
