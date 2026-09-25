import pytest

from agentic_rag.evaluation.dataset_audit import audit_question_set


def test_audit_separates_supported_scope_and_missing_sources(tmp_path):
    (tmp_path / "present.pdf").touch()
    rows = [
        {"user_input": "Question A", "reference": "Answer A", "source_document": "present.pdf", "source_pages": "2-3"},
        {"user_input": "Question B", "reference": "Answer B", "source_document": "missing.pdf", "source_pages": "Chapter 2"},
        {"user_input": "Question C", "reference": "Answer C", "source_document": "present.pdf + missing.pdf", "source_pages": "2"},
    ]

    report = audit_question_set(rows, tmp_path)

    assert report["single_document_questions"] == 2
    assert report["cross_document_questions"] == 1
    assert report["available_documents"] == ["present.pdf"]
    assert report["missing_documents"] == ["missing.pdf"]
    assert report["numeric_page_references"] == 1
    assert report["ambiguous_page_references"] == 1
    assert report["relevance_labeled_questions"] == 0


def test_audit_rejects_incomplete_cases(tmp_path):
    with pytest.raises(ValueError, match="missing: source_pages"):
        audit_question_set(
            [{"user_input": "Q", "reference": "A", "source_document": "a.pdf"}],
            tmp_path,
        )
