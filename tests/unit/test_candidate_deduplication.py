from types import SimpleNamespace

from agentic_rag.retrieval.candidates import deduplicate_candidates


def _document(text: str, *, page: str = "1"):
    return SimpleNamespace(
        page_content=text,
        metadata={"document_id": "doc", "type": "content", "page_label": page},
    )


def test_candidate_deduplication_preserves_first_rrf_occurrence() -> None:
    first = (_document("Repeated  passage"), 0.03)
    duplicate = (_document(" repeated passage "), 0.02)
    other_page = (_document("repeated passage", page="2"), 0.01)

    kept, removed = deduplicate_candidates([first, duplicate, other_page])

    assert kept == [first, other_page]
    assert removed == 1
