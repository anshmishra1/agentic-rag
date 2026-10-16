from langchain_core.documents import Document

from agentic_rag.policies.evidence import (
    evidence_requirement_is_satisfied,
    select_required_evidence,
)


def _doc(record: str, kind: str, page: str, score: float):
    return (
        Document(
            page_content=record,
            metadata={
                "record_id": record,
                "artifact_id": f"artifact-{record}",
                "artifact_kind": kind,
                "page_label": page,
                "parent_artifact_id": f"page-{page}",
            },
        ),
        score,
    )


def test_visual_selection_retains_rrf_artifacts_and_same_page_context() -> None:
    rrf_ranked = [
        _doc("text-a", "text_block", "1", 0.05),
        _doc("figure-correct", "figure", "2", 0.04),
        _doc("figure-other", "figure", "3", 0.03),
        _doc("caption", "text_block", "2", 0.02),
        _doc("text-b", "text_block", "4", 0.01),
        _doc("text-c", "text_block", "5", 0.01),
    ]
    reranked = [
        _doc("text-a", "text_block", "1", 0.99),
        _doc("text-b", "text_block", "4", 0.98),
        _doc("text-c", "text_block", "5", 0.97),
        _doc("caption", "text_block", "2", 0.70),
        _doc("figure-other", "figure", "3", 0.20),
        _doc("figure-correct", "figure", "2", 0.10),
    ]

    selected = select_required_evidence(
        reranked,
        rrf_ranked,
        requirement="visual",
        top_k=5,
        required_slots=2,
        companion_slots=1,
    )
    selected_ids = {document.metadata["record_id"] for document, _ in selected}

    assert {"figure-correct", "figure-other", "caption"}.issubset(selected_ids)
    assert len(selected) == 5
    assert evidence_requirement_is_satisfied(
        [document for document, _ in selected], "visual"
    )


def test_text_selection_preserves_reranker_top_k() -> None:
    reranked = [_doc(f"text-{index}", "text_block", str(index), 1 - index / 10) for index in range(7)]

    selected = select_required_evidence(
        reranked,
        list(reversed(reranked)),
        requirement="text",
        top_k=5,
        required_slots=2,
        companion_slots=1,
    )

    assert selected == reranked[:5]
    assert not evidence_requirement_is_satisfied(
        [document for document, _ in selected], "table"
    )
