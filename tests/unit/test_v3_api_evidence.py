from langchain_core.documents import Document

from agentic_rag.api.evidence import build_structured_evidence


def test_only_cited_v3_documents_become_structured_evidence() -> None:
    documents = [
        Document(
            page_content="figure evidence",
            metadata={
                "document_id": "doc-1",
                "artifact_id": "artifact-1",
                "record_id": "record-1",
                "artifact_kind": "figure",
                "grounding_basis": "vision_description",
                "filename": "paper.pdf",
                "page_start": 2,
                "page_end": 2,
                "page_label": "2",
                "bbox_left": 0.1,
                "bbox_top": 0.2,
                "bbox_right": 0.8,
                "bbox_bottom": 0.7,
                "type": "content",
            },
        ),
        Document(
            page_content="uncited evidence",
            metadata={
                "document_id": "doc-1",
                "artifact_id": "artifact-2",
                "record_id": "record-2",
                "artifact_kind": "text_block",
                "grounding_basis": "direct_text",
                "filename": "paper.pdf",
                "page_label": "3",
                "type": "content",
            },
        ),
    ]

    evidence = build_structured_evidence(
        documents,
        ["[S1] paper.pdf, page 2"],
        retrieval_version="v3",
    )

    assert evidence == [{
        "label": "S1",
        "document_id": "doc-1",
        "artifact_id": "artifact-1",
        "record_id": "record-1",
        "kind": "figure",
        "grounding_basis": "vision_description",
        "source_name": "paper.pdf",
        "description": "[S1] paper.pdf, page 2, figure evidence (vision description)",
        "locator": {
            "page_start": 2,
            "page_end": 2,
            "page_label": "2",
            "bbox": [0.1, 0.2, 0.8, 0.7],
            "start_ms": None,
            "end_ms": None,
        },
        "asset_url": "/v3/documents/doc-1/artifacts/artifact-1/asset",
    }]


def test_v2_citations_do_not_claim_v3_artifact_provenance() -> None:
    document = Document(
        page_content="legacy chunk",
        metadata={"filename": "paper.pdf", "page_label": "1"},
    )

    assert build_structured_evidence(
        [document],
        ["[S1] paper.pdf, page 1"],
        retrieval_version="v2",
    ) == []
