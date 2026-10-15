from types import SimpleNamespace

from app.streamlit_support import (
    document_key,
    document_label,
    ingestion_summary,
    partition_uploads,
)


def test_pdf_uploads_use_v3_while_current_non_pdf_inputs_remain_v2() -> None:
    pdf = SimpleNamespace(name="paper.PDF")
    image = SimpleNamespace(name="diagram.png")
    audio = SimpleNamespace(name="meeting.mp3")

    v3, v2 = partition_uploads([pdf, image, audio])

    assert v3 == [pdf]
    assert v2 == [image, audio]


def test_document_identity_keeps_same_hash_in_v2_and_v3_distinct() -> None:
    v2 = {
        "document_id": "same-hash",
        "filename": "paper.pdf",
        "retrieval_version": "v2",
        "chunk_count": 4,
    }
    v3 = {
        "document_id": "same-hash",
        "filename": "paper.pdf",
        "retrieval_version": "v3",
        "record_count": 9,
    }

    assert document_key(v2) == "v2:same-hash"
    assert document_key(v3) == "v3:same-hash"
    assert document_label(v2) == "paper.pdf · V2 · 4 chunks · same-has"
    assert document_label(v3) == "paper.pdf · V3 · 9 records · same-has"


def test_v3_ingestion_summary_exposes_structural_output() -> None:
    assert ingestion_summary({
        "filename": "paper.pdf",
        "retrieval_version": "v3",
        "pages_parsed": 8,
        "artifacts_stored": 24,
        "search_records_indexed": 17,
    }) == (
        "paper.pdf: 8 pages parsed, 24 artifacts stored, and "
        "17 search records indexed"
    )
