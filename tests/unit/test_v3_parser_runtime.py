from types import SimpleNamespace

import pytest

from agentic_rag.ingestion import parser_runtime


def _fake_parser(**kwargs):
    return SimpleNamespace(
        name="pymupdf_layout",
        revision="revision-1",
        parser_options=kwargs,
        capabilities=SimpleNamespace(to_dict=lambda: {"tables": True}),
    )


def test_runtime_builds_only_the_selected_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return _fake_parser(**kwargs)

    monkeypatch.setattr(parser_runtime, "PyMuPDFLayoutParserAdapter", factory)

    parser = parser_runtime.build_v3_parser(
        render_dpi=180,
        ocr_dpi=200,
        enable_ocr=False,
    )

    assert parser.name == "pymupdf_layout"
    assert calls == [
        {"render_dpi": 180, "ocr_dpi": 200, "enable_ocr": False}
    ]


def test_runtime_rejects_unreviewed_parser_backend() -> None:
    with pytest.raises(ValueError, match="unsupported V3 parser"):
        parser_runtime.build_v3_parser(backend="docling")


def test_runtime_summary_contains_identity_without_document_content() -> None:
    summary = parser_runtime.safe_runtime_summary(_fake_parser(render_dpi=144))

    assert summary["parser_name"] == "pymupdf_layout"
    assert summary["parser_revision"] == "revision-1"
    assert summary["parser_options"] == {"render_dpi": 144}
    assert summary["capabilities"] == {"tables": True}
    assert summary["expected_packages"] == {
        "PyMuPDF": "1.28.2",
        "pymupdf4llm": "1.28.2",
        "pymupdf-layout": "1.28.2",
        "rapidocr": "3.9.2",
    }


def test_schema_factory_binds_parser_embedding_and_v3_chunking_identity() -> None:
    parser = _fake_parser(render_dpi=144, accelerator_device="cpu")

    schema = parser_runtime.build_v3_retrieval_schema(
        parser,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_revision="embedding-revision",
        embedding_dimension=384,
    )

    assert schema.parser_name == parser.name
    assert schema.parser_revision == parser.revision
    assert schema.parser_options == parser.parser_options
    assert schema.chunking_strategy == "artifact_structure_v3_token_safe_240"
    assert schema.representation_contract == "artifact_search_record_v2"
    assert schema.text_embedding_dimension == 384
    assert schema.schema_id.startswith("multimodal_v3:")
