import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_rag.ingestion.asset_store import LocalAssetStore
from agentic_rag.ingestion.contracts import (
    ArtifactKind,
    GroundingBasis,
    RepresentationKind,
    RetrievalSchema,
)
from agentic_rag.ingestion.parsers import pymupdf_layout_adapter as module


_VERSIONS = {
    "PyMuPDF": "1.28.2",
    "pymupdf4llm": "1.28.2",
    "pymupdf-layout": "1.28.2",
    "rapidocr": "3.9.2",
}


def _adapter(monkeypatch: pytest.MonkeyPatch, **kwargs) -> module.PyMuPDFLayoutParserAdapter:
    monkeypatch.setattr(module, "_package_versions", lambda: dict(_VERSIONS))
    return module.PyMuPDFLayoutParserAdapter(**kwargs)


def _schema(adapter: module.PyMuPDFLayoutParserAdapter, **overrides) -> RetrievalSchema:
    values = {
        "parser_name": adapter.name,
        "parser_revision": adapter.revision,
        "parser_options": adapter.parser_options,
        "chunking_strategy": "artifact_structure_v3",
        "text_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "text_embedding_revision": "revision-1",
        "text_embedding_dimension": 384,
    }
    values.update(overrides)
    return RetrievalSchema(**values)


class _FakePixmap:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def tobytes(self, format_name: str) -> bytes:
        assert format_name == "png"
        return self.content


class _FakePage:
    rect = SimpleNamespace(width=100.0, height=100.0)

    def get_pixmap(self, **kwargs):
        if "clip" in kwargs:
            return _FakePixmap(f"crop-{tuple(kwargs['clip'])}".encode())
        return _FakePixmap(b"page-png")

    def get_text(self, mode: str, *, sort: bool, clip=None):
        assert mode == "text"
        assert sort is True
        if clip is None:
            return "A structured title\nx = softmax(qk)"
        assert tuple(clip) == (10.0, 75.0, 90.0, 85.0)
        return "x = softmax(qk)"


class _FakeDocument:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __len__(self):
        return 1

    def load_page(self, index: int):
        assert index == 0
        return _FakePage()


def _fake_pymupdf():
    return SimpleNamespace(
        open=lambda _path: _FakeDocument(),
        Rect=lambda *values: tuple(values),
        mupdf=SimpleNamespace(FZ_STEXT_STROKED=1, FZ_STEXT_FILLED=2),
    )


def _line(text: str, *, ocr: bool = False) -> dict:
    return {
        "bbox": [10.0, 10.0, 90.0, 20.0],
        "spans": [
            {
                "text": text,
                "font": "OCR" if ocr else "Native",
                "char_flags": 0 if ocr else 2,
            }
        ],
    }


def _layout_payload() -> dict:
    return {
        "pages": [
            {
                "page_number": 1,
                "width": 100.0,
                "height": 100.0,
                "boxes": [
                    {
                        "x0": 10.0,
                        "y0": 5.0,
                        "x1": 90.0,
                        "y1": 15.0,
                        "boxclass": "title",
                        "header_level": 1,
                        "textlines": [_line("A structured title")],
                    },
                    {
                        "x0": 10.0,
                        "y0": 20.0,
                        "x1": 90.0,
                        "y1": 45.0,
                        "boxclass": "picture",
                        "header_level": 0,
                        "textlines": [_line("OCR diagram label", ocr=True)],
                    },
                    {
                        "x0": 10.0,
                        "y0": 50.0,
                        "x1": 90.0,
                        "y1": 70.0,
                        "boxclass": "table",
                        "header_level": 0,
                        "textlines": None,
                        "table": {
                            "markdown": "| Metric | Value |\n|---|---|\n| Recall | 0.91 |"
                        },
                    },
                    {
                        "x0": 10.0,
                        "y0": 75.0,
                        "x1": 90.0,
                        "y1": 85.0,
                        "boxclass": "formula",
                        "header_level": 0,
                        "textlines": None,
                    },
                    {
                        "x0": 10.0,
                        "y0": 90.0,
                        "x1": 90.0,
                        "y1": 95.0,
                        "boxclass": "page-footer",
                        "header_level": 0,
                        "textlines": [_line("Repeated footer")],
                    },
                ],
            }
        ]
    }


class _FakePyMuPDF4LLM:
    def __init__(self, expected_ocr) -> None:
        self.expected_ocr = expected_ocr
        self.calls: list[dict] = []

    def to_json(self, document, **kwargs):
        assert isinstance(document, _FakeDocument)
        assert kwargs["pages"] == [0]
        assert kwargs["force_text"] is True
        assert kwargs["ocr_function"] is self.expected_ocr
        self.calls.append(kwargs)
        return json.dumps(_layout_payload())


def test_adapter_identity_records_layout_and_ocr_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(monkeypatch)

    assert adapter.revision == (
        "PyMuPDF=1.28.2;pymupdf4llm=1.28.2;"
        "pymupdf-layout=1.28.2;rapidocr=3.9.2"
    )
    assert adapter.capabilities.headings is True
    assert adapter.capabilities.formulas is True
    assert adapter.capabilities.conditional_ocr is True
    assert adapter.parser_options["ocr"] == "conditional_rapidocr"
    assert adapter.parser_options["ignored_layout_classes"] == (
        "page-footer",
        "page-header",
    )


def test_adapter_rejects_unreviewed_package_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = dict(_VERSIONS)
    changed["pymupdf4llm"] = "1.28.3"
    monkeypatch.setattr(module, "_package_versions", lambda: changed)

    with pytest.raises(module.ParserDependencyUnavailable, match="version mismatch"):
        module.PyMuPDFLayoutParserAdapter()


@pytest.mark.parametrize(
    ("argument", "value"),
    (("render_dpi", 36), ("ocr_dpi", 700)),
)
def test_adapter_rejects_invalid_dpi(
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: int,
) -> None:
    monkeypatch.setattr(module, "_package_versions", lambda: dict(_VERSIONS))

    with pytest.raises(ValueError, match=argument):
        module.PyMuPDFLayoutParserAdapter(**{argument: value})


def test_table_markdown_falls_back_to_extracted_grid() -> None:
    assert module._table_markdown(
        {"extract": [["Name", "Value"], ["A|B", "one\ntwo"]]}
    ) == (
        "| Name | Value |\n"
        "| --- | --- |\n"
        "| A\\|B | one two |"
    )


def test_native_lineage_requires_substantial_word_overlap() -> None:
    native = "The transformer encoder uses multi head attention projections."

    assert module._is_native_text("Transformer encoder uses attention", native) is True
    assert module._is_native_text("Text recovered only through OCR", native) is False


def test_text_lines_preserve_phrases_across_pdf_wrapping() -> None:
    lines = [
        {"spans": [{"text": "The reparametriza-"}]},
        {"spans": [{"text": "tion in Algorithm"}]},
        {"spans": [{"text": " 1 is efficient."}]},
    ]

    assert module._text_from_lines(lines) == (
        "The reparametrization in Algorithm 1 is efficient."
    )


def test_text_lines_recover_spaces_from_span_geometry() -> None:
    lines = [
        {
            "spans": [
                {"text": "Algorithm", "bbox": [10, 10, 40, 20], "size": 10},
                {"text": "1", "bbox": [43, 10, 48, 20], "size": 10},
                {"text": ".", "bbox": [48, 10, 50, 20], "size": 10},
            ]
        }
    ]

    assert module._text_from_lines(lines) == "Algorithm 1."


def test_adapter_rejects_schema_from_another_parser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _adapter(monkeypatch)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf fixture")

    with pytest.raises(ValueError, match="does not match"):
        adapter._validate_request(
            source,
            (1,),
            _schema(adapter, parser_name="docling"),
        )


def test_parse_emits_ordered_regions_with_native_and_ocr_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _adapter(monkeypatch)
    ocr_function = object()
    layout = _FakePyMuPDF4LLM(ocr_function)
    monkeypatch.setattr(module, "_pymupdf", _fake_pymupdf)
    monkeypatch.setattr(module, "_pymupdf4llm", lambda: layout)
    monkeypatch.setattr(module, "_rapidocr_function", lambda: ocr_function)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf fixture")
    schema = _schema(adapter)

    first = adapter.parse(
        source,
        document_id="doc-123",
        selected_pages=(1,),
        schema=schema,
        asset_store=LocalAssetStore(tmp_path / "first-assets"),
    )
    second = adapter.parse(
        source,
        document_id="doc-123",
        selected_pages=(1,),
        schema=schema,
        asset_store=LocalAssetStore(tmp_path / "second-assets"),
    )

    assert [artifact.kind for artifact in first.artifacts] == [
        ArtifactKind.PAGE_IMAGE,
        ArtifactKind.TEXT_BLOCK,
        ArtifactKind.FIGURE,
        ArtifactKind.TABLE,
        ArtifactKind.FORMULA,
    ]
    assert [record.representation_kind for record in first.records] == [
        RepresentationKind.NATIVE_TEXT,
        RepresentationKind.OCR_TEXT,
        RepresentationKind.TABLE_MARKDOWN,
        RepresentationKind.NATIVE_TEXT,
    ]
    assert [record.grounding_basis for record in first.records] == [
        GroundingBasis.DIRECT_TEXT,
        GroundingBasis.OCR_DERIVED,
        GroundingBasis.TABLE_EXTRACTION,
        GroundingBasis.DIRECT_TEXT,
    ]
    assert all("Repeated footer" not in record.search_text for record in first.records)
    assert all(
        artifact.parent_artifact_id is not None
        for artifact in first.artifacts
        if artifact.kind is not ArtifactKind.PAGE_IMAGE
    )
    assert {artifact.artifact_id for artifact in first.artifacts} == {
        artifact.artifact_id for artifact in second.artifacts
    }
    assert {record.record_id for record in first.records} == {
        record.record_id for record in second.records
    }
    assert len(layout.calls) == 2


def test_disable_ocr_changes_parser_identity_and_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _adapter(monkeypatch, enable_ocr=False)
    layout = _FakePyMuPDF4LLM(None)
    monkeypatch.setattr(module, "_pymupdf", _fake_pymupdf)
    monkeypatch.setattr(module, "_pymupdf4llm", lambda: layout)
    monkeypatch.setattr(
        module,
        "_rapidocr_function",
        lambda: pytest.fail("RapidOCR must not be imported when OCR is disabled"),
    )
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf fixture")

    adapter.parse(
        source,
        document_id="doc-123",
        selected_pages=(1,),
        schema=_schema(adapter),
        asset_store=LocalAssetStore(tmp_path / "assets"),
    )

    assert adapter.parser_options["ocr"] == "disabled"
    assert layout.calls[0]["use_ocr"] is False
    assert layout.calls[0]["ocr_function"] is None
