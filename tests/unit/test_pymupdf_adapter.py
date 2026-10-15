from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_rag.ingestion.asset_store import LocalAssetStore
from agentic_rag.ingestion.contracts import ArtifactKind, RepresentationKind, RetrievalSchema
from agentic_rag.ingestion.parsers import pymupdf_adapter as module


def _adapter(monkeypatch: pytest.MonkeyPatch) -> module.PyMuPDFParserAdapter:
    monkeypatch.setattr(module, "_package_version", lambda: "1.28.2")
    return module.PyMuPDFParserAdapter(render_dpi=144)


def _schema(adapter: module.PyMuPDFParserAdapter, **overrides) -> RetrievalSchema:
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


class _FakeTable:
    bbox = (10.0, 60.0, 90.0, 90.0)

    def to_markdown(self) -> str:
        return "| Metric | Value |\n|---|---|\n| Recall | 0.91 |"


class _FakePage:
    rect = SimpleNamespace(width=100.0, height=100.0)

    def get_pixmap(self, **kwargs):
        if "clip" in kwargs:
            return _FakePixmap(b"figure-png")
        return _FakePixmap(b"page-png")

    def get_text(self, mode: str, *, sort: bool):
        assert mode == "dict"
        assert sort is True
        return {
            "blocks": [
                {
                    "type": 0,
                    "bbox": (10.0, 20.0, 90.0, 40.0),
                    "lines": [
                        {"spans": [{"text": "Native PDF text"}]},
                        {"spans": [{"text": "continues here."}]},
                    ],
                },
                {
                    "type": 1,
                    "bbox": (20.0, 42.0, 80.0, 58.0),
                    "image": b"embedded-image",
                },
            ]
        }

    def find_tables(self):
        return SimpleNamespace(tables=[_FakeTable()])


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
    )


def test_adapter_declares_fast_path_capabilities_and_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(monkeypatch)

    assert adapter.revision == "PyMuPDF=1.28.2"
    assert adapter.capabilities.text_blocks is True
    assert adapter.capabilities.tables is True
    assert adapter.capabilities.formulas is False
    assert adapter.capabilities.conditional_ocr is False
    assert adapter.parser_options["page_render_dpi"] == 144


def test_adapter_rejects_invalid_render_dpi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_package_version", lambda: "1.28.2")

    with pytest.raises(ValueError, match="render_dpi"):
        module.PyMuPDFParserAdapter(render_dpi=36)


def test_normalized_bbox_clamps_page_coordinates() -> None:
    assert module._normalized_bbox(
        (-10.0, 20.0, 120.0, 180.0),
        page_width=100.0,
        page_height=200.0,
    ) == (0.0, 0.1, 1.0, 0.9)
    assert module._normalized_bbox(
        (10.0, 10.0, 10.0, 20.0),
        page_width=100.0,
        page_height=200.0,
    ) is None


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


def test_parse_emits_stable_native_figure_table_and_page_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(module, "_pymupdf", _fake_pymupdf)
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

    assert {artifact.kind for artifact in first.artifacts} == {
        ArtifactKind.PAGE_IMAGE,
        ArtifactKind.TEXT_BLOCK,
        ArtifactKind.FIGURE,
        ArtifactKind.TABLE,
    }
    assert {record.representation_kind for record in first.records} == {
        RepresentationKind.NATIVE_TEXT,
        RepresentationKind.TABLE_MARKDOWN,
    }
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


def test_parse_warns_when_page_has_no_native_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _adapter(monkeypatch)
    empty_page = _FakePage()
    monkeypatch.setattr(empty_page, "get_text", lambda *_args, **_kwargs: {"blocks": []})
    document = _FakeDocument()
    monkeypatch.setattr(document, "load_page", lambda _index: empty_page)
    monkeypatch.setattr(
        module,
        "_pymupdf",
        lambda: SimpleNamespace(open=lambda _path: document, Rect=lambda *values: values),
    )
    source = tmp_path / "scan.pdf"
    source.write_bytes(b"pdf fixture")

    run = adapter.parse(
        source,
        document_id="doc-scan",
        selected_pages=(1,),
        schema=_schema(adapter),
        asset_store=LocalAssetStore(tmp_path / "assets"),
    )

    assert run.warnings == ("no_native_text:page=1",)
    assert {artifact.kind for artifact in run.artifacts} == {
        ArtifactKind.PAGE_IMAGE,
        ArtifactKind.TABLE,
    }


def test_parse_rejects_page_outside_pdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(module, "_pymupdf", _fake_pymupdf)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf fixture")

    with pytest.raises(ValueError, match="outside the PDF"):
        adapter.parse(
            source,
            document_id="doc-123",
            selected_pages=(2,),
            schema=_schema(adapter),
            asset_store=LocalAssetStore(tmp_path / "assets"),
        )
