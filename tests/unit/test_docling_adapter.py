from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_rag.ingestion.asset_store import LocalAssetStore
from agentic_rag.ingestion.contracts import (
    ArtifactKind,
    Derivation,
    RetrievalSchema,
)
from agentic_rag.ingestion.parsers import docling_adapter as module


_VERSIONS = {
    "docling": "2.129.0",
    "docling-core": "2.101.1",
    "docling-parse": "7.22.2",
    "docling-ibm-models": "4.0.3",
    "rapidocr": "3.9.2",
}


def _adapter(monkeypatch: pytest.MonkeyPatch) -> module.DoclingParserAdapter:
    monkeypatch.setattr(module, "_package_versions", lambda: dict(_VERSIONS))
    return module.DoclingParserAdapter(artifacts_path="/models")


def _schema(adapter: module.DoclingParserAdapter, **overrides) -> RetrievalSchema:
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


def test_adapter_revision_records_every_parser_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(monkeypatch)

    assert adapter.revision == (
        "docling=2.129.0;docling-core=2.101.1;docling-parse=7.22.2;"
        "docling-ibm-models=4.0.3;rapidocr=3.9.2"
    )
    assert adapter.capabilities.formulas is True
    assert adapter.capabilities.conditional_ocr is True


def test_adapter_rejects_unknown_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_package_versions", lambda: dict(_VERSIONS))

    with pytest.raises(ValueError, match="accelerator_device"):
        module.DoclingParserAdapter(
            artifacts_path="/models",
            accelerator_device="unsupported",
        )


def test_accelerator_is_part_of_parser_and_index_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(module, "_package_versions", lambda: dict(_VERSIONS))
    cpu = module.DoclingParserAdapter(
        artifacts_path="/models",
        accelerator_device="cpu",
    )
    cuda = module.DoclingParserAdapter(
        artifacts_path="/models",
        accelerator_device="cuda",
    )
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf fixture")

    assert cpu.parser_options["accelerator_device"] == "cpu"
    assert cuda.parser_options["accelerator_device"] == "cuda"
    assert _schema(cpu).schema_id != _schema(cuda).schema_id
    with pytest.raises(ValueError, match="does not match"):
        cuda._validate_request(source, (1,), _schema(cpu))


def test_adapter_reuses_initialized_converter_within_one_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(monkeypatch)
    converter = object()
    creations = 0

    def create_converter():
        nonlocal creations
        creations += 1
        return converter

    monkeypatch.setattr(adapter, "_create_converter", create_converter)

    assert adapter._converter() is converter
    assert adapter._converter() is converter
    assert creations == 1


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
            _schema(adapter, parser_name="pymupdf"),
        )


def test_native_text_lineage_requires_substantial_word_overlap() -> None:
    native = "The transformer encoder uses multi head attention projections."

    assert module._is_native_text("Transformer encoder uses attention", native) is True
    assert module._is_native_text("Text recovered only through OCR", native) is False


def test_native_page_text_uses_one_based_pdf_page_numbering() -> None:
    reader = SimpleNamespace(
        pages=[
            SimpleNamespace(extract_text=lambda: "first page"),
            SimpleNamespace(extract_text=lambda: "  second page  "),
        ]
    )

    assert module.DoclingParserAdapter._native_page_text(reader, 2) == "second page"


def test_asset_artifact_stores_bytes_under_its_stable_id(tmp_path: Path) -> None:
    store = LocalAssetStore(tmp_path / "assets")
    artifact = module._asset_artifact(
        document_id="doc-123",
        kind=ArtifactKind.PAGE_IMAGE,
        content=b"deterministic image bytes",
        derivation=Derivation(producer="docling", revision="2.129.0"),
        locator=module.SourceLocator(page_start=1, page_end=1),
        asset_store=store,
    )

    assert artifact.asset_key is not None
    assert len(artifact.asset_key) < 120
    with store.open(artifact.asset_key) as stream:
        assert stream.read() == b"deterministic image bytes"


def test_source_locator_normalizes_top_left_coordinates() -> None:
    class Box:
        def to_top_left_origin(self, _height):
            return self

        def normalized(self, _size):
            return SimpleNamespace(l=0.1, t=0.2, r=0.8, b=0.9)

    item = SimpleNamespace(prov=[SimpleNamespace(page_no=3, bbox=Box())])
    document = SimpleNamespace(
        pages={3: SimpleNamespace(size=SimpleNamespace(width=100, height=200))}
    )

    locator = module._source_locator(item, document)

    assert locator is not None
    assert locator.page_start == 3
    assert locator.bbox == (0.1, 0.2, 0.8, 0.9)
