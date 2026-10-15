"""Construction and safe validation for the selected V3 parser runtime."""

from __future__ import annotations

from typing import Any

from agentic_rag.ingestion.contracts import RetrievalSchema
from agentic_rag.ingestion.parsers.base import ParserAdapter
from agentic_rag.ingestion.parsers.pymupdf_layout_adapter import (
    EXPECTED_PACKAGE_VERSIONS,
    PyMuPDFLayoutParserAdapter,
)


SELECTED_V3_PARSER = "pymupdf_layout"
V3_CHUNKING_STRATEGY = "artifact_structure_v3"


def build_v3_parser(
    *,
    backend: str = SELECTED_V3_PARSER,
    render_dpi: int = 144,
    ocr_dpi: int = 150,
    enable_ocr: bool = True,
) -> ParserAdapter:
    """Build the selected parser and reject unreviewed runtime identities."""

    if backend != SELECTED_V3_PARSER:
        raise ValueError(
            f"unsupported V3 parser {backend!r}; expected {SELECTED_V3_PARSER!r}"
        )
    return PyMuPDFLayoutParserAdapter(
        render_dpi=render_dpi,
        ocr_dpi=ocr_dpi,
        enable_ocr=enable_ocr,
    )


def safe_runtime_summary(parser: ParserAdapter) -> dict[str, Any]:
    """Return dependency and capability identity without document content."""

    return {
        "parser_name": parser.name,
        "parser_revision": parser.revision,
        "parser_options": parser.parser_options,
        "capabilities": parser.capabilities.to_dict(),
        "expected_packages": dict(EXPECTED_PACKAGE_VERSIONS),
    }


def build_v3_retrieval_schema(
    parser: ParserAdapter,
    *,
    embedding_model: str,
    embedding_revision: str,
    embedding_dimension: int,
) -> RetrievalSchema:
    """Bind the selected parser and embedding identity into one V3 schema."""

    return RetrievalSchema(
        parser_name=parser.name,
        parser_revision=parser.revision,
        parser_options=parser.parser_options,
        chunking_strategy=V3_CHUNKING_STRATEGY,
        text_embedding_model=embedding_model,
        text_embedding_revision=embedding_revision,
        text_embedding_dimension=embedding_dimension,
    )
