"""Optional PyMuPDF adapter for the multimodal V3 parser bakeoff."""

from __future__ import annotations

import importlib.metadata
import math
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from agentic_rag.ingestion.asset_store import AssetStore
from agentic_rag.ingestion.contracts import (
    Artifact,
    ArtifactKind,
    Derivation,
    GroundingBasis,
    RepresentationKind,
    RetrievalSchema,
    SearchRecord,
    SourceLocator,
    sha256_bytes,
)
from agentic_rag.ingestion.parsers.base import (
    ParseRun,
    ParserCapabilities,
    ParserDependencyUnavailable,
)


def _package_version() -> str:
    try:
        return importlib.metadata.version("PyMuPDF")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ParserDependencyUnavailable(
            "PyMuPDF is not installed; provision it only in the disposable parser "
            "comparison runtime until a parser is selected"
        ) from exc


def _pymupdf() -> Any:
    try:
        import pymupdf
    except ImportError as exc:
        raise ParserDependencyUnavailable(
            "PyMuPDF imports failed inside the parser comparison runtime"
        ) from exc
    return pymupdf


def _peak_memory_mib() -> float:
    try:
        import resource
    except ImportError:  # pragma: no cover - comparison execution uses Linux Docker
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _normalized_bbox(
    value: Sequence[float] | None,
    *,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float] | None:
    if value is None or len(value) != 4 or page_width <= 0 or page_height <= 0:
        return None
    left, top, right, bottom = (
        float(value[0]) / page_width,
        float(value[1]) / page_height,
        float(value[2]) / page_width,
        float(value[3]) / page_height,
    )
    if not all(math.isfinite(item) for item in (left, top, right, bottom)):
        return None
    left, top, right, bottom = (
        min(1.0, max(0.0, item)) for item in (left, top, right, bottom)
    )
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def _text_from_block(block: dict[str, Any]) -> str:
    lines: list[str] = []
    for line in block.get("lines", ()) or ():
        text = "".join(
            str(span.get("text", "")) for span in line.get("spans", ()) or ()
        ).strip()
        if text:
            lines.append(text)
    return "\n".join(lines).strip()


def _asset_artifact(
    *,
    document_id: str,
    kind: ArtifactKind,
    content: bytes,
    derivation: Derivation,
    locator: SourceLocator,
    asset_store: AssetStore,
    parent_artifact_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    digest = sha256_bytes(content)
    draft = Artifact(
        document_id=document_id,
        kind=kind,
        content_sha256=digest,
        derivation=derivation,
        locator=locator,
        asset_key=f"pending:{digest}",
        parent_artifact_id=parent_artifact_id,
        metadata=metadata or {},
    )
    stored = asset_store.put(
        document_id=document_id,
        artifact_id=draft.artifact_id,
        content=content,
        media_type="image/png",
    )
    return replace(draft, asset_key=stored.asset_key)


class PyMuPDFParserAdapter:
    """Emit deterministic V3 evidence from PyMuPDF's native PDF APIs."""

    name = "pymupdf"
    capabilities = ParserCapabilities(
        text_blocks=True,
        headings=False,
        figures=True,
        tables=True,
        formulas=False,
        conditional_ocr=False,
        page_images=True,
    )

    def __init__(self, *, render_dpi: int = 144) -> None:
        if render_dpi < 72 or render_dpi > 600:
            raise ValueError("render_dpi must be between 72 and 600")
        self.render_dpi = render_dpi
        self.version = _package_version()
        self.revision = f"PyMuPDF={self.version}"

    @property
    def parser_options(self) -> dict[str, Any]:
        return {
            "embedded_image_crops": True,
            "formula_extraction": False,
            "ocr": False,
            "page_render_dpi": self.render_dpi,
            "table_extraction": "find_tables_markdown",
            "text_extraction": "dict_sorted",
        }

    def _validate_request(
        self,
        path: Path,
        selected_pages: tuple[int, ...],
        schema: RetrievalSchema,
    ) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.casefold() != ".pdf":
            raise ValueError("PyMuPDF parser adapter currently accepts PDF files only")
        if not selected_pages or tuple(sorted(set(selected_pages))) != selected_pages:
            raise ValueError("selected_pages must be sorted, unique, and nonempty")
        if any(page < 1 for page in selected_pages):
            raise ValueError("selected_pages must use one-based page numbers")
        if (
            schema.parser_name != self.name
            or schema.parser_revision != self.revision
            or dict(schema.parser_options) != self.parser_options
        ):
            raise ValueError("retrieval schema does not match the PyMuPDF parser runtime")

    @staticmethod
    def _locator(
        page_number: int,
        bbox: Sequence[float] | None,
        *,
        page_width: float,
        page_height: float,
    ) -> SourceLocator:
        return SourceLocator(
            page_start=page_number,
            page_end=page_number,
            page_label=str(page_number),
            bbox=_normalized_bbox(
                bbox,
                page_width=page_width,
                page_height=page_height,
            ),
        )

    def parse(
        self,
        path: Path,
        *,
        document_id: str,
        selected_pages: Sequence[int],
        schema: RetrievalSchema,
        asset_store: AssetStore,
    ) -> ParseRun:
        path = Path(path)
        pages = tuple(selected_pages)
        self._validate_request(path, pages, schema)
        pymupdf = _pymupdf()
        derivation = Derivation(
            producer=self.name,
            revision=self.revision,
            parameters=self.parser_options,
        )
        artifacts: list[Artifact] = []
        records: list[SearchRecord] = []
        warnings: list[str] = []
        started = time.perf_counter()

        with pymupdf.open(str(path)) as document:
            for page_number in pages:
                if page_number > len(document):
                    raise ValueError(f"selected page {page_number} is outside the PDF")
                page = document.load_page(page_number - 1)
                page_width = float(page.rect.width)
                page_height = float(page.rect.height)
                page_locator = self._locator(
                    page_number,
                    None,
                    page_width=page_width,
                    page_height=page_height,
                )
                page_content = page.get_pixmap(
                    dpi=self.render_dpi,
                    alpha=False,
                ).tobytes("png")
                page_artifact = _asset_artifact(
                    document_id=document_id,
                    kind=ArtifactKind.PAGE_IMAGE,
                    content=page_content,
                    derivation=derivation,
                    locator=page_locator,
                    asset_store=asset_store,
                    metadata={"dpi": self.render_dpi},
                )
                artifacts.append(page_artifact)

                native_text_count = 0
                blocks = page.get_text("dict", sort=True).get("blocks", ())
                for block_number, block in enumerate(blocks):
                    block_type = int(block.get("type", -1))
                    locator = self._locator(
                        page_number,
                        block.get("bbox"),
                        page_width=page_width,
                        page_height=page_height,
                    )
                    if block_type == 0:
                        text = _text_from_block(block)
                        if not text:
                            continue
                        artifact = Artifact.from_text(
                            document_id=document_id,
                            kind=ArtifactKind.TEXT_BLOCK,
                            text=text,
                            derivation=derivation,
                            locator=locator,
                            parent_artifact_id=page_artifact.artifact_id,
                            metadata={
                                "block_number": block_number,
                                "source": "native_pdf_text",
                            },
                        )
                        artifacts.append(artifact)
                        records.append(
                            SearchRecord.from_artifact(
                                artifact,
                                representation_kind=RepresentationKind.NATIVE_TEXT,
                                search_text=text,
                                grounding_basis=GroundingBasis.DIRECT_TEXT,
                                retrieval_schema_id=schema.schema_id,
                                metadata={
                                    "block_number": block_number,
                                    "page": page_number,
                                },
                            )
                        )
                        native_text_count += 1
                    elif block_type == 1 and locator.bbox is not None:
                        clip = pymupdf.Rect(*block["bbox"])
                        figure_content = page.get_pixmap(
                            clip=clip,
                            dpi=self.render_dpi,
                            alpha=False,
                        ).tobytes("png")
                        artifacts.append(
                            _asset_artifact(
                                document_id=document_id,
                                kind=ArtifactKind.FIGURE,
                                content=figure_content,
                                derivation=derivation,
                                locator=locator,
                                asset_store=asset_store,
                                parent_artifact_id=page_artifact.artifact_id,
                                metadata={
                                    "block_number": block_number,
                                    "source": "embedded_image_block",
                                },
                            )
                        )

                try:
                    tables = tuple(page.find_tables().tables)
                except Exception as exc:
                    warnings.append(
                        f"table_detection_error:page={page_number}:{type(exc).__name__}"
                    )
                    tables = ()
                for table_number, table in enumerate(tables):
                    markdown = str(table.to_markdown()).strip()
                    if not markdown:
                        continue
                    artifact = Artifact.from_text(
                        document_id=document_id,
                        kind=ArtifactKind.TABLE,
                        text=markdown,
                        derivation=derivation,
                        locator=self._locator(
                            page_number,
                            table.bbox,
                            page_width=page_width,
                            page_height=page_height,
                        ),
                        parent_artifact_id=page_artifact.artifact_id,
                        metadata={"table_number": table_number},
                    )
                    artifacts.append(artifact)
                    records.append(
                        SearchRecord.from_artifact(
                            artifact,
                            representation_kind=RepresentationKind.TABLE_MARKDOWN,
                            search_text=markdown,
                            grounding_basis=GroundingBasis.TABLE_EXTRACTION,
                            retrieval_schema_id=schema.schema_id,
                            metadata={
                                "page": page_number,
                                "table_number": table_number,
                            },
                        )
                    )

                if native_text_count == 0:
                    warnings.append(f"no_native_text:page={page_number}")

        return ParseRun(
            parser_name=self.name,
            parser_revision=self.revision,
            document_id=document_id,
            retrieval_schema_id=schema.schema_id,
            selected_pages=pages,
            artifacts=tuple(artifacts),
            records=tuple(records),
            elapsed_seconds=time.perf_counter() - started,
            peak_memory_mib=_peak_memory_mib(),
            warnings=tuple(warnings),
        )
