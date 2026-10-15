"""PyMuPDF4LLM plus PyMuPDF Layout adapter for V3 evidence extraction."""

from __future__ import annotations

import importlib.metadata
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
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


_PACKAGE_NAMES = ("PyMuPDF", "pymupdf4llm", "pymupdf-layout", "rapidocr")
EXPECTED_PACKAGE_VERSIONS = MappingProxyType(
    {
        "PyMuPDF": "1.28.2",
        "pymupdf4llm": "1.28.2",
        "pymupdf-layout": "1.28.2",
        "rapidocr": "3.9.2",
    }
)
EXPECTED_REVISION = ";".join(
    f"{name}={EXPECTED_PACKAGE_VERSIONS[name]}" for name in _PACKAGE_NAMES
)
_IGNORED_LAYOUT_CLASSES = frozenset({"page-header", "page-footer"})
_TEXT_LAYOUT_CLASSES = frozenset(
    {"caption", "footnote", "list-item", "section-header", "text", "title"}
)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        for name in _PACKAGE_NAMES:
            versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ParserDependencyUnavailable(
            "PyMuPDF4LLM, PyMuPDF Layout, and RapidOCR must be provisioned in "
            "the application runtime"
        ) from exc
    return versions


def _pymupdf() -> Any:
    try:
        import pymupdf
    except ImportError as exc:
        raise ParserDependencyUnavailable(
            "PyMuPDF imports failed inside the application runtime"
        ) from exc
    return pymupdf


def _pymupdf4llm() -> Any:
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise ParserDependencyUnavailable(
            "PyMuPDF4LLM imports failed inside the application runtime"
        ) from exc
    return pymupdf4llm


def _rapidocr_function() -> Any:
    try:
        from pymupdf4llm.ocr.rapidocr_api import exec_ocr
    except ImportError as exc:
        raise ParserDependencyUnavailable(
            "PyMuPDF4LLM could not load the pinned RapidOCR integration"
        ) from exc
    return exec_ocr


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


def _box_bbox(box: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        values = tuple(float(box[key]) for key in ("x0", "y0", "x1", "y1"))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    if values[0] >= values[2] or values[1] >= values[3]:
        return None
    return values


def _text_from_lines(lines: Any) -> str:
    if not isinstance(lines, list):
        return ""
    output: list[str] = []
    for line in lines:
        if not isinstance(line, Mapping):
            continue
        spans = line.get("spans")
        if not isinstance(spans, list):
            continue
        text = _text_from_spans(spans)
        if text:
            output.append(text)
    return _normalize_search_text("\n".join(output))


def _text_from_spans(spans: list[Any]) -> str:
    output = ""
    previous_x1: float | None = None
    previous_size: float | None = None
    for span in spans:
        if not isinstance(span, Mapping):
            continue
        text = str(span.get("text", ""))
        if not text:
            continue
        bbox = span.get("bbox")
        current_x0: float | None = None
        current_x1: float | None = None
        if isinstance(bbox, Sequence) and len(bbox) == 4:
            try:
                current_x0 = float(bbox[0])
                current_x1 = float(bbox[2])
            except (TypeError, ValueError):
                current_x0 = current_x1 = None
        try:
            current_size = float(span.get("size"))
        except (TypeError, ValueError):
            current_size = None

        needs_space = False
        if output and not output[-1].isspace() and not text[0].isspace():
            if previous_x1 is not None and current_x0 is not None:
                sizes = [size for size in (previous_size, current_size) if size]
                threshold = max(0.5, 0.12 * min(sizes)) if sizes else 0.75
                needs_space = current_x0 - previous_x1 > threshold
        if needs_space:
            output += " "
        output += text
        previous_x1 = current_x1
        previous_size = current_size
    return output.strip()


def _normalize_search_text(text: str) -> str:
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalized_words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def _is_native_text(text: str, native_page_text: str) -> bool:
    words = _normalized_words(text)
    if not words:
        return False
    native_words = _normalized_words(native_page_text)
    return len(words & native_words) / len(words) >= 0.8


def _lines_are_ocr(lines: Any, pymupdf: Any) -> bool:
    if not isinstance(lines, list):
        return False
    stroked = int(pymupdf.mupdf.FZ_STEXT_STROKED)
    filled = int(pymupdf.mupdf.FZ_STEXT_FILLED)
    for line in lines:
        if not isinstance(line, Mapping):
            continue
        spans = line.get("spans")
        if not isinstance(spans, list):
            continue
        for span in spans:
            if not isinstance(span, Mapping) or not str(span.get("text", "")).strip():
                continue
            if span.get("font") == "GlyphLessFont":
                return True
            try:
                flags = int(span.get("char_flags", 0))
            except (TypeError, ValueError):
                flags = 0
            if not (flags & stroked or flags & filled):
                return True
    return False


def _table_markdown(table: Any) -> str:
    if not isinstance(table, Mapping):
        return ""
    existing = table.get("markdown")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    rows = table.get("extract")
    if not isinstance(rows, list) or not rows:
        return ""
    width = max((len(row) for row in rows if isinstance(row, list)), default=0)
    if width == 0:
        return ""

    def cell(value: Any) -> str:
        return str(value or "").replace("|", "\\|").replace("\n", " ").strip()

    normalized = [
        [cell(value) for value in row] + [""] * (width - len(row))
        for row in rows
        if isinstance(row, list)
    ]
    if not normalized:
        return ""
    header = normalized[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


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


class PyMuPDFLayoutParserAdapter:
    """Emit V3 evidence from PyMuPDF4LLM's ordered Layout JSON."""

    name = "pymupdf_layout"
    capabilities = ParserCapabilities(
        text_blocks=True,
        headings=True,
        figures=True,
        tables=True,
        formulas=True,
        conditional_ocr=True,
        page_images=True,
    )

    def __init__(
        self,
        *,
        render_dpi: int = 144,
        ocr_dpi: int = 150,
        enable_ocr: bool = True,
    ) -> None:
        if render_dpi < 72 or render_dpi > 600:
            raise ValueError("render_dpi must be between 72 and 600")
        if ocr_dpi < 72 or ocr_dpi > 600:
            raise ValueError("ocr_dpi must be between 72 and 600")
        self.render_dpi = render_dpi
        self.ocr_dpi = ocr_dpi
        self.enable_ocr = enable_ocr
        versions = _package_versions()
        if versions != EXPECTED_PACKAGE_VERSIONS:
            expected = ", ".join(
                f"{name}=={version}"
                for name, version in EXPECTED_PACKAGE_VERSIONS.items()
            )
            actual = ", ".join(
                f"{name}=={versions.get(name, 'missing')}" for name in _PACKAGE_NAMES
            )
            raise ParserDependencyUnavailable(
                "PyMuPDF Layout runtime version mismatch; "
                f"expected [{expected}], found [{actual}]"
            )
        self.versions = versions
        self.revision = EXPECTED_REVISION

    @property
    def parser_options(self) -> dict[str, Any]:
        return {
            "force_text_in_picture_regions": True,
            "ignored_layout_classes": tuple(sorted(_IGNORED_LAYOUT_CLASSES)),
            "layout_output": "json",
            "ocr": "conditional_rapidocr" if self.enable_ocr else "disabled",
            "ocr_dpi": self.ocr_dpi,
            "page_render_dpi": self.render_dpi,
            "render_html_tables": False,
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
            raise ValueError("PyMuPDF Layout adapter currently accepts PDF files only")
        if not selected_pages or tuple(sorted(set(selected_pages))) != selected_pages:
            raise ValueError("selected_pages must be sorted, unique, and nonempty")
        if any(page < 1 for page in selected_pages):
            raise ValueError("selected_pages must use one-based page numbers")
        if (
            schema.parser_name != self.name
            or schema.parser_revision != self.revision
            or dict(schema.parser_options) != self.parser_options
        ):
            raise ValueError(
                "retrieval schema does not match the PyMuPDF Layout parser runtime"
            )

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

    @staticmethod
    def _record(
        artifact: Artifact,
        *,
        text: str,
        is_ocr: bool,
        schema: RetrievalSchema,
        metadata: dict[str, Any],
    ) -> SearchRecord:
        return SearchRecord.from_artifact(
            artifact,
            representation_kind=(
                RepresentationKind.OCR_TEXT if is_ocr else RepresentationKind.NATIVE_TEXT
            ),
            search_text=text,
            grounding_basis=(
                GroundingBasis.OCR_DERIVED if is_ocr else GroundingBasis.DIRECT_TEXT
            ),
            retrieval_schema_id=schema.schema_id,
            metadata=metadata,
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
        pymupdf4llm = _pymupdf4llm()
        ocr_function = _rapidocr_function() if self.enable_ocr else None
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
            if pages[-1] > len(document):
                raise ValueError(f"selected page {pages[-1]} is outside the PDF")
            native_page_text = {
                page_number: str(
                    document.load_page(page_number - 1).get_text("text", sort=True)
                ).strip()
                for page_number in pages
            }
            # PyMuPDF4LLM inserts OCR text into the document it parses. Keep a
            # separate document open for page and region rendering so derived
            # evidence images remain pixel-faithful to the original PDF.
            with pymupdf.open(str(path)) as layout_document:
                raw = pymupdf4llm.to_json(
                    layout_document,
                    pages=[page - 1 for page in pages],
                    image_dpi=self.render_dpi,
                    ocr_dpi=self.ocr_dpi,
                    use_ocr=self.enable_ocr,
                    force_ocr=False,
                    ocr_function=ocr_function,
                    force_text=True,
                    render_html_tables=False,
                    embed_images=False,
                    write_images=False,
                    show_progress=False,
                )
            payload = json.loads(raw) if isinstance(raw, str) else raw
            layout_pages = payload.get("pages") if isinstance(payload, Mapping) else None
            if not isinstance(layout_pages, list):
                raise ValueError("PyMuPDF4LLM returned an invalid Layout JSON payload")
            page_by_number = {
                int(page["page_number"]): page
                for page in layout_pages
                if isinstance(page, Mapping) and "page_number" in page
            }

            for page_number in pages:
                layout_page = page_by_number.get(page_number)
                if layout_page is None:
                    raise ValueError(
                        f"PyMuPDF4LLM omitted requested page {page_number}"
                    )
                page = document.load_page(page_number - 1)
                page_width = float(layout_page.get("width", page.rect.width))
                page_height = float(layout_page.get("height", page.rect.height))
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

                searchable_count = 0
                boxes = layout_page.get("boxes", ())
                if not isinstance(boxes, list):
                    warnings.append(f"invalid_layout_boxes:page={page_number}")
                    boxes = []
                for box_number, box in enumerate(boxes):
                    if not isinstance(box, Mapping):
                        warnings.append(
                            f"invalid_layout_box:page={page_number}:box={box_number}"
                        )
                        continue
                    layout_class = str(box.get("boxclass", "unknown"))
                    if layout_class in _IGNORED_LAYOUT_CLASSES:
                        continue
                    bbox = _box_bbox(box)
                    locator = self._locator(
                        page_number,
                        bbox,
                        page_width=page_width,
                        page_height=page_height,
                    )
                    if bbox is None or locator.bbox is None:
                        warnings.append(
                            f"invalid_bbox:page={page_number}:box={box_number}"
                        )
                        continue
                    metadata = {
                        "box_number": box_number,
                        "layout_class": layout_class,
                        "page": page_number,
                    }
                    header_level = box.get("header_level")
                    if layout_class in {"title", "section-header"} and isinstance(
                        header_level, int
                    ):
                        metadata["header_level"] = header_level

                    if layout_class in _TEXT_LAYOUT_CLASSES:
                        lines = box.get("textlines")
                        text = _text_from_lines(lines)
                        if not text:
                            warnings.append(
                                f"empty_text_region:page={page_number}:box={box_number}"
                            )
                            continue
                        is_ocr = _lines_are_ocr(lines, pymupdf) or not _is_native_text(
                            text,
                            native_page_text[page_number],
                        )
                        artifact = Artifact.from_text(
                            document_id=document_id,
                            kind=ArtifactKind.TEXT_BLOCK,
                            text=text,
                            derivation=derivation,
                            locator=locator,
                            parent_artifact_id=page_artifact.artifact_id,
                            metadata={**metadata, "source": "layout_text"},
                        )
                        artifacts.append(artifact)
                        records.append(
                            self._record(
                                artifact,
                                text=text,
                                is_ocr=is_ocr,
                                schema=schema,
                                metadata=metadata,
                            )
                        )
                        searchable_count += 1
                        continue

                    if layout_class == "table":
                        markdown = _table_markdown(box.get("table"))
                        if not markdown:
                            warnings.append(
                                f"empty_table:page={page_number}:box={box_number}"
                            )
                            continue
                        artifact = Artifact.from_text(
                            document_id=document_id,
                            kind=ArtifactKind.TABLE,
                            text=markdown,
                            derivation=derivation,
                            locator=locator,
                            parent_artifact_id=page_artifact.artifact_id,
                            metadata=metadata,
                        )
                        artifacts.append(artifact)
                        records.append(
                            SearchRecord.from_artifact(
                                artifact,
                                representation_kind=RepresentationKind.TABLE_MARKDOWN,
                                search_text=markdown,
                                grounding_basis=GroundingBasis.TABLE_EXTRACTION,
                                retrieval_schema_id=schema.schema_id,
                                metadata=metadata,
                            )
                        )
                        searchable_count += 1
                        continue

                    if layout_class not in {"picture", "formula"}:
                        warnings.append(
                            f"unsupported_layout_class:page={page_number}:"
                            f"box={box_number}:class={layout_class}"
                        )
                        continue

                    clip = pymupdf.Rect(*bbox)
                    content = page.get_pixmap(
                        clip=clip,
                        dpi=self.render_dpi,
                        alpha=False,
                    ).tobytes("png")
                    kind = (
                        ArtifactKind.FIGURE
                        if layout_class == "picture"
                        else ArtifactKind.FORMULA
                    )
                    artifact = _asset_artifact(
                        document_id=document_id,
                        kind=kind,
                        content=content,
                        derivation=derivation,
                        locator=locator,
                        asset_store=asset_store,
                        parent_artifact_id=page_artifact.artifact_id,
                        metadata=metadata,
                    )
                    artifacts.append(artifact)

                    lines = box.get("textlines")
                    text = _text_from_lines(lines)
                    is_ocr = _lines_are_ocr(lines, pymupdf) or not _is_native_text(
                        text,
                        native_page_text[page_number],
                    )
                    if layout_class == "formula" and not text:
                        text = _normalize_search_text(
                            str(page.get_text("text", clip=clip, sort=True))
                        )
                        is_ocr = not _is_native_text(
                            text,
                            native_page_text[page_number],
                        )
                    if text:
                        records.append(
                            self._record(
                                artifact,
                                text=text,
                                is_ocr=is_ocr,
                                schema=schema,
                                metadata=metadata,
                            )
                        )
                        searchable_count += 1

                if searchable_count == 0:
                    warnings.append(f"no_searchable_text:page={page_number}")

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
