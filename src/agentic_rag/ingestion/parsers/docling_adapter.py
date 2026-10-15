"""Optional Docling adapter for the multimodal V3 parser bakeoff."""

from __future__ import annotations

import importlib.metadata
import io
import math
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from pypdf import PdfReader

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
from agentic_rag.local_inference import local_inference_slot


_VERSION_PACKAGES = (
    "docling",
    "docling-core",
    "docling-parse",
    "docling-ibm-models",
    "rapidocr",
)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _VERSION_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ParserDependencyUnavailable(
                "Docling parser dependencies are not installed; use the "
                "disposable Docling evaluation image"
            ) from exc
    return versions


def _revision(versions: dict[str, str]) -> str:
    return ";".join(f"{name}={versions[name]}" for name in _VERSION_PACKAGES)


def _label_value(item: Any) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label))


def _normalized_words(text: str) -> set[str]:
    return {word.casefold() for word in text.split() if word.strip()}


def _is_native_text(text: str, native_page_text: str) -> bool:
    words = _normalized_words(text)
    if not words:
        return False
    native_words = _normalized_words(native_page_text)
    return len(words & native_words) / len(words) >= 0.8


def _source_locator(item: Any, document: Any) -> SourceLocator | None:
    provenance = tuple(getattr(item, "prov", ()) or ())
    if not provenance:
        return None
    page_numbers = sorted({int(entry.page_no) for entry in provenance})
    page_start, page_end = page_numbers[0], page_numbers[-1]
    bbox: tuple[float, float, float, float] | None = None
    if len(provenance) == 1:
        page = document.pages.get(page_start)
        if page is not None:
            top_left = provenance[0].bbox.to_top_left_origin(page.size.height)
            normalized = top_left.normalized(page.size)
            values = (
                float(normalized.l),
                float(normalized.t),
                float(normalized.r),
                float(normalized.b),
            )
            if all(math.isfinite(value) for value in values):
                left, top, right, bottom = (
                    min(1.0, max(0.0, value)) for value in values
                )
                if left < right and top < bottom:
                    bbox = (left, top, right, bottom)
    return SourceLocator(
        page_start=page_start,
        page_end=page_end,
        page_label=str(page_start) if page_start == page_end else f"{page_start}-{page_end}",
        bbox=bbox,
    )


def _image_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    content = buffer.getvalue()
    if not content:
        raise ValueError("Docling returned an empty image")
    return content


def _asset_artifact(
    *,
    document_id: str,
    kind: ArtifactKind,
    content: bytes,
    derivation: Derivation,
    locator: SourceLocator,
    asset_store: AssetStore,
    native_text: str | None = None,
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
        native_text=native_text,
        parent_artifact_id=parent_artifact_id,
        metadata=metadata or {},
    )
    asset = asset_store.put(
        document_id=document_id,
        artifact_id=draft.artifact_id,
        content=content,
        media_type="image/png",
    )
    return replace(draft, asset_key=asset.asset_key)


def _peak_memory_mib() -> float:
    # Linux reports ru_maxrss in KiB. The evaluation image is Linux-only.
    try:
        import resource
    except ImportError:  # pragma: no cover - adapter execution uses Linux Docker
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class DoclingParserAdapter:
    """Convert selected PDF pages into V3 artifacts without live services."""

    name = "docling"
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
        artifacts_path: str | Path,
        ocr_language: str = "en",
        formula_enrichment: bool = True,
        accelerator_device: str = "cpu",
    ) -> None:
        if accelerator_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("accelerator_device must be auto, cpu, or cuda")
        self.artifacts_path = Path(artifacts_path)
        self.ocr_language = ocr_language
        self.formula_enrichment = formula_enrichment
        self.accelerator_device = accelerator_device
        self.versions = _package_versions()
        self.revision = _revision(self.versions)
        self._converter_instance: Any | None = None

    @property
    def parser_options(self) -> dict[str, Any]:
        """Return every parser option that can affect emitted evidence."""

        return {
            "ocr_engine": "rapidocr-onnxruntime",
            "ocr_language": self.ocr_language,
            "table_mode": "accurate",
            "formula_enrichment": self.formula_enrichment,
            "accelerator_device": self.accelerator_device,
            "remote_services": False,
        }

    def _converter(self) -> Any:
        if self._converter_instance is None:
            self._converter_instance = self._create_converter()
        return self._converter_instance

    def _create_converter(self) -> Any:
        try:
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                OcrMode,
                PdfPipelineOptions,
                RapidOcrOptions,
                TableFormerMode,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise ParserDependencyUnavailable(
                "Docling imports failed inside the parser evaluation runtime"
            ) from exc

        if self.accelerator_device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable to PyTorch")

        options = PdfPipelineOptions(
            artifacts_path=self.artifacts_path,
            accelerator_options=AcceleratorOptions(device=self.accelerator_device),
            do_ocr=True,
            do_table_structure=True,
            do_formula_enrichment=self.formula_enrichment,
            do_picture_classification=False,
            generate_page_images=True,
            generate_picture_images=True,
            enable_remote_services=False,
            allow_external_plugins=False,
            ocr_options=RapidOcrOptions(
                backend="onnxruntime",
                lang=[self.ocr_language],
                mode=OcrMode.DEFAULT,
            ),
        )
        options.table_structure_options.mode = TableFormerMode.ACCURATE
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            }
        )

    def _validate_request(
        self,
        path: Path,
        selected_pages: tuple[int, ...],
        schema: RetrievalSchema,
    ) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.casefold() != ".pdf":
            raise ValueError("Docling parser adapter currently accepts PDF files only")
        if not selected_pages or tuple(sorted(set(selected_pages))) != selected_pages:
            raise ValueError("selected_pages must be sorted, unique, and nonempty")
        if any(page < 1 for page in selected_pages):
            raise ValueError("selected_pages must use one-based page numbers")
        if (
            schema.parser_name != self.name
            or schema.parser_revision != self.revision
            or dict(schema.parser_options) != self.parser_options
        ):
            raise ValueError("retrieval schema does not match the Docling parser runtime")

    @staticmethod
    def _native_page_text(reader: PdfReader, page_number: int) -> str:
        try:
            return (reader.pages[page_number - 1].extract_text() or "").strip()
        except IndexError as exc:
            raise ValueError(f"selected page {page_number} is outside the PDF") from exc

    @staticmethod
    def _captions(item: Any, document: Any) -> str:
        parts: list[str] = []
        for reference in getattr(item, "captions", ()) or ():
            resolved = reference.resolve(document)
            text = str(getattr(resolved, "text", "")).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)

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
        converter = self._converter()
        derivation = Derivation(
            producer=self.name,
            revision=self.revision,
            parameters=self.parser_options,
        )
        artifacts: list[Artifact] = []
        records: list[SearchRecord] = []
        warnings: list[str] = []
        started = time.perf_counter()
        native_reader = PdfReader(path)

        for requested_page in pages:
            with local_inference_slot():
                result = converter.convert(
                    path,
                    page_range=(requested_page, requested_page),
                )
            status = str(getattr(result.status, "value", result.status))
            if status not in {"success", "partial_success"}:
                raise RuntimeError(
                    f"Docling conversion failed on page {requested_page}: {status}"
                )
            if status == "partial_success":
                warnings.append(f"partial_success:page={requested_page}")
            for error in result.errors:
                component = str(getattr(error.component_type, "value", error.component_type))
                category = str(getattr(error.category, "value", error.category))
                warnings.append(
                    f"docling_error:component={component}:category={category}:page={error.page_no}"
                )

            document = result.document
            page_item = document.pages.get(requested_page)
            page_artifact: Artifact | None = None
            if page_item is not None and page_item.image is not None:
                page_artifact = _asset_artifact(
                    document_id=document_id,
                    kind=ArtifactKind.PAGE_IMAGE,
                    content=_image_bytes(page_item.image.pil_image),
                    derivation=derivation,
                    locator=SourceLocator(
                        page_start=requested_page,
                        page_end=requested_page,
                        page_label=str(requested_page),
                    ),
                    asset_store=asset_store,
                    metadata={"dpi": page_item.image.dpi},
                )
                artifacts.append(page_artifact)
            else:
                warnings.append(f"missing_page_image:page={requested_page}")

            native_page_text = self._native_page_text(native_reader, requested_page)
            attached_captions = {
                reference.cref
                for item, _ in document.iterate_items()
                for reference in (getattr(item, "captions", ()) or ())
            }

            for item, _level in document.iterate_items():
                locator = _source_locator(item, document)
                if locator is None or locator.page_start != requested_page:
                    continue
                label = _label_value(item)
                parent_id = page_artifact.artifact_id if page_artifact is not None else None
                item_ref = str(getattr(item, "self_ref", ""))
                metadata = {"docling_label": label, "docling_ref": item_ref}

                if label in {"picture", "chart"}:
                    caption = self._captions(item, document)
                    image = item.get_image(document)
                    if image is None and not caption:
                        warnings.append(f"empty_picture:page={requested_page}:ref={item_ref}")
                        continue
                    if image is not None:
                        artifact = _asset_artifact(
                            document_id=document_id,
                            kind=ArtifactKind.FIGURE,
                            content=_image_bytes(image),
                            derivation=derivation,
                            locator=locator,
                            asset_store=asset_store,
                            native_text=caption or None,
                            parent_artifact_id=parent_id,
                            metadata=metadata,
                        )
                    else:
                        artifact = Artifact.from_text(
                            document_id=document_id,
                            kind=ArtifactKind.FIGURE,
                            text=caption,
                            derivation=derivation,
                            locator=locator,
                            parent_artifact_id=parent_id,
                            metadata=metadata,
                        )
                    artifacts.append(artifact)
                    if caption:
                        direct = _is_native_text(caption, native_page_text)
                        records.append(
                            SearchRecord.from_artifact(
                                artifact,
                                representation_kind=RepresentationKind.CAPTION,
                                search_text=caption,
                                grounding_basis=(
                                    GroundingBasis.DIRECT_TEXT
                                    if direct
                                    else GroundingBasis.OCR_DERIVED
                                ),
                                retrieval_schema_id=schema.schema_id,
                            )
                        )
                    continue

                if label in {"table", "document_index"}:
                    table_text = item.export_to_markdown(document).strip()
                    caption = self._captions(item, document)
                    search_text = "\n\n".join(part for part in (caption, table_text) if part)
                    if not search_text:
                        warnings.append(f"empty_table:page={requested_page}:ref={item_ref}")
                        continue
                    artifact = Artifact.from_text(
                        document_id=document_id,
                        kind=ArtifactKind.TABLE,
                        text=search_text,
                        derivation=derivation,
                        locator=locator,
                        parent_artifact_id=parent_id,
                        metadata=metadata,
                    )
                    artifacts.append(artifact)
                    records.append(
                        SearchRecord.from_artifact(
                            artifact,
                            representation_kind=RepresentationKind.TABLE_MARKDOWN,
                            search_text=search_text,
                            grounding_basis=GroundingBasis.TABLE_EXTRACTION,
                            retrieval_schema_id=schema.schema_id,
                        )
                    )
                    continue

                text = str(getattr(item, "text", "")).strip()
                if not text or (label == "caption" and item_ref in attached_captions):
                    continue
                if label == "formula":
                    kind = ArtifactKind.FORMULA
                    representation = RepresentationKind.FORMULA_LATEX
                    basis = GroundingBasis.FORMULA_EXTRACTION
                else:
                    kind = ArtifactKind.TEXT_BLOCK
                    direct = _is_native_text(text, native_page_text)
                    representation = (
                        RepresentationKind.NATIVE_TEXT
                        if direct
                        else RepresentationKind.OCR_TEXT
                    )
                    basis = (
                        GroundingBasis.DIRECT_TEXT
                        if direct
                        else GroundingBasis.OCR_DERIVED
                    )
                artifact = Artifact.from_text(
                    document_id=document_id,
                    kind=kind,
                    text=text,
                    derivation=derivation,
                    locator=locator,
                    parent_artifact_id=parent_id,
                    metadata=metadata,
                )
                artifacts.append(artifact)
                records.append(
                    SearchRecord.from_artifact(
                        artifact,
                        representation_kind=representation,
                        search_text=text,
                        grounding_basis=basis,
                        retrieval_schema_id=schema.schema_id,
                    )
                )

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
