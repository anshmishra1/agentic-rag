"""Storage-neutral parser boundary for multimodal ingestion V3."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from agentic_rag.ingestion.asset_store import AssetStore
from agentic_rag.ingestion.contracts import Artifact, RetrievalSchema, SearchRecord


class ParserDependencyUnavailable(RuntimeError):
    """Raised when an optional parser runtime has not been provisioned."""


@dataclass(frozen=True)
class ParserCapabilities:
    text_blocks: bool
    headings: bool
    figures: bool
    tables: bool
    formulas: bool
    conditional_ocr: bool
    page_images: bool

    def to_dict(self) -> dict[str, bool]:
        return {
            "text_blocks": self.text_blocks,
            "headings": self.headings,
            "figures": self.figures,
            "tables": self.tables,
            "formulas": self.formulas,
            "conditional_ocr": self.conditional_ocr,
            "page_images": self.page_images,
        }


@dataclass(frozen=True)
class ParseRun:
    """One parser's normalized output for one document and page selection."""

    parser_name: str
    parser_revision: str
    document_id: str
    retrieval_schema_id: str
    selected_pages: tuple[int, ...]
    artifacts: tuple[Artifact, ...]
    records: tuple[SearchRecord, ...]
    elapsed_seconds: float
    peak_memory_mib: float
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("parser_name", "parser_revision", "document_id", "retrieval_schema_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be nonempty")
        if not self.selected_pages or any(page < 1 for page in self.selected_pages):
            raise ValueError("selected_pages must contain one-based page numbers")
        if tuple(sorted(set(self.selected_pages))) != self.selected_pages:
            raise ValueError("selected_pages must be sorted and unique")
        if self.elapsed_seconds < 0 or self.peak_memory_mib < 0:
            raise ValueError("runtime measurements cannot be negative")

        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        record_ids = [record.record_id for record in self.records]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("parse run contains duplicate artifact IDs")
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("parse run contains duplicate search-record IDs")
        artifact_by_id = {
            artifact.artifact_id: artifact for artifact in self.artifacts
        }
        artifact_id_set = set(artifact_by_id)

        for artifact in self.artifacts:
            if artifact.document_id != self.document_id:
                raise ValueError("artifact belongs to another document")
            page_start = artifact.locator.page_start
            page_end = artifact.locator.page_end or page_start
            if page_start is not None and any(
                page not in self.selected_pages
                for page in range(page_start, page_end + 1)
            ):
                raise ValueError("artifact is outside the requested page selection")
            parent_id = artifact.parent_artifact_id
            if parent_id is not None:
                if parent_id == artifact.artifact_id:
                    raise ValueError("artifact cannot be its own parent")
                parent = artifact_by_id.get(parent_id)
                if parent is None:
                    raise ValueError("artifact refers to an unavailable parent")
                parent_start = parent.locator.page_start
                parent_end = parent.locator.page_end or parent_start
                if (
                    page_start is not None
                    and parent_start is not None
                    and not (
                        parent_start <= page_start
                        and (parent_end or parent_start) >= (page_end or page_start)
                    )
                ):
                    raise ValueError("artifact lies outside its parent page range")
        for record in self.records:
            if record.document_id != self.document_id:
                raise ValueError("search record belongs to another document")
            if record.artifact_id not in artifact_id_set:
                raise ValueError("search record refers to an unavailable artifact")
            if record.retrieval_schema_id != self.retrieval_schema_id:
                raise ValueError("search record uses another retrieval schema")

        for artifact in self.artifacts:
            visited: set[str] = set()
            current = artifact
            while current.parent_artifact_id is not None:
                if current.artifact_id in visited:
                    raise ValueError("artifact parent relationships contain a cycle")
                visited.add(current.artifact_id)
                current = artifact_by_id[current.parent_artifact_id]

    def safe_summary(self) -> dict:
        """Return measurements and identities without extracted passage text."""
        artifact_counts = Counter(artifact.kind.value for artifact in self.artifacts)
        representation_counts = Counter(
            record.representation_kind.value for record in self.records
        )
        pages_with_artifacts = sorted(
            {
                artifact.locator.page_start
                for artifact in self.artifacts
                if artifact.locator.page_start is not None
            }
        )
        return {
            "parser": {"name": self.parser_name, "revision": self.parser_revision},
            "document_id": self.document_id,
            "retrieval_schema_id": self.retrieval_schema_id,
            "selected_pages": list(self.selected_pages),
            "pages_with_artifacts": pages_with_artifacts,
            "artifact_counts": dict(sorted(artifact_counts.items())),
            "representation_counts": dict(sorted(representation_counts.items())),
            "artifact_ids": sorted(artifact.artifact_id for artifact in self.artifacts),
            "record_ids": sorted(record.record_id for record in self.records),
            "elapsed_seconds": self.elapsed_seconds,
            "peak_memory_mib": self.peak_memory_mib,
            "warnings": list(self.warnings),
        }


class ParserAdapter(Protocol):
    name: str
    revision: str
    capabilities: ParserCapabilities

    @property
    def parser_options(self) -> dict[str, object]: ...

    def parse(
        self,
        path: Path,
        *,
        document_id: str,
        selected_pages: Sequence[int],
        schema: RetrievalSchema,
        asset_store: AssetStore,
    ) -> ParseRun: ...
