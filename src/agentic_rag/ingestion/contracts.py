"""Versioned contracts for multimodal ingestion and retrieval evidence.

V3 separates a source artifact from the text representations used to find it.
The contracts in this module are storage neutral: binary data belongs in an
asset store, searchable text belongs in retrieval records, and answer evidence
links the two without copying either object into durable graph state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CITATION_LABEL_PATTERN = re.compile(r"^S[1-9][0-9]*$")


class ArtifactKind(StrEnum):
    TEXT_BLOCK = "text_block"
    FIGURE = "figure"
    TABLE = "table"
    FORMULA = "formula"
    PAGE_IMAGE = "page_image"
    STANDALONE_IMAGE = "standalone_image"
    AUDIO_SEGMENT = "audio_segment"


class RepresentationKind(StrEnum):
    NATIVE_TEXT = "native_text"
    OCR_TEXT = "ocr_text"
    CAPTION = "caption"
    TABLE_MARKDOWN = "table_markdown"
    FORMULA_LATEX = "formula_latex"
    TRANSCRIPT = "transcript"
    PAGE_SUMMARY = "page_summary"


class GroundingBasis(StrEnum):
    DIRECT_TEXT = "direct_text"
    OCR_DERIVED = "ocr_derived"
    TABLE_EXTRACTION = "table_extraction"
    FORMULA_EXTRACTION = "formula_extraction"
    VISION_DESCRIPTION = "vision_description"
    AUDIO_TRANSCRIPT = "audio_transcript"


JsonValue = str | int | float | bool | None | Sequence["JsonValue"] | Mapping[str, "JsonValue"]


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _freeze_json(value: JsonValue) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON contract values must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON contract object keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"Unsupported JSON contract value: {type(value).__name__}")


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _plain_json(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _stable_id(namespace: str, value: Mapping[str, Any]) -> str:
    digest = sha256_text(f"{namespace}\n{_canonical_json(value)}")
    return f"{namespace}_{digest}"


def _require_nonempty(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _validate_sha256(name: str, value: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class SourceLocator:
    """Location of an artifact in its original source.

    PDF page numbers are one based. Bounding boxes use normalized coordinates
    ``(left, top, right, bottom)`` so they remain independent of render DPI.
    Audio offsets are milliseconds from the start of the source.
    """

    page_start: int | None = None
    page_end: int | None = None
    page_label: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    start_ms: int | None = None
    end_ms: int | None = None

    def __post_init__(self) -> None:
        if self.page_start is not None and self.page_start < 1:
            raise ValueError("page_start must be one based")
        if self.page_end is not None and self.page_end < 1:
            raise ValueError("page_end must be one based")
        if self.page_end is not None and self.page_start is None:
            raise ValueError("page_end requires page_start")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must be greater than or equal to page_start")
        if self.page_label is not None and not self.page_label.strip():
            raise ValueError("page_label must be nonempty when provided")

        if self.bbox is not None:
            if self.page_start is None:
                raise ValueError("bbox requires a source page")
            if self.page_end not in (None, self.page_start):
                raise ValueError("bbox cannot span multiple pages")
            left, top, right, bottom = self.bbox
            if not all(math.isfinite(value) for value in self.bbox):
                raise ValueError("bbox coordinates must be finite")
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise ValueError("bbox coordinates must be normalized and ordered")

        has_start = self.start_ms is not None
        has_end = self.end_ms is not None
        if has_start != has_end:
            raise ValueError("audio locators require both start_ms and end_ms")
        if has_start:
            assert self.start_ms is not None and self.end_ms is not None
            if self.start_ms < 0 or self.end_ms <= self.start_ms:
                raise ValueError("audio offsets must define a positive interval")
            if self.page_start is not None or self.bbox is not None:
                raise ValueError("a locator cannot mix page and audio coordinates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_start": self.page_start,
            "page_end": self.page_end,
            "page_label": self.page_label,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }


@dataclass(frozen=True)
class Derivation:
    """Identifies the parser or model that produced an artifact."""

    producer: str
    revision: str
    parameters: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty("producer", self.producer)
        _require_nonempty("revision", self.revision)
        object.__setattr__(self, "parameters", _freeze_json(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "revision": self.revision,
            "parameters": _plain_json(self.parameters),
        }


@dataclass(frozen=True)
class RetrievalSchema:
    """Complete identity of data that may coexist in one retrieval index."""

    parser_name: str
    parser_revision: str
    chunking_strategy: str
    text_embedding_model: str
    text_embedding_revision: str
    text_embedding_dimension: int
    parser_options: Mapping[str, JsonValue] = field(default_factory=dict)
    schema_family: str = "multimodal_v3"
    representation_contract: str = "artifact_search_record_v1"
    sparse_encoder: str = "bm25_document_v1"
    vector_id_contract: str = "artifact_representation_hash_v1"
    visual_embedding_model: str | None = None
    visual_embedding_revision: str | None = None
    visual_embedding_dimension: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "schema_family",
            "parser_name",
            "parser_revision",
            "chunking_strategy",
            "representation_contract",
            "text_embedding_model",
            "text_embedding_revision",
            "sparse_encoder",
            "vector_id_contract",
        ):
            _require_nonempty(name, str(getattr(self, name)))
        if self.text_embedding_dimension < 1:
            raise ValueError("text_embedding_dimension must be positive")
        visual_values = (
            self.visual_embedding_model,
            self.visual_embedding_revision,
            self.visual_embedding_dimension,
        )
        if any(value is not None for value in visual_values) and not all(
            value is not None for value in visual_values
        ):
            raise ValueError("visual embedding identity must be supplied as a complete set")
        if self.visual_embedding_dimension is not None and self.visual_embedding_dimension < 1:
            raise ValueError("visual_embedding_dimension must be positive")
        object.__setattr__(self, "parser_options", _freeze_json(self.parser_options))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_family": self.schema_family,
            "parser": {
                "name": self.parser_name,
                "revision": self.parser_revision,
                "options": _plain_json(self.parser_options),
            },
            "chunking_strategy": self.chunking_strategy,
            "representation_contract": self.representation_contract,
            "text_embedding": {
                "model": self.text_embedding_model,
                "revision": self.text_embedding_revision,
                "dimension": self.text_embedding_dimension,
            },
            "sparse_encoder": self.sparse_encoder,
            "vector_id_contract": self.vector_id_contract,
            "visual_embedding": (
                {
                    "model": self.visual_embedding_model,
                    "revision": self.visual_embedding_revision,
                    "dimension": self.visual_embedding_dimension,
                }
                if self.visual_embedding_model is not None
                else None
            ),
        }

    @property
    def fingerprint(self) -> str:
        return sha256_text(_canonical_json(self.to_dict()))

    @property
    def schema_id(self) -> str:
        return f"{self.schema_family}:{self.fingerprint[:16]}"


@dataclass(frozen=True)
class Artifact:
    """Original source unit or a deterministic derivative of one."""

    document_id: str
    kind: ArtifactKind
    content_sha256: str
    derivation: Derivation
    locator: SourceLocator = field(default_factory=SourceLocator)
    asset_key: str | None = None
    native_text: str | None = None
    parent_artifact_id: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        _require_nonempty("document_id", self.document_id)
        _validate_sha256("content_sha256", self.content_sha256)
        if self.asset_key is not None:
            _require_nonempty("asset_key", self.asset_key)
        if self.native_text is not None and not self.native_text.strip():
            raise ValueError("native_text must be nonempty when provided")
        if self.native_text is None and self.asset_key is None:
            raise ValueError("an artifact requires native_text or an asset_key")
        if (
            self.native_text is not None
            and self.asset_key is None
            and sha256_text(self.native_text) != self.content_sha256
        ):
            raise ValueError("text-only artifact digest must match native_text")
        if self.parent_artifact_id is not None:
            _require_nonempty("parent_artifact_id", self.parent_artifact_id)
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))
        object.__setattr__(
            self,
            "artifact_id",
            _stable_id(
                "artifact",
                {
                    "document_id": self.document_id,
                    "kind": self.kind.value,
                    "content_sha256": self.content_sha256,
                    "locator": self.locator.to_dict(),
                    "parent_artifact_id": self.parent_artifact_id,
                },
            ),
        )

    @classmethod
    def from_text(
        cls,
        *,
        document_id: str,
        kind: ArtifactKind,
        text: str,
        derivation: Derivation,
        locator: SourceLocator | None = None,
        parent_artifact_id: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> "Artifact":
        return cls(
            document_id=document_id,
            kind=kind,
            content_sha256=sha256_text(text),
            derivation=derivation,
            locator=locator or SourceLocator(),
            native_text=text,
            parent_artifact_id=parent_artifact_id,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "document_id": self.document_id,
            "kind": self.kind.value,
            "content_sha256": self.content_sha256,
            "derivation": self.derivation.to_dict(),
            "locator": self.locator.to_dict(),
            "asset_key": self.asset_key,
            "native_text": self.native_text,
            "parent_artifact_id": self.parent_artifact_id,
            "metadata": _plain_json(self.metadata),
        }


@dataclass(frozen=True)
class SearchRecord:
    """Searchable representation that points back to exactly one artifact."""

    document_id: str
    artifact_id: str
    representation_kind: RepresentationKind
    search_text: str
    grounding_basis: GroundingBasis
    retrieval_schema_id: str
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("document_id", "artifact_id", "search_text", "retrieval_schema_id"):
            _require_nonempty(name, str(getattr(self, name)))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))
        object.__setattr__(
            self,
            "record_id",
            _stable_id(
                "record",
                {
                    "document_id": self.document_id,
                    "artifact_id": self.artifact_id,
                    "representation_kind": self.representation_kind.value,
                    "search_text_sha256": sha256_text(self.search_text),
                    "grounding_basis": self.grounding_basis.value,
                    "retrieval_schema_id": self.retrieval_schema_id,
                },
            ),
        )

    @classmethod
    def from_artifact(
        cls,
        artifact: Artifact,
        *,
        representation_kind: RepresentationKind,
        search_text: str,
        grounding_basis: GroundingBasis,
        retrieval_schema_id: str,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> "SearchRecord":
        return cls(
            document_id=artifact.document_id,
            artifact_id=artifact.artifact_id,
            representation_kind=representation_kind,
            search_text=search_text,
            grounding_basis=grounding_basis,
            retrieval_schema_id=retrieval_schema_id,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "document_id": self.document_id,
            "artifact_id": self.artifact_id,
            "representation_kind": self.representation_kind.value,
            "search_text": self.search_text,
            "grounding_basis": self.grounding_basis.value,
            "retrieval_schema_id": self.retrieval_schema_id,
            "metadata": _plain_json(self.metadata),
        }


@dataclass(frozen=True)
class AssetRef:
    asset_key: str
    media_type: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _require_nonempty("asset_key", self.asset_key)
        _require_nonempty("media_type", self.media_type)
        if self.byte_count < 1:
            raise ValueError("byte_count must be positive")
        _validate_sha256("sha256", self.sha256)


@dataclass(frozen=True)
class EvidenceRef:
    """Answer-facing provenance without passage text or binary payloads."""

    label: str
    document_id: str
    artifact_id: str
    record_id: str
    artifact_kind: ArtifactKind
    grounding_basis: GroundingBasis
    source_name: str
    description: str
    locator: SourceLocator = field(default_factory=SourceLocator)
    asset_url: str | None = None

    def __post_init__(self) -> None:
        if not _CITATION_LABEL_PATTERN.fullmatch(self.label):
            raise ValueError("label must use the canonical S<number> form")
        for name in (
            "document_id",
            "artifact_id",
            "record_id",
            "source_name",
            "description",
        ):
            _require_nonempty(name, str(getattr(self, name)))
        if self.asset_url is not None:
            _require_nonempty("asset_url", self.asset_url)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "document_id": self.document_id,
            "artifact_id": self.artifact_id,
            "record_id": self.record_id,
            "kind": self.artifact_kind.value,
            "grounding_basis": self.grounding_basis.value,
            "source_name": self.source_name,
            "description": self.description,
            "locator": self.locator.to_dict(),
            "asset_url": self.asset_url,
        }
