"""Asset storage boundary for V3 source artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, ContextManager, Protocol

from agentic_rag.ingestion.contracts import AssetRef, sha256_bytes


_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")


class AssetStore(Protocol):
    def put(
        self,
        *,
        document_id: str,
        artifact_id: str,
        content: bytes,
        media_type: str,
    ) -> AssetRef: ...

    def open(self, asset_key: str) -> ContextManager[BinaryIO]: ...

    def delete_document(self, document_id: str) -> int: ...


def _safe_segment(name: str, value: str) -> str:
    if not _SAFE_SEGMENT.fullmatch(value):
        raise ValueError(f"{name} contains unsupported path characters")
    return value


def _document_storage_key(document_id: str) -> str:
    document_id = _safe_segment("document_id", document_id)
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    return f"d-{digest[:32]}"


class LocalAssetStore:
    """Content-addressed development asset store rooted in one directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _path_for_key(self, asset_key: str) -> Path:
        parts = asset_key.split("/")
        if len(parts) != 2:
            raise ValueError("asset_key must contain document storage key and digest")
        document_key, digest = (
            _safe_segment("asset key segment", part) for part in parts
        )
        if not re.fullmatch(r"d-[0-9a-f]{32}", document_key):
            raise ValueError("asset document key must be a truncated SHA-256 digest")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("asset key digest must be lowercase SHA-256")
        path = (self.root / document_key / digest).resolve()
        if self.root not in path.parents:
            raise ValueError("asset_key resolves outside the asset store")
        return path

    def put(
        self,
        *,
        document_id: str,
        artifact_id: str,
        content: bytes,
        media_type: str,
    ) -> AssetRef:
        _safe_segment("document_id", document_id)
        _safe_segment("artifact_id", artifact_id)
        if not isinstance(content, bytes) or not content:
            raise ValueError("content must be nonempty bytes")
        if not media_type or not media_type.strip():
            raise ValueError("media_type must be nonempty")

        digest = sha256_bytes(content)
        asset_key = f"{_document_storage_key(document_id)}/{digest}"
        target = self._path_for_key(asset_key)
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            if sha256_bytes(target.read_bytes()) != digest:
                raise OSError(f"Existing asset is corrupt: {asset_key}")
        else:
            handle = tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=".asset-",
                delete=False,
            )
            temporary = Path(handle.name)
            try:
                with handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)

        return AssetRef(
            asset_key=asset_key,
            media_type=media_type.strip(),
            byte_count=len(content),
            sha256=digest,
        )

    @contextmanager
    def open(self, asset_key: str) -> Iterator[BinaryIO]:
        path = self._path_for_key(asset_key)
        with path.open("rb") as stream:
            yield stream

    def delete_document(self, document_id: str) -> int:
        document_key = _document_storage_key(document_id)
        directory = (self.root / document_key).resolve()
        if self.root not in directory.parents:
            raise ValueError("document directory resolves outside the asset store")
        if not directory.exists():
            return 0
        file_count = sum(1 for path in directory.rglob("*") if path.is_file())
        shutil.rmtree(directory)
        return file_count
