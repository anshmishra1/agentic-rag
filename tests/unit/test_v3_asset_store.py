from pathlib import Path

import pytest

from agentic_rag.ingestion.asset_store import LocalAssetStore


def test_local_asset_store_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    store = LocalAssetStore(tmp_path / "assets")
    content = b"synthetic figure bytes"

    first = store.put(
        document_id="doc-123",
        artifact_id="artifact-123",
        content=content,
        media_type="image/png",
    )
    second = store.put(
        document_id="doc-123",
        artifact_id="artifact-123",
        content=content,
        media_type="image/png",
    )

    assert first == second
    assert len(first.asset_key) < 120
    with store.open(first.asset_key) as stream:
        assert stream.read() == content
    assert len(list((tmp_path / "assets").rglob(first.sha256))) == 1


def test_local_asset_store_deduplicates_equal_bytes_within_a_document(
    tmp_path: Path,
) -> None:
    store = LocalAssetStore(tmp_path / "assets")

    first = store.put(
        document_id="d" * 64,
        artifact_id="artifact-" + "a" * 64,
        content=b"shared rendering",
        media_type="image/png",
    )
    second = store.put(
        document_id="d" * 64,
        artifact_id="artifact-" + "b" * 64,
        content=b"shared rendering",
        media_type="image/png",
    )

    assert first == second
    assert len(first.asset_key.split("/")) == 2
    assert len(list((tmp_path / "assets").rglob(first.sha256))) == 1


def test_local_asset_store_rejects_path_traversal(tmp_path: Path) -> None:
    store = LocalAssetStore(tmp_path / "assets")

    with pytest.raises(ValueError, match="unsupported path characters"):
        store.put(
            document_id="../outside",
            artifact_id="artifact-123",
            content=b"payload",
            media_type="application/octet-stream",
        )
    with pytest.raises(ValueError):
        with store.open("../outside/artifact/digest"):
            pass


def test_delete_document_removes_only_that_documents_assets(tmp_path: Path) -> None:
    store = LocalAssetStore(tmp_path / "assets")
    first = store.put(
        document_id="doc-one",
        artifact_id="artifact-one",
        content=b"first",
        media_type="image/png",
    )
    second = store.put(
        document_id="doc-two",
        artifact_id="artifact-two",
        content=b"second",
        media_type="image/png",
    )

    assert store.delete_document("doc-one") == 1
    assert store.delete_document("doc-one") == 0
    with pytest.raises(FileNotFoundError):
        with store.open(first.asset_key):
            pass
    with store.open(second.asset_key) as stream:
        assert stream.read() == b"second"
