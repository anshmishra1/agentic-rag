from types import SimpleNamespace

import pytest

from agentic_rag.retrieval.v3_reader import retrieve_v3_hybrid_with_scores


class _ManifestStore:
    def __init__(self, manifest) -> None:
        self.manifest = manifest
        self.calls = []

    def get_ready_manifest(self, **kwargs):
        self.calls.append(kwargs)
        return self.manifest


class _Index:
    def __init__(self) -> None:
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        if "sparse_vector" in kwargs:
            matches = [
                {
                    "id": "record-shared",
                    "score": 4.0,
                    "metadata": {
                        "text": "shared evidence",
                        "artifact_id": "artifact-shared",
                        "record_id": "record-shared",
                        "artifact_kind": "figure",
                        "grounding_basis": "vision_description",
                        "page_label": "2",
                        "document_id": "doc-1",
                    },
                }
            ]
        else:
            matches = [
                {
                    "id": "record-dense",
                    "score": 0.8,
                    "metadata": {
                        "text": "dense evidence",
                        "artifact_id": "artifact-dense",
                        "record_id": "record-dense",
                        "artifact_kind": "text_block",
                        "grounding_basis": "direct_text",
                        "page_label": "1",
                        "document_id": "doc-1",
                    },
                },
                {
                    "id": "record-shared",
                    "score": 0.7,
                    "metadata": {
                        "text": "shared evidence",
                        "artifact_id": "artifact-shared",
                        "record_id": "record-shared",
                        "artifact_kind": "figure",
                        "grounding_basis": "vision_description",
                        "page_label": "2",
                        "document_id": "doc-1",
                    },
                },
            ]
        return {"matches": matches}


def _manifest():
    return {
        "document_id": "doc-1",
        "retrieval_schema_id": "multimodal_v3:schema",
        "bm25_params": "serialized-bm25",
    }


def test_v3_reader_scopes_both_queries_and_preserves_artifact_provenance() -> None:
    store = _ManifestStore(_manifest())
    index = _Index()
    sparse_encoder = object()

    results, diagnostics = retrieve_v3_hybrid_with_scores(
        query="explain the diagram",
        document_id="doc-1",
        index_name="agentic-rag-hybrid-v3",
        index=index,
        manifest_store=store,
        query_builder=lambda query, encoder: SimpleNamespace(
            dense=[0.1, 0.2, 0.3],
            sparse={"indices": [7], "values": [1.0]},
        ),
        sparse_loader=lambda payload: sparse_encoder,
        candidate_k=5,
        rrf_k=60,
        alpha=0.5,
    )

    assert store.calls == [{
        "index_name": "agentic-rag-hybrid-v3",
        "document_id": "doc-1",
    }]
    assert len(index.calls) == 2
    expected_filter = {
        "$and": [
            {"document_id": {"$eq": "doc-1"}},
            {"retrieval_schema_id": {"$eq": "multimodal_v3:schema"}},
            {"type": {"$eq": "content"}},
        ]
    }
    assert all(call["filter"] == expected_filter for call in index.calls)
    assert index.calls[1]["sparse_vector"] == {"indices": [7], "values": [1.0]}
    assert results[0][0].metadata["artifact_id"] == "artifact-shared"
    assert results[0][0].metadata["grounding_basis"] == "vision_description"
    assert diagnostics["rrf_count"] == 2
    assert diagnostics["rrf"][0]["record_id"] == "record-shared"
    assert diagnostics["rrf"][0]["dense_score"] == pytest.approx(0.7)
    assert diagnostics["rrf"][0]["sparse_score"] == pytest.approx(4.0)
    assert diagnostics["rrf"][0]["grounding_basis"] == "vision_description"


def test_v3_reader_fails_before_vector_access_when_manifest_is_not_ready() -> None:
    index = _Index()

    with pytest.raises(ValueError, match="not ready"):
        retrieve_v3_hybrid_with_scores(
            query="question",
            document_id="doc-1",
            index_name="agentic-rag-hybrid-v3",
            index=index,
            manifest_store=_ManifestStore(None),
            query_builder=lambda *_: None,
            sparse_loader=lambda *_: None,
            candidate_k=5,
            rrf_k=60,
            alpha=0.5,
        )

    assert index.calls == []


def test_v3_reader_can_use_dense_only_when_sparse_state_is_absent() -> None:
    manifest = _manifest()
    manifest["bm25_params"] = None
    index = _Index()

    results, diagnostics = retrieve_v3_hybrid_with_scores(
        query="question",
        document_id="doc-1",
        index_name="agentic-rag-hybrid-v3",
        index=index,
        manifest_store=_ManifestStore(manifest),
        query_builder=lambda query, encoder: SimpleNamespace(
            dense=[0.1, 0.2, 0.3], sparse=None
        ),
        sparse_loader=lambda *_: pytest.fail("sparse state should not load"),
        candidate_k=5,
        rrf_k=60,
        alpha=0.5,
    )

    assert len(index.calls) == 1
    assert len(results) == 2
    assert diagnostics["sparse_count"] == 0


def test_v3_reader_adds_artifact_lane_without_weakening_document_scope() -> None:
    manifest = _manifest()
    manifest["bm25_params"] = None
    store = _ManifestStore(manifest)
    index = _Index()

    _, diagnostics = retrieve_v3_hybrid_with_scores(
        query="explain the figure",
        document_id="doc-1",
        index_name="agentic-rag-hybrid-v3",
        index=index,
        manifest_store=store,
        query_builder=lambda query, encoder: SimpleNamespace(
            dense=[0.1, 0.2, 0.3], sparse=None
        ),
        sparse_loader=lambda *_: pytest.fail("sparse state should not load"),
        candidate_k=5,
        rrf_k=60,
        alpha=0.5,
        artifact_kinds=frozenset({"figure", "image"}),
    )

    assert index.calls[0]["filter"] == {
        "$and": [
            {"document_id": {"$eq": "doc-1"}},
            {"retrieval_schema_id": {"$eq": "multimodal_v3:schema"}},
            {"type": {"$eq": "content"}},
            {"artifact_kind": {"$in": ["figure", "image"]}},
        ]
    }
    assert diagnostics["artifact_kinds"] == ["figure", "image"]
