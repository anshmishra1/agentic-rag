import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_rag.evaluation.chunking_pilot import (
    build_catalog,
    export_reviewed_pairs,
    stable_chunk_id,
    validate_manifest,
    validate_runtime_schema,
)


def _manifest(tmp_path: Path) -> dict:
    documents = []
    for key in ("a", "b"):
        content = f"PDF-{key}".encode()
        filename = f"{key}.pdf"
        (tmp_path / filename).write_bytes(content)
        documents.append(
            {
                "document_key": key,
                "filename": filename,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )
    return {
        "schema_version": 1,
        "pilot_id": "pilot",
        "index_name": "agentic-rag-hybrid-v2",
        "retrieval_schema": {
            "embedding_model": "model",
            "embedding_model_revision": "revision",
            "chunk_size_tokens": 240,
            "chunk_overlap_tokens": 40,
            "chunking_strategy": "token_window_v1",
            "vector_id_contract": "contract",
        },
        "documents": documents,
        "questions": [
            {
                "question_id": f"{key}-{split}",
                "query": "What is in this document?",
                "source_document_key": key,
                "split": split,
                "review_status": "pending",
                "source_pages": [],
                "relevant_chunk_ids": [],
                "review_notes": "",
            }
            for key in ("a", "b")
            for split in ("development", "holdout")
        ],
    }


def test_manifest_requires_review_evidence_and_both_splits(tmp_path):
    manifest = _manifest(tmp_path)
    validate_manifest(manifest)

    manifest["questions"][0]["review_status"] = "verified"
    with pytest.raises(ValueError, match="pages, chunk IDs, and review notes"):
        validate_manifest(manifest)

    manifest = _manifest(tmp_path)
    manifest["questions"] = [row for row in manifest["questions"] if row["split"] == "development"]
    with pytest.raises(ValueError, match="development and holdout"):
        validate_manifest(manifest)


def test_runtime_schema_must_match_manifest(tmp_path):
    manifest = _manifest(tmp_path)
    validate_runtime_schema(
        manifest,
        embedding_model="model",
        embedding_model_revision="revision",
        chunk_size_tokens=240,
        chunk_overlap_tokens=40,
    )
    with pytest.raises(ValueError, match="chunk_size_tokens"):
        validate_runtime_schema(
            manifest,
            embedding_model="model",
            embedding_model_revision="revision",
            chunk_size_tokens=128,
            chunk_overlap_tokens=40,
        )


def test_catalog_hashes_files_and_uses_production_chunk_id_shape(tmp_path):
    manifest = _manifest(tmp_path)

    def load_pdf(path):
        return [SimpleNamespace(page_content=f"Page for {path.stem}", metadata={"page": 0, "page_label": "1"})]

    def chunk_documents(pages, *, document_id, filename, strategy):
        assert strategy == "token_window_v1"
        return [SimpleNamespace(page_content=pages[0].page_content, metadata={"page": 0, "page_label": "1"})]

    catalog = build_catalog(
        manifest,
        tmp_path,
        load_pdf=load_pdf,
        chunk_documents=chunk_documents,
        strategy="token_window_v1",
    )

    assert len(catalog["chunks"]) == 2
    first = catalog["chunks"][0]
    assert first["chunk_id"] == stable_chunk_id(manifest["documents"][0]["sha256"], "Page for a")
    assert first["page_label"] == "1"
    assert first["text"] == "Page for a"
    assert first["chunking_strategy"] == "token_window_v1"

    (tmp_path / "a.pdf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        build_catalog(
            manifest,
            tmp_path,
            load_pdf=load_pdf,
            chunk_documents=chunk_documents,
            strategy="token_window_v1",
        )


def test_reviewed_export_rejects_unknown_chunk_and_matches_existing_pair_schema(tmp_path):
    manifest = _manifest(tmp_path)
    document = manifest["documents"][0]
    chunk_id = stable_chunk_id(document["sha256"], "Page for a")
    for question in manifest["questions"]:
        question["review_status"] = "verified"
        question["source_pages"] = ["1"]
        question["review_notes"] = "Read the cited passage and confirmed direct support."
        question["relevant_chunk_ids"] = [
            chunk_id if question["source_document_key"] == "a" else stable_chunk_id(manifest["documents"][1]["sha256"], "Page for b")
        ]
    catalog = {
        "chunks": [
            {"document_key": "a", "chunk_id": chunk_id},
            {"document_key": "b", "chunk_id": stable_chunk_id(manifest["documents"][1]["sha256"], "Page for b")},
        ]
    }

    pairs = export_reviewed_pairs(manifest, catalog)
    assert len(pairs) == 4
    assert set(pairs[0]) == {
        "case_id",
        "query",
        "document_id",
        "should_match",
        "split",
        "source_document",
        "target_document",
        "source_pages",
        "relevant_chunk_ids",
    }
    assert pairs[0]["should_match"] is True

    manifest["questions"][0]["relevant_chunk_ids"] = ["missing"]
    with pytest.raises(ValueError, match="absent from the catalog"):
        export_reviewed_pairs(manifest, catalog)


def test_repository_pilot_manifest_is_verified_and_has_three_balanced_documents():
    root = Path(__file__).parents[2]
    manifest = json.loads((root / "eval_set/chunking_pilot_manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)

    assert len(manifest["documents"]) == 3
    assert len(manifest["questions"]) == 12
    assert sum(row["split"] == "holdout" for row in manifest["questions"]) == 3
    assert all(row["review_status"] == "verified" for row in manifest["questions"])
    assert all(row["source_pages"] and row["relevant_chunk_ids"] for row in manifest["questions"])

    catalog = {
        "chunks": [
            {"document_key": row["source_document_key"], "chunk_id": chunk_id}
            for row in manifest["questions"]
            for chunk_id in row["relevant_chunk_ids"]
        ]
    }
    exported = json.loads((root / "eval_set/chunking_pilot_reviewed_pairs.json").read_text(encoding="utf-8"))
    assert export_reviewed_pairs(manifest, catalog) == exported


def test_v2_reviewed_pairs_keep_the_control_questions_and_documents():
    root = Path(__file__).parents[2]
    v1 = json.loads(
        (root / "eval_set/chunking_pilot_reviewed_pairs.json").read_text(
            encoding="utf-8"
        )
    )
    v2 = json.loads(
        (root / "eval_set/chunking_pilot_v2_reviewed_pairs.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(v1) == len(v2) == 12
    for control, treatment in zip(v1, v2, strict=True):
        for field in (
            "case_id",
            "query",
            "document_id",
            "should_match",
            "split",
            "source_document",
            "target_document",
            "source_pages",
        ):
            assert treatment[field] == control[field]
        assert treatment["relevant_chunk_ids"]
        assert set(treatment["relevant_chunk_ids"]).isdisjoint(
            control["relevant_chunk_ids"]
        )
        assert all(
            chunk_id.startswith(f"{treatment['document_id']}-content-")
            for chunk_id in treatment["relevant_chunk_ids"]
        )


def test_pilot_cli_imports_settings_from_an_isolated_working_directory():
    root = Path(__file__).parents[2]
    script = (root / "scripts/prepare_chunking_pilot.py").read_text(encoding="utf-8")

    assert script.index("os.chdir(isolated_cwd)") < script.index("from agentic_rag.config import settings")
