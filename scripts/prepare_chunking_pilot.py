"""Build or validate the offline three-document chunking pilot."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from agentic_rag.evaluation.chunking_pilot import (
    build_catalog,
    export_reviewed_pairs,
    read_json,
    validate_manifest,
    validate_runtime_schema,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("eval_set/chunking_pilot_manifest.json"))
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument(
        "--catalog-output",
        type=Path,
        default=Path("logs/evaluation/chunking-pilot-v1/chunk_catalog.json"),
        help="ignored local JSON containing full passage text",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        help="optional reviewed retrieval pairs; all questions must be verified",
    )
    parser.add_argument(
        "--strategy",
        choices=("token_window_v1", "structure_aware_v2"),
        help="override the manifest strategy for a comparison catalog",
    )
    args = parser.parse_args()

    # Settings normally support a project-local .env. This offline preparation
    # command needs no secrets, so import settings from an empty working
    # directory after making every user path absolute.
    manifest_path = args.manifest.resolve()
    pdf_dir = args.pdf_dir.resolve()
    catalog_output = args.catalog_output.resolve()
    pairs_output = args.pairs_output.resolve() if args.pairs_output else None
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="chunking-pilot-") as isolated_cwd:
        os.chdir(isolated_cwd)
        try:
            from agentic_rag.config import settings
            from agentic_rag.ingestion.chunking import chunk_documents, pdf_extraction_mode
            from agentic_rag.ingestion.loaders import load_pdf
            from agentic_rag.model_identity import canonical_model_name
        finally:
            os.chdir(original_cwd)

    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    strategy = args.strategy or manifest["retrieval_schema"]["chunking_strategy"]
    validate_runtime_schema(
        manifest,
        embedding_model=canonical_model_name(settings.embedding_model),
        embedding_model_revision=settings.embedding_model_revision,
        chunk_size_tokens=settings.chunk_size,
        chunk_overlap_tokens=settings.chunk_overlap,
    )
    catalog = build_catalog(
        manifest,
        pdf_dir,
        load_pdf=lambda path: load_pdf(path, extraction_mode=pdf_extraction_mode(strategy)),
        chunk_documents=chunk_documents,
        strategy=strategy,
    )
    write_json(catalog_output, catalog)

    report = {
        "pilot_id": manifest["pilot_id"],
        "chunking_strategy": strategy,
        "documents": len(catalog["documents"]),
        "pages": sum(document["page_count"] for document in catalog["documents"]),
        "content_chunks": len(catalog["chunks"]),
        "pending_questions": sum(question["review_status"] == "pending" for question in manifest["questions"]),
        "verified_questions": sum(question["review_status"] == "verified" for question in manifest["questions"]),
        "catalog_output": str(catalog_output),
    }
    if pairs_output:
        if strategy != manifest["retrieval_schema"]["chunking_strategy"]:
            parser.error("--pairs-output requires the reviewed manifest strategy")
        pairs = export_reviewed_pairs(manifest, catalog)
        write_json(pairs_output, pairs)
        report["pairs_output"] = str(pairs_output)
        report["reviewed_pairs"] = len(pairs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
