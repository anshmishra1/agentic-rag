"""Prepare document-scoped retrieval pairs for human relevance review.

This module uses only local question metadata and PDF file bytes. Generated
labels are candidates, never calibration ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DOCUMENT_ID = re.compile(r"[0-9a-f]{64}\Z")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_candidates(
    questions: list[dict[str, Any]], pdf_dir: Path, *, per_document: int = 4
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample source questions and pair each with its source and another PDF.

    The negative pair is only a review candidate: different documents can
    discuss the same topic. Both labels stay pending until evidence is checked.
    Paired rows share a split so one question cannot leak into both sets.
    """
    if per_document < 2:
        raise ValueError("per_document must be at least 2 for a holdout")
    if not isinstance(questions, list):
        raise ValueError("questions must be a JSON list")

    by_source: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    excluded = {"cross_document": 0, "missing_pdf": 0}
    for source_row, row in enumerate(questions, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Question {source_row} must be an object")
        source = row.get("source_document")
        query = row.get("user_input")
        if not isinstance(source, str) or not isinstance(query, str) or not query.strip():
            raise ValueError(f"Question {source_row} needs source_document and user_input")
        if "+" in source:
            excluded["cross_document"] += 1
        elif Path(source).name != source:
            raise ValueError(f"Question {source_row} has an unsafe source_document")
        elif not (pdf_dir / source).is_file():
            excluded["missing_pdf"] += 1
        else:
            by_source[source].append((source_row, row))

    sources = sorted(by_source)
    if len(sources) < 2:
        raise ValueError("At least two available source PDFs are needed for negative candidates")
    document_ids = {source: _file_digest(pdf_dir / source) for source in sources}
    candidates: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        rows = by_source[source]
        sample_size = min(per_document, len(rows))
        if sample_size < 2:
            raise ValueError(f"{source} has fewer than two usable questions")
        # Evenly spaced source rows avoid selecting only the first chapter.
        selected = [rows[index * len(rows) // sample_size] for index in range(sample_size)]
        other_source = sources[(source_index + 1) % len(sources)]
        for position, (source_row, row) in enumerate(selected):
            split = "holdout" if position == sample_size - 1 else "development"
            for candidate_label, target in ((True, source), (False, other_source)):
                candidates.append(
                    {
                        "case_id": f"q{source_row:03d}-{'source' if candidate_label else 'other'}",
                        "source_row": source_row,
                        "query": row["user_input"],
                        "reference": row.get("reference", ""),
                        "source_document": source,
                        "source_pages": row.get("source_pages", ""),
                        "target_document": target,
                        "document_id": document_ids[target],
                        "candidate_should_match": candidate_label,
                        "split": split,
                        "review_status": "pending",
                        "review_notes": "",
                        "relevant_chunk_ids": [],
                    }
                )

    report = {
        "candidate_pairs": len(candidates),
        "sampled_questions": len(candidates) // 2,
        "development_pairs": sum(row["split"] == "development" for row in candidates),
        "holdout_pairs": sum(row["split"] == "holdout" for row in candidates),
        "source_documents": sources,
        "excluded_questions": excluded,
        "verified_pairs": 0,
    }
    return candidates, report


def export_reviewed_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Export only explicitly verified labels for the retrieval collector."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("Review input must be a nonempty JSON list")
    exported = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or row.get("review_status") != "verified":
            raise ValueError(f"Pair {index} has not been verified")
        if not isinstance(row.get("should_match"), bool):
            raise ValueError(f"Pair {index} needs a reviewed boolean should_match")
        if not isinstance(row.get("review_notes"), str) or not row["review_notes"].strip():
            raise ValueError(f"Pair {index} needs evidence in review_notes")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in seen_ids:
            raise ValueError(f"Pair {index} needs a unique case_id")
        if not isinstance(row.get("query"), str) or not row["query"].strip():
            raise ValueError(f"Pair {index} needs a nonempty query")
        if not isinstance(row.get("document_id"), str) or not DOCUMENT_ID.fullmatch(row["document_id"]):
            raise ValueError(f"Pair {index} needs a 64-character document_id")
        if row.get("split") not in {"development", "holdout"}:
            raise ValueError(f"Pair {index} needs a development or holdout split")
        if not isinstance(row.get("relevant_chunk_ids"), list) or any(
            not isinstance(value, str) or not value for value in row["relevant_chunk_ids"]
        ):
            raise ValueError(f"Pair {index} needs a list of relevant_chunk_ids")
        seen_ids.add(case_id)
        exported.append(
            {
                "case_id": case_id,
                "query": row["query"],
                "document_id": row["document_id"],
                "should_match": row["should_match"],
                "split": row["split"],
                "source_document": row["source_document"],
                "target_document": row["target_document"],
                "source_pages": row["source_pages"],
                "relevant_chunk_ids": row["relevant_chunk_ids"],
            }
        )
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("eval_set/ragas_eval_set_80.json"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("PDF"))
    parser.add_argument("--output", type=Path, default=Path("eval_set/retrieval_label_candidates.json"))
    parser.add_argument("--per-document", type=int, default=4)
    parser.add_argument("--export-reviewed", type=Path, help="read reviewed candidates instead of preparing them")
    args = parser.parse_args()

    if args.export_reviewed:
        if args.output.resolve() == args.export_reviewed.resolve():
            parser.error("--output must differ from --export-reviewed to preserve review notes")
        rows = json.loads(args.export_reviewed.read_text(encoding="utf-8"))
        output = export_reviewed_pairs(rows)
        report = {"verified_pairs": len(output)}
    else:
        questions = json.loads(args.questions.read_text(encoding="utf-8"))
        output, report = prepare_candidates(questions, args.pdf_dir, per_document=args.per_document)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
