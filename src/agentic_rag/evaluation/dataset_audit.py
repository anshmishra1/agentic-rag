"""Audit the existing question set before using it for retrieval calibration.

This module has no application or provider imports. It inspects question
metadata and local file availability; it never reads PDF contents.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = ("user_input", "reference", "source_document", "source_pages")
NUMERIC_PAGES = re.compile(r"\d+(?:\s*[-,;]\s*\d+)*\Z")


def audit_question_set(rows: list[dict[str, Any]], pdf_dir: Path) -> dict[str, Any]:
    """Return coverage and readiness facts without inventing relevance labels."""
    if not isinstance(rows, list):
        raise ValueError("Evaluation set must be a JSON list")

    single_documents: Counter[str] = Counter()
    cross_document_count = 0
    numeric_page_references = 0
    missing_references = 0
    relevance_labeled_questions = 0
    questions: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be an object")
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Row {index} is missing: {', '.join(missing)}")

        question = str(row["user_input"]).strip()
        source = str(row["source_document"]).strip()
        if not question or not source:
            raise ValueError(f"Row {index} needs a question and source document")

        questions[question.casefold()] += 1
        if not str(row["reference"]).strip():
            missing_references += 1
        if "+" in source:
            cross_document_count += 1
            continue

        if Path(source).name != source:
            raise ValueError(f"Row {index} source_document must be a filename")
        single_documents[source] += 1
        if NUMERIC_PAGES.fullmatch(str(row["source_pages"]).strip()):
            numeric_page_references += 1
        labels = row.get("relevant_chunk_ids")
        if labels is not None and (
            not isinstance(labels, list)
            or any(not isinstance(label, str) or not label for label in labels)
        ):
            raise ValueError(f"Row {index} relevant_chunk_ids must be a list of IDs")
        if labels:
            relevance_labeled_questions += 1

    available = sorted(name for name in single_documents if (pdf_dir / name).is_file())
    unavailable = sorted(set(single_documents) - set(available))
    single_count = sum(single_documents.values())

    return {
        "total_questions": len(rows),
        "single_document_questions": single_count,
        "cross_document_questions": cross_document_count,
        "questions_by_document": dict(sorted(single_documents.items())),
        "available_documents": available,
        "missing_documents": unavailable,
        "numeric_page_references": numeric_page_references,
        "ambiguous_page_references": single_count - numeric_page_references,
        "missing_answer_references": missing_references,
        "duplicate_question_texts": sum(count - 1 for count in questions.values() if count > 1),
        "relevance_labeled_questions": relevance_labeled_questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=Path("eval_set/ragas_eval_set_80.json"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("PDF"))
    args = parser.parse_args()

    rows = json.loads(args.eval_set.read_text(encoding="utf-8"))
    report = audit_question_set(rows, args.pdf_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
