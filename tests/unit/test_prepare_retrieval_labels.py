import hashlib
import json
from pathlib import Path

import pytest

from agentic_rag.evaluation.prepare_retrieval_labels import (
    export_reviewed_pairs,
    prepare_candidates,
)


def test_candidates_are_pending_document_scoped_pairs_with_shared_split(tmp_path):
    for name in ("a.pdf", "b.pdf"):
        (tmp_path / name).write_bytes(name.encode())
    questions = [
        {
            "user_input": f"Question {source} {number}",
            "reference": "Answer",
            "source_document": source,
            "source_pages": str(number),
        }
        for source in ("a.pdf", "b.pdf")
        for number in range(1, 4)
    ]
    questions.append({"user_input": "Both?", "source_document": "a.pdf + b.pdf"})
    questions.append({"user_input": "Missing?", "source_document": "missing.pdf"})

    rows, report = prepare_candidates(questions, tmp_path, per_document=2)

    assert report["candidate_pairs"] == 8
    assert report["excluded_questions"] == {"cross_document": 1, "missing_pdf": 1}
    assert report["development_pairs"] == report["holdout_pairs"] == 4
    by_question = {}
    for row in rows:
        by_question.setdefault(row["source_row"], []).append(row)
        assert row["review_status"] == "pending"
        assert row["document_id"] == hashlib.sha256(row["target_document"].encode()).hexdigest()
        assert row["candidate_should_match"] == (row["source_document"] == row["target_document"])
    assert all(len(pair) == 2 and pair[0]["split"] == pair[1]["split"] for pair in by_question.values())
    with pytest.raises(ValueError, match="not been verified"):
        export_reviewed_pairs(rows)


def test_export_requires_explicit_review_and_evidence():
    row = {
        "case_id": "q001-source",
        "query": "Question",
        "document_id": "a" * 64,
        "candidate_should_match": True,
        "should_match": False,
        "review_status": "verified",
        "review_notes": "Checked the complete short document; no answer is present.",
        "split": "holdout",
        "source_document": "a.pdf",
        "target_document": "b.pdf",
        "source_pages": "2",
        "relevant_chunk_ids": [],
    }
    exported = export_reviewed_pairs([row])
    assert exported[0]["should_match"] is False
    assert "candidate_should_match" not in exported[0]
    with pytest.raises(ValueError, match="evidence"):
        export_reviewed_pairs([{**row, "review_notes": ""}])
    with pytest.raises(ValueError, match="unique case_id"):
        export_reviewed_pairs([row, row])


def test_reviewed_pilot_is_exportable_and_keeps_corrected_overlap_labels():
    root = Path(__file__).parents[2]
    reviewed = json.loads((root / "eval_set/retrieval_reviewed_pilot.json").read_text(encoding="utf-8"))
    exported = json.loads((root / "eval_set/retrieval_reviewed_pairs.json").read_text(encoding="utf-8"))

    assert len(reviewed) == len(exported) == 46
    assert export_reviewed_pairs(reviewed) == exported
    assert {row["split"] for row in exported} == {"development", "holdout"}
    assert {row["should_match"] for row in exported} == {True, False}
    assert {row["case_id"] for row in exported if row["should_match"] and row["target_document"] != row["source_document"]} == {
        "q033-other",
        "q038-other",
    }
