"""Analyze labeled retrieval scores, with optional explicit live collection.

Input is a JSON list of objects with ``query``, ``document_id``, and boolean
``should_match``. Offline analysis also requires ``top_score`` in each row.
``--live`` runs the same retrieval node as the API to collect top scores from
already-ingested documents; it queries Pinecone and PostgreSQL, but does not
invoke an LLM or change the index. Never use an unlabeled score sample as a
threshold recommendation.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from agentic_rag.evaluation.score_calibration import analyze_scored_rows


DOCUMENT_ID = re.compile(r"[0-9a-f]{64}\Z")


def summarize_retrieval_result(result: dict[str, Any], document_id: str, chunk_id_fn) -> dict[str, Any]:
    """Save policy inputs and ranked chunk identities, without document text."""
    docs = result["documents"]
    scores = result["retrieval_scores"]
    if len(docs) != len(scores):
        raise ValueError("Retrieval documents and scores have different lengths")
    candidates = []
    for doc, score in zip(docs, scores):
        metadata = doc.metadata
        candidates.append(
            {
                "chunk_id": chunk_id_fn(
                    metadata.get("document_id") or document_id,
                    metadata.get("type") or "content",
                    doc.page_content,
                ),
                "score": float(score),
                "type": metadata.get("type") or "content",
                "page": metadata.get("page_label", metadata.get("page")),
            }
        )
    return {
        "top_score": float(result["retrieval_top_score"]),
        "second_score": float(result["retrieval_second_score"]),
        "mean_score": float(result["retrieval_mean_score"]),
        "top_to_mean_ratio": float(result["retrieval_top_to_mean_ratio"]),
        "gap_ratio": float(result["retrieval_gap_ratio"]),
        "overview_top_score": result["retrieval_overview_top_score"],
        "content_top_score": result["retrieval_content_top_score"],
        "top_doc_type": candidates[0]["type"] if candidates else None,
        "ranked_candidates": candidates,
    }


def collect_live_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the API's retrieval node without generation or durable trace data."""
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or type(row.get("should_match")) is not bool:
            raise ValueError(f"Row {index} needs a boolean should_match label")
        query = row.get("query")
        document_id = row.get("document_id")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Row {index} needs a nonempty query")
        if not isinstance(document_id, str) or not DOCUMENT_ID.fullmatch(document_id):
            raise ValueError(f"Row {index} needs a 64-character document_id")

    from agentic_rag.core.timing import (
        PerformanceTracker,
        reset_current_tracker,
        set_current_tracker,
    )
    from agentic_rag.graph.nodes import retrieve
    from agentic_rag.retrieval.vectorstore import _stable_chunk_id

    scored = []
    for row in rows:
        query = row["query"]
        document_id = row["document_id"]
        tracker_token = set_current_tracker(PerformanceTracker())
        try:
            # The retrieval node prints document previews. Do not emit source
            # text or query contents in a calibration report.
            with redirect_stdout(io.StringIO()):
                result = retrieve({"question": query, "document_id": document_id})
        finally:
            reset_current_tracker(tracker_token)
        scored.append({**row, **summarize_retrieval_result(result, document_id, _stable_chunk_id)})
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON labeled score rows or query/document pairs")
    parser.add_argument("--live", action="store_true", help="query PostgreSQL and Pinecone for current scores")
    parser.add_argument("--scores-output", type=Path, help="save collected rows for offline review")
    parser.add_argument("--minimum-per-class", type=int, default=5)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        parser.error("input must be a JSON list")
    if args.live:
        rows = collect_live_scores(rows)
    report = analyze_scored_rows(rows, minimum_per_class=args.minimum_per_class)

    if args.scores_output:
        args.scores_output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
