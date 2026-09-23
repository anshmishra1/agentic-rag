"""
scripts/run_ragas_eval.py

Runs the eval set against the live FastAPI app and scores it with RAGAS.
Cross-document questions are skipped - the app has no multi-document
retrieval mode, so those can only be answered wrong for architectural
reasons unrelated to pipeline quality (see session notes).
"""

import json
import uuid
import time 
import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

API_URL = "http://127.0.0.1:8000"
EVAL_SET_PATH = "eval_set\\ragas_eval_set_80.json"
REQUEST_DELAY_SECONDS = 3.0  # tune based on which provider's limit you hit most


def load_document_map() -> dict[str, str]:
    """filename -> document_id, from the live registry."""
    docs = requests.get(f"{API_URL}/documents", timeout=30).json()
    return {d["filename"]: d["document_id"] for d in docs}


def run_eval():
    eval_set = json.load(open(EVAL_SET_PATH))
    doc_map = load_document_map()

    runnable = [
        row for row in eval_set
        if "+" not in row["source_document"]          # skip cross-document
        and row["source_document"] in doc_map          # skip un-ingested docs
    ]

    skipped = len(eval_set) - len(runnable)
    print(f"Running {len(runnable)} / {len(eval_set)} questions "
          f"({skipped} skipped: cross-document or not ingested)")

    session_id = str(uuid.uuid4())  # fresh thread per eval run
    rows = []

    for i, item in enumerate(runnable, 1):
        document_id = doc_map[item["source_document"]]

        try:
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "question": item["user_input"],
                    "session_id": session_id,
                    "document_id": document_id,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            rows.append({
                "user_input": item["user_input"],
                "response": data["answer"],
                "retrieved_contexts": data["contexts"],
                "reference": item["reference"],
            })

            print(f"[{i}/{len(runnable)}] {item['user_input'][:60]}... "
                  f"grounded={data['grounded']}")

            time.sleep(REQUEST_DELAY_SECONDS)

        except requests.exceptions.RequestException as exc:
            print(f"[{i}/{len(runnable)}] FAILED: {item['user_input'][:60]}... "
                  f"error={exc}")
            continue

    dataset = Dataset.from_list(rows)

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\n=== RAGAS RESULTS ===")
    print(results)

    results.to_pandas().to_csv("ragas_results.csv", index=False)
    print("\nSaved per-question breakdown to ragas_results.csv")


if __name__ == "__main__":
    run_eval()