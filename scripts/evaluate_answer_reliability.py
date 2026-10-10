"""Evaluate sanitized live results against the v2 reliability contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag.evaluation.answer_reliability import (
    evaluate_acceptance,
    render_acceptance_summary,
)


def _json_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check one live run against reviewed v2 answer invariants."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval_set/answer_reliability_v2_acceptance.json"),
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = _json_rows(args.results)
    report = evaluate_acceptance(manifest, results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema_version": 1,
        "acceptance_name": manifest.get("name"),
        "acceptance_source_run": manifest.get("source_run"),
        "result_rows": len(results),
        "answer_or_passage_text_persisted": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "query_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in report["cases"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_dir / "summary.md").write_text(
        render_acceptance_summary(report),
        encoding="utf-8",
    )

    print(render_acceptance_summary(report), end="")
    return 0 if report["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
