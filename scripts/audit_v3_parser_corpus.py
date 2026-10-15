"""Audit the fixed V3 parser corpus without running a parser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag.evaluation.parser_bakeoff import (
    audit_parser_corpus,
    load_parser_cases,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval_set/v3_parser_bakeoff_cases.json"),
    )
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    cases = load_parser_cases(args.manifest)
    result = audit_parser_corpus(
        cases,
        {
            "research_articles": args.research_root,
            "repository": args.repository_root,
        },
    )
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
