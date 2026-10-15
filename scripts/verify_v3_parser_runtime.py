"""Fail fast when the packaged V3 parser runtime differs from its contract."""

from __future__ import annotations

import json

from agentic_rag.ingestion.parser_runtime import (
    build_v3_parser,
    safe_runtime_summary,
)


def main() -> None:
    parser = build_v3_parser()
    print(json.dumps(safe_runtime_summary(parser), sort_keys=True))


if __name__ == "__main__":
    main()
