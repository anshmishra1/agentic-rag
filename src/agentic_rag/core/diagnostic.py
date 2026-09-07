from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_json(value: Any) -> Any:
    """Convert arbitrary values into JSON-serializable values."""

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _safe_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in value]

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _slugify(value: str, max_length: int = 60) -> str:
    """
    Convert a query into a human-readable filesystem-safe name.

    Example:
        "What is RAG?" -> "what_is_rag"
    """

    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")

    if not value:
        value = "query"

    return value[:max_length].rstrip("_")


class DiagnosticCollector:
    """
    Collect structured diagnostics for one HTTP query.

    The application run directory is created and owned by logging.py.
    This class creates only the query-specific directory underneath it.
    """

    def __init__(
        self,
        *,
        run_id: str,
        run_directory: str | Path,
        query: str,
        session_id: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.run_directory = Path(run_directory)
        self.query = query
        self.session_id = session_id

        self.started_at = datetime.now().astimezone()

        # ---------------------------------------------------------
        # All query diagnostics belong underneath the current run.
        # ---------------------------------------------------------

        self.queries_directory = (
            self.run_directory / "queries"
        )

        self.queries_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------
        # Assign the next human-readable query number.
        # ---------------------------------------------------------

        existing_numbers: list[int] = []

        for directory in self.queries_directory.iterdir():
            if not directory.is_dir():
                continue

            match = re.match(
                r"^(\d+)_",
                directory.name,
            )

            if match:
                existing_numbers.append(
                    int(match.group(1))
                )

        self.query_number = (
            max(existing_numbers, default=0) + 1
        )

        # ---------------------------------------------------------
        # Human-readable query directory.
        # ---------------------------------------------------------

        query_slug = _slugify(query)

        self.query_directory = (
            self.queries_directory
            / f"{self.query_number:03d}_{query_slug}"
        )

        self.query_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        # ---------------------------------------------------------
        # Events collected during this request.
        # ---------------------------------------------------------

        self.events: list[dict[str, Any]] = []

    # =============================================================
    # Query numbering
    # =============================================================

    @staticmethod
    def _next_query_number(
        queries_directory: Path,
    ) -> int:

        numbers: list[int] = []

        for directory in queries_directory.iterdir():

            if not directory.is_dir():
                continue

            match = re.match(
                r"^(\d+)_",
                directory.name,
            )

            if match:
                numbers.append(
                    int(match.group(1))
                )

        return max(numbers, default=0) + 1

    # =============================================================
    # Event collection
    # =============================================================

    def record(
        self,
        event: str,
        *,
        stage: str | None = None,
        **data: Any,
    ) -> dict[str, Any]:

        payload: dict[str, Any] = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "run_id": self.run_id,

            "query_number": self.query_number,

            "event": event,
        }

        if self.session_id is not None:
            payload["session_id"] = self.session_id

        if stage is not None:
            payload["stage"] = stage

        payload.update(
            _safe_json(data)
        )

        self.events.append(payload)

        return payload

    # =============================================================
    # Convenience methods
    # =============================================================

    def retrieval(
        self,
        event: str,
        **data: Any,
    ) -> dict[str, Any]:

        return self.record(
            event,
            stage="retrieval",
            **data,
        )

    def llm(
        self,
        event: str,
        **data: Any,
    ) -> dict[str, Any]:

        return self.record(
            event,
            stage="llm",
            **data,
        )

    def graph(
        self,
        event: str,
        **data: Any,
    ) -> dict[str, Any]:

        return self.record(
            event,
            stage="graph",
            **data,
        )

    def grounding(
        self,
        event: str,
        **data: Any,
    ) -> dict[str, Any]:

        return self.record(
            event,
            stage="grounding",
            **data,
        )

    def performance(
        self,
        event: str,
        **data: Any,
    ) -> dict[str, Any]:

        return self.record(
            event,
            stage="performance",
            **data,
        )

    # =============================================================
    # Export
    # =============================================================

    def write_jsonl(self) -> Path:

        path = (
            self.query_directory
            / "diagnostics.jsonl"
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            for event in self.events:

                file.write(
                    json.dumps(
                        event,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                    + "\n"
                )

        return path

    def write_json(
        self,
        *,
        summary: dict[str, Any] | None = None,
    ) -> Path:

        finished_at = datetime.now(
            timezone.utc
        )

        payload: dict[str, Any] = {

            "run_id": self.run_id,

            "query_number": self.query_number,

            "query": self.query,

            "session_id": self.session_id,

            "started_at": (
                self.started_at.isoformat()
            ),

            "finished_at": (
                finished_at.isoformat()
            ),

            "event_count": len(
                self.events
            ),

            "events": self.events,
        }

        if summary is not None:
            payload["summary"] = _safe_json(
                summary
            )

        path = (
            self.query_directory
            / "diagnostics.json"
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        return path

    def finalize(
        self,
        *,
        summary: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:

        jsonl_path = self.write_jsonl()

        json_path = self.write_json(
            summary=summary
        )

        return (
            jsonl_path,
            json_path,
        )