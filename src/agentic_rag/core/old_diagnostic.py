"""Per-run structured diagnostics for Agentic RAG.

This module deliberately stays outside LangGraph state.

Responsibilities:
- collect meaningful diagnostic events for one application run
- write events as JSONL
- write a compact final JSON summary
- provide a small API that graph nodes can use without knowing
  how diagnostics are persisted

It does NOT:
- store diagnostics in RAGState
- manage PerformanceTracker
- configure Python logging
- create a global/singleton collector
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class DiagnosticCollector:
    """Collect structured diagnostics for a single RAG run."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        output_dir: str | Path = "logs/runs",
    ) -> None:
        self.run_id = run_id or uuid4().hex
        self.started_at = datetime.now(timezone.utc)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Event collection
    # ------------------------------------------------------------------

    def record(
        self,
        event: str,
        *,
        stage: str | None = None,
        **data: Any,
    ) -> dict[str, Any]:
        """Record one structured diagnostic event."""

        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
        }

        if stage is not None:
            payload["stage"] = stage

        payload.update(data)

        self.events.append(payload)

        return payload

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def write_jsonl(self) -> Path:
        """Write the complete event stream for this run."""

        path = self.output_dir / f"{self.run_id}.jsonl"

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
        """Write a compact JSON representation of the run."""

        finished_at = datetime.now(timezone.utc)

        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "event_count": len(self.events),
            "events": self.events,
        }

        if summary:
            payload["summary"] = summary

        path = self.output_dir / f"{self.run_id}.json"

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
        """Write both the JSONL event stream and JSON run summary."""

        jsonl_path = self.write_jsonl()
        json_path = self.write_json(summary=summary)

        return jsonl_path, json_path