"""Request-local, content-free accounting for outbound LLM calls."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any


_current_counts: ContextVar[dict[str, Any] | None] = ContextVar(
    "llm_call_counts",
    default=None,
)


def start_llm_call_accounting():
    """Start an isolated counter set for the current request context."""

    return _current_counts.set({"total_attempts": 0, "by_stage": {}})


def reset_llm_call_accounting(token) -> None:
    _current_counts.reset(token)


def record_llm_event(
    *,
    stage: str,
    event: str,
    provider: str,
    tier: str,
) -> None:
    """Record only routing metadata; prompts and document text are excluded."""

    counts = _current_counts.get()
    if counts is None:
        return

    stage_counts = counts["by_stage"].setdefault(
        stage,
        {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "rate_limits": 0,
            "cooldown_skips": 0,
            "duplicate_skips": 0,
            "providers": {},
        },
    )
    if event == "attempt":
        counts["total_attempts"] += 1
        stage_counts["attempts"] += 1
    elif event == "success":
        stage_counts["successes"] += 1
    elif event == "failure":
        stage_counts["failures"] += 1
    elif event == "rate_limit":
        stage_counts["rate_limits"] += 1
    elif event == "cooldown_skip":
        stage_counts["cooldown_skips"] += 1
    elif event == "duplicate_skip":
        stage_counts["duplicate_skips"] += 1

    provider_key = f"{tier}:{provider}"
    provider_counts = stage_counts["providers"].setdefault(
        provider_key,
        {"attempts": 0, "successes": 0, "failures": 0},
    )
    provider_counter = {
        "attempt": "attempts",
        "success": "successes",
        "failure": "failures",
    }.get(event)
    if provider_counter is not None:
        provider_counts[provider_counter] += 1


def get_llm_call_summary() -> dict[str, Any]:
    counts = _current_counts.get()
    if counts is None:
        return {"total_attempts": 0, "by_stage": {}}
    return {
        "total_attempts": counts["total_attempts"],
        "by_stage": {
            stage: {
                key: dict(value) if key == "providers" else value
                for key, value in stage_counts.items()
            }
            for stage, stage_counts in counts["by_stage"].items()
        },
    }
