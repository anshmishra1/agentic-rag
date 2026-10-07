"""Deterministic policy for the small closed set of control messages."""

from __future__ import annotations

import re


_CONTROL_PATTERNS = (
    r"^\s*that's all[\s.!]*$",
    r"^\s*thats all[\s.!]*$",
    r"^\s*that is all[\s.!]*$",
    r"^\s*i(?:'m| am) done[\s.!]*$",
    r"^\s*done[\s.!]*$",
    r"^\s*no more questions[\s.!]*$",
    r"^\s*i don't have any more questions[\s.!]*$",
    r"^\s*i do not have any more questions[\s.!]*$",
    r"^\s*no further questions[\s.!]*$",
    r"^\s*nothing else[\s.!]*$",
    r"^\s*thank(?:s| you)[\s.!]*$",
    r"^\s*stop[\s.!]*$",
    r"^\s*end this[\s.!]*$",
    r"^\s*goodbye[\s.!]*$",
    r"^\s*bye[\s.!]*$",
)

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def is_control_query(question: str) -> bool:
    """Return True for closed commands that safely bypass all model calls."""
    normalized = _normalize(question)

    if not normalized:
        return True

    return any(re.search(pattern, normalized) for pattern in _CONTROL_PATTERNS)
