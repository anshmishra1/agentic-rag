"""Shared answer-format contract for document and general generation."""

from __future__ import annotations


def answer_format_instructions(response_format: str = "requested") -> str:
    instructions = [
        "Honor the user's requested presentation format.",
        "For display equations, use valid LaTeX inside $$ delimiters.",
        "For inline mathematics, use valid LaTeX inside single $ delimiters.",
        "Keep source citations outside math delimiters.",
        "After every non-trivial formula, define its symbols and explain the "
        "meaning in plain language.",
        "Do not emit raw \\[...\\] equation delimiters.",
    ]

    if response_format == "ascii_flowchart":
        instructions.append(
            "Give the requested textual flowchart as a fenced plain-text code "
            "block, then explain its important transitions."
        )
    elif response_format == "math":
        instructions.append(
            "Lead with an intuitive explanation, then present the equation and "
            "a symbol-by-symbol interpretation."
        )
    elif response_format == "table":
        instructions.append("Use a concise Markdown table where it improves clarity.")
    elif response_format == "bullets":
        instructions.append("Use a concise bulleted structure.")

    return "\n".join(f"- {instruction}" for instruction in instructions)
