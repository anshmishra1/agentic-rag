from agentic_rag.policies.answer_format import answer_format_instructions


def test_math_contract_requires_valid_delimiters_and_plain_explanation() -> None:
    instructions = answer_format_instructions("math")

    assert "$$ delimiters" in instructions
    assert "single $ delimiters" in instructions
    assert "define its symbols" in instructions
    assert "plain language" in instructions
    assert "Keep source citations outside" in instructions


def test_textual_flowchart_uses_a_plain_text_code_block() -> None:
    instructions = answer_format_instructions("ascii_flowchart")

    assert "textual flowchart" in instructions
    assert "fenced plain-text code block" in instructions
    assert "citation on every factual node or transition" in instructions
    assert "arrows and borders need no citation" in instructions
