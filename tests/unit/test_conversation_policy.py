from agentic_rag.policies.conversation import is_control_query


def test_control_message_accepts_terminal_punctuation() -> None:
    assert is_control_query("Thanks!") is True


def test_documented_stop_message_is_control() -> None:
    assert is_control_query("Stop!") is True


def test_empty_message_is_control() -> None:
    assert is_control_query("   ") is True


def test_open_ended_language_is_left_to_semantic_planning() -> None:
    assert is_control_query("Are you sure?") is False
    assert is_control_query("Can you elaborate on that?") is False
    assert is_control_query("What is BM25?") is False
