import ast
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from agentic_rag.policies.conversation import classify_query_intent


HISTORY = [
    HumanMessage(content="What is retrieval-augmented generation?"),
    AIMessage(content="It combines retrieval with language-model generation."),
]


def test_standalone_question_is_new_question() -> None:
    assert classify_query_intent("What is BM25?", HISTORY) == (
        "new_question",
        False,
    )


def test_referential_question_is_follow_up() -> None:
    assert classify_query_intent("Can you elaborate on that?", HISTORY) == (
        "follow_up",
        False,
    )


def test_follow_up_requires_history() -> None:
    assert classify_query_intent("Can you elaborate on that?", []) == (
        "new_question",
        False,
    )


def test_control_message_accepts_terminal_punctuation() -> None:
    assert classify_query_intent("Thanks!", HISTORY) == (
        "control",
        True,
    )


def test_documented_stop_message_is_control() -> None:
    assert classify_query_intent("Stop!", HISTORY) == (
        "control",
        True,
    )


def test_empty_message_is_control() -> None:
    assert classify_query_intent("   ", HISTORY) == (
        "control",
        True,
    )


def test_graph_nodes_do_not_shadow_conversation_policy() -> None:
    nodes_path = (
        Path(__file__).parents[2]
        / "src"
        / "agentic_rag"
        / "graph"
        / "nodes.py"
    )
    module = ast.parse(nodes_path.read_text(encoding="utf-8"))

    local_definitions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "classify_query_intent"
    ]

    assert local_definitions == []
