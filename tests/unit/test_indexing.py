import pytest

from agentic_rag.indexing import require_versioned_index_name


@pytest.mark.parametrize(
    "name",
    ["agentic-rag-hybrid-v1", "agentic-rag-hybrid-v2", "retrieval-2026-v12"],
)
def test_versioned_index_name_accepts_numeric_suffix(name: str) -> None:
    assert require_versioned_index_name(f" {name} ") == name


@pytest.mark.parametrize(
    "name",
    [
        "agentic-rag-hybrid",
        "agentic-rag-hybrid-next",
        "Agentic-Rag-v2",
        "agentic_rag-v2",
        "-agentic-rag-v2",
        "agentic-rag-v0",
        "a" * 43 + "-v2",
    ],
)
def test_versioned_index_name_rejects_ambiguous_or_invalid_names(name: str) -> None:
    with pytest.raises(ValueError, match="numeric version"):
        require_versioned_index_name(name)
