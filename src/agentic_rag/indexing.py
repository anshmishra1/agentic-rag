"""Index identity rules for retrieval schema migrations."""

from __future__ import annotations

import re


LEGACY_INDEX_NAME = "agentic-rag-hybrid"
_VERSIONED_INDEX_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?-v[1-9][0-9]*$")


def require_versioned_index_name(index_name: str) -> str:
    """Return a normalized Pinecone index name or reject unsafe migration names.

    Pinecone index names are deliberately kept separate for retrieval schema
    versions. A numeric ``-vN`` suffix makes the migration target explicit and
    prevents an experiment from replacing the legacy production index.
    """
    normalized = index_name.strip()
    if len(normalized) > 45 or not _VERSIONED_INDEX_NAME.fullmatch(normalized):
        raise ValueError(
            "Index names created by this project must be lowercase, use only "
            "letters, numbers, or hyphens, be at most 45 characters, and end "
            "with a numeric version such as '-v2'."
        )
    return normalized
