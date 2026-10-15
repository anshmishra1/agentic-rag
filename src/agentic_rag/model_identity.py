"""Canonical identifiers for locally provisioned Hugging Face models."""

from __future__ import annotations


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_DIMENSION = 384
LEGACY_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ettin-reranker-32m-v1"
CROSS_ENCODER_REVISION = "b33e5ceb5110773ea9cf5e00c9bedc83a8c2afdd"

_MODEL_ALIASES = {
    "all-MiniLM-L6-v2": EMBEDDING_MODEL,
    "cross-encoder/ms-marco-MiniLM-L-6-v2": LEGACY_CROSS_ENCODER_MODEL,
}


def canonical_model_name(model_name: str) -> str:
    """Map historical local settings to the baked Hugging Face repository."""
    return _MODEL_ALIASES.get(model_name, model_name)
