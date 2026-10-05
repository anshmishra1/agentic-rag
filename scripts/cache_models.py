"""Download or verify the exact local models used by retrieval."""

from __future__ import annotations

import argparse
import os

from sentence_transformers import CrossEncoder, SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL"))
    parser.add_argument(
        "--embedding-revision",
        default=os.getenv("EMBEDDING_MODEL_REVISION"),
    )
    parser.add_argument("--reranker-model", default=os.getenv("CROSS_ENCODER_MODEL"))
    parser.add_argument(
        "--reranker-revision",
        default=os.getenv("CROSS_ENCODER_MODEL_REVISION"),
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not all((
        args.embedding_model,
        args.embedding_revision,
        args.reranker_model,
        args.reranker_revision,
    )):
        parser.error("model names and revisions are required")

    local_only = args.verify_only
    embedding = SentenceTransformer(
        args.embedding_model,
        revision=args.embedding_revision,
        device="cpu",
        local_files_only=local_only,
    )
    vector = embedding.encode(
        ["model cache verification"],
        show_progress_bar=False,
    )
    if tuple(vector.shape) != (1, 384):
        raise RuntimeError(f"Unexpected embedding shape: {vector.shape}")

    reranker = CrossEncoder(
        args.reranker_model,
        revision=args.reranker_revision,
        device="cpu",
        local_files_only=local_only,
    )
    scores = reranker.predict(
        [("verification query", "verification passage")],
        batch_size=1,
        show_progress_bar=False,
    )
    if len(scores) != 1:
        raise RuntimeError(f"Unexpected reranker output length: {len(scores)}")

    action = "Verified" if local_only else "Cached"
    print(f"{action} embedding model {args.embedding_model}@{args.embedding_revision}")
    print(f"{action} reranker model {args.reranker_model}@{args.reranker_revision}")


if __name__ == "__main__":
    main()
