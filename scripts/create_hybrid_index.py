"""Create the dot-product Pinecone index required by hybrid retrieval.

Changing the index metric is an index migration. This command never deletes
or converts an existing index; use a new versioned index name and re-ingest.
"""

from pinecone import Pinecone, ServerlessSpec

from agentic_rag.config import settings


EMBEDDING_DIMENSION = 384


def main() -> None:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = [index["name"] for index in pc.list_indexes()]
    target_name = settings.pinecone_index_name

    if target_name in existing:
        print(f"Index '{target_name}' already exists.")
        print(
            "Confirm that it uses metric='dotproduct'. If it does not, choose "
            "a new versioned PINECONE_INDEX_NAME and run this command again."
        )
        return

    pc.create_index(
        name=target_name,
        dimension=EMBEDDING_DIMENSION,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(
        f"Created hybrid index '{target_name}' "
        f"(metric=dotproduct, dimension={EMBEDDING_DIMENSION})."
    )
    print("Update configuration if needed, restart the app, and re-ingest data.")


if __name__ == "__main__":
    main()
