"""
Isolated test for the application logging layer.

This test does not import or execute the RAG graph.
It verifies only:
    1. Standard logger initialization
    2. Category-specific logging
    3. Disabled category suppression
    4. Global debug override
"""

from agentic_rag.core.logging import (
    configure_logging,
    get_logger,
    log_graph,
    log_llm,
    log_performance,
    log_retrieval,
)


def test_enabled_categories() -> None:
    """Verify that explicitly enabled categories are emitted."""

    logger = get_logger("logging_test.enabled")

    print("\n--- ENABLED CATEGORY TEST ---")

    log_retrieval(
        logger,
        "query=%s | candidates=%d | top_score=%.4f",
        "test retrieval query",
        5,
        0.9234,
        enabled=True,
    )

    log_llm(
        logger,
        "provider=%s | model=%s | latency_ms=%d",
        "test-provider",
        "test-model",
        250,
        enabled=True,
    )

    log_performance(
        logger,
        "stage=%s | latency_ms=%d",
        "test_stage",
        125,
        enabled=True,
    )

    log_graph(
        logger,
        "node=%s | status=%s",
        "test_node",
        "completed",
        enabled=True,
    )


def test_disabled_categories() -> None:
    """Verify that disabled categories produce no log records."""

    logger = get_logger("logging_test.disabled")

    print("\n--- DISABLED CATEGORY TEST ---")
    print("The following messages should NOT appear:")

    log_retrieval(
        logger,
        "DISABLED RETRIEVAL LOG",
        enabled=False,
    )

    log_llm(
        logger,
        "DISABLED LLM LOG",
        enabled=False,
    )

    log_performance(
        logger,
        "DISABLED PERFORMANCE LOG",
        enabled=False,
    )

    log_graph(
        logger,
        "DISABLED GRAPH LOG",
        enabled=False,
    )


def test_debug_override() -> None:
    """Verify that debug configuration can be initialized."""

    print("\n--- DEBUG CONFIGURATION TEST ---")

    # This confirms that the function accepts the debug configuration
    # without requiring the RAG application to start.
    configure_logging(debug=True)

    logger = get_logger("logging_test.debug")

    logger.debug("Global DEBUG logging is available.")


def main() -> None:
    print("=" * 60)
    print("AGENTIC RAG LOGGING TEST")
    print("=" * 60)

    configure_logging(debug=False)

    logger = get_logger("logging_test")

    logger.info("Logging system initialized.")

    test_enabled_categories()
    test_disabled_categories()
    test_debug_override()

    print("\n" + "=" * 60)
    print("LOGGING TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()