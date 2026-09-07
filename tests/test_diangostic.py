from pathlib import Path
from tempfile import TemporaryDirectory

from agentic_rag.core.diagnostic import DiagnosticCollector


def main() -> None:
    print("=" * 60)
    print("AGENTIC RAG DIAGNOSTICS TEST")
    print("=" * 60)

    with TemporaryDirectory() as temp_dir:
        collector = DiagnosticCollector(
            run_id="diagnostics_test",
            output_dir=Path(temp_dir),
        )

        collector.record(
            "run_started",
            question="What are large language models?",
        )

        collector.retrieval(
            "rerank_complete",
            candidate_count=8,
            top_score=0.8839,
            second_score=0.1963,
        )

        collector.llm(
            "completed",
            role="fast",
            provider="test-provider",
            model="test-model",
            latency_ms=250,
        )

        collector.grounding(
            "verification_complete",
            grade="grounded",
            retry_count=0,
        )

        collector.performance(
            "run_complete",
            total_time=10.024,
            bottleneck="retrieve",
        )

        jsonl_path, json_path = collector.finalize(
            summary={
                "status": "success",
                "question": "What are large language models?",
            }
        )

        assert jsonl_path.exists()
        assert json_path.exists()

        assert len(collector.events) == 5

        print(f"JSONL: {jsonl_path}")
        print(f"JSON : {json_path}")
        print(f"Events: {len(collector.events)}")

    print("=" * 60)
    print("DIAGNOSTICS TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()