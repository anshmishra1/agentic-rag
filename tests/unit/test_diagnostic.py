import json

from agentic_rag.core.diagnostic import DiagnosticCollector


def test_diagnostic_collector_writes_run_artifacts(tmp_path) -> None:
    collector = DiagnosticCollector(
        run_id="diagnostics-test",
        run_directory=tmp_path,
        query="What are large language models?",
    )

    collector.record("run_started", question="What are large language models?")
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

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["summary"]["status"] == "success"
    assert summary["events"][-1]["stage"] == "performance"
