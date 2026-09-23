from agentic_rag.core.timing import PerformanceTracker


def test_performance_tracker_records_stage() -> None:
    tracker = PerformanceTracker()

    with tracker.measure("test_stage"):
        pass

    summary = tracker.summary_data()

    assert summary["stages"]["test_stage"]["calls"] == 1
    assert summary["stages"]["test_stage"]["total_time"] >= 0
    assert summary["total_time"] >= 0
    assert summary["bottleneck"] == "test_stage"


def test_performance_tracker_accumulates_calls() -> None:
    tracker = PerformanceTracker()

    with tracker.measure("test_stage"):
        pass

    with tracker.measure("test_stage"):
        pass

    summary = tracker.summary_data()
    stage = summary["stages"]["test_stage"]

    assert stage["calls"] == 2
    assert stage["total_time"] >= 0
    assert summary["total_time"] == stage["total_time"]
