from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass
class StageMetric:
    calls: int = 0
    total_time: float = 0.0


class PerformanceTracker:
    def __init__(self) -> None:
        self.stages: dict[str, StageMetric] = {}

    def measure(self, stage: str):
        return _StageTimer(self, stage)

    def _record(self, stage: str, elapsed: float) -> None:
        metric = self.stages.setdefault(stage, StageMetric())
        metric.calls += 1
        metric.total_time += elapsed

    def summary_data(self) -> dict[str, Any]:
        stages = {
            name: {
                "calls": metric.calls,
                "total_time": round(metric.total_time, 4),
            }
            for name, metric in self.stages.items()
        }

        total_time = sum(
            metric.total_time
            for metric in self.stages.values()
        )

        bottleneck = None

        if self.stages:
            bottleneck = max(
                self.stages.items(),
                key=lambda item: item[1].total_time,
            )[0]

        return {
            "stages": stages,
            "total_time": round(total_time, 4),
            "bottleneck": bottleneck,
        }

    def summary(self) -> dict[str, Any]:
        data = self.summary_data()

        print("\n" + "=" * 60)
        print("RUN PERFORMANCE SUMMARY")
        print("=" * 60)

        for stage, metric in data["stages"].items():
            print(
                f"{stage:<28}"
                f"{metric['calls']:>5} calls   "
                f"{metric['total_time']:>8.3f}s"
            )

        print("-" * 60)
        print(f"TOTAL: {data['total_time']:.3f}s")

        if data["bottleneck"]:
            print(f"BOTTLENECK: {data['bottleneck']}")

        print("=" * 60)

        return data


class _StageTimer:
    def __init__(self, tracker: PerformanceTracker, stage: str):
        self.tracker = tracker
        self.stage = stage
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = perf_counter() - self.start_time
        self.tracker._record(self.stage, elapsed)


_current_tracker: ContextVar[
    PerformanceTracker | None
] = ContextVar(
    "current_performance_tracker",
    default=None,
)


def set_current_tracker(
    tracker: PerformanceTracker,
):
    return _current_tracker.set(tracker)


def reset_current_tracker(token) -> None:
    _current_tracker.reset(token)


def get_current_tracker() -> PerformanceTracker:
    tracker = _current_tracker.get()

    if tracker is None:
        raise RuntimeError(
            "No PerformanceTracker is active for the current request."
        )

    return tracker