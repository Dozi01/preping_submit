"""Shared benchmark convenience base classes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from preping.core.interfaces.benchmark import BenchmarkResult, RunnerConfig


class BaseBenchmark:
    """Convenience base class with sensible defaults."""

    name: str = "benchmark"

    def prepare(self, *, runner_config: RunnerConfig) -> Mapping[str, Any] | None:  # noqa: D401
        return {}

    def should_skip(
        self,
        task_args: Mapping[str, Any],
        *,
        runner_config: RunnerConfig,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        return False

    def build_result(
        self,
        raw_result: Any,
        task_args: Mapping[str, Any],
        *,
        runner_config: RunnerConfig,
        context: Mapping[str, Any] | None = None,
    ) -> BenchmarkResult:
        return BenchmarkResult(
            task_id=str(task_args.get("task_id")),
            split=str(task_args.get("split", "")),
            success=bool(getattr(raw_result, "success", False) or raw_result.get("success", False)),
            payload=raw_result if isinstance(raw_result, dict) else {"raw_result": raw_result},
            skipped=bool(raw_result.get("skipped", False)) if isinstance(raw_result, dict) else False,
            error=raw_result.get("error") if isinstance(raw_result, dict) else None,
        )

    def summarize(
        self,
        results: Sequence[BenchmarkResult],
        *,
        runner_config: RunnerConfig,
        context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        successes = [r for r in results if r.success]
        return {
            "total_tasks": len(results),
            "successful_tasks": [r.task_id for r in successes],
        }
