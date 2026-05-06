"""Core benchmark interfaces shared across benchmark implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Protocol, Sequence, runtime_checkable


@dataclass
class BenchmarkTask:
    """Lightweight representation of a single benchmark task."""

    task_id: str
    split: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Standardized per-task result used by the runner."""

    task_id: str
    split: str
    success: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    error: str | None = None


@dataclass
class RunnerConfig:
    """Runner-level options that are orthogonal to a specific benchmark."""

    num_workers: int = 1
    debug: bool = False
    continue_existing: bool = False
    max_retries: int = 0
    task_timeout_seconds: int | None = None
    progress: bool = True


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Protocol all benchmark adapters should satisfy."""

    name: str

    def prepare(self, *, runner_config: RunnerConfig) -> Mapping[str, Any] | None:
        """Optional one-time setup before tasks run."""

    def iter_tasks(
        self,
        *,
        runner_config: RunnerConfig,
        context: Mapping[str, Any] | None = None,
    ) -> Iterable[BenchmarkTask]:
        """Yield the tasks to execute."""

    def build_task_args(
        self,
        task: BenchmarkTask,
        *,
        runner_config: RunnerConfig,
        context: Mapping[str, Any] | None = None,
        position: int = 0,
        total: int = 0,
    ) -> Mapping[str, Any]:
        """Return serializable arguments for the worker function."""

    def should_skip(
        self,
        task_args: Mapping[str, Any],
        *,
        runner_config: RunnerConfig,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Return True if the task should be skipped."""

    def get_worker(self):
        """Return a picklable callable that accepts task_args and returns a raw result."""

    def build_result(
        self,
        raw_result: Any,
        task_args: Mapping[str, Any],
        *,
        runner_config: RunnerConfig,
        context: Mapping[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Convert a raw worker return value into a BenchmarkResult."""

    def summarize(
        self,
        results: Sequence[BenchmarkResult],
        *,
        runner_config: RunnerConfig,
        context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Produce a benchmark-level summary."""
