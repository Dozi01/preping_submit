"""Canonical record types for PrePing cycle bookkeeping."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GeneratedTaskRecord:
    """Synthetic task metadata persisted at the task level."""

    task_id: str
    cycle: int
    synthetic_task_index: Optional[int]
    instruction: str
    category: str
    involved_apps: List[str] = field(default_factory=list)
    involved_apis: List[str] = field(default_factory=list)
    memory_utilization: str = ""
    repeat_index: Optional[int] = None
    repeat_total: Optional[int] = None
    validation: Dict[str, Any] = field(default_factory=dict)
    task_diagnostics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task_dict(cls, task: Dict[str, Any]) -> "GeneratedTaskRecord":
        return cls(
            task_id=str(task.get("task_id", "")),
            cycle=int(task.get("cycle", 0) or 0),
            synthetic_task_index=task.get("synthetic_task_index"),
            instruction=str(task.get("instruction", task.get("task_description", ""))),
            category=str(task.get("category", "")),
            involved_apps=[str(item) for item in task.get("involved_apps", [])],
            involved_apis=[str(item) for item in task.get("involved_apis", [])],
            memory_utilization=str(task.get("memory_utilization", "")),
            repeat_index=task.get("repeat_index"),
            repeat_total=task.get("repeat_total"),
            validation=dict(task.get("validation") or {}),
            task_diagnostics=dict(task.get("task_diagnostics") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRunRecord:
    """Execution-level record for one repeated run of a synthetic task."""

    task_id: str
    synthetic_task_index: Optional[int]
    cycle: int
    instruction: str
    task_text: str
    repeat_index: Optional[int]
    repeat_total: Optional[int]
    success: bool
    error: str
    has_trajectory: bool
    task_output_dir: str
    validation: Dict[str, Any] = field(default_factory=dict)
    aggregate_validation: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result_dict(
        cls,
        result: Dict[str, Any],
        aggregate_validation: Optional[Dict[str, Any]] = None,
    ) -> "TaskRunRecord":
        original_task = result.get("original_task") or {}
        llm_history = result.get("llm_history")
        trajectory = result.get("trajectory")
        has_trajectory = bool(llm_history) or bool(trajectory)
        return cls(
            task_id=str(result.get("task_id", "")),
            synthetic_task_index=original_task.get("synthetic_task_index"),
            cycle=int(original_task.get("cycle", 0) or 0),
            instruction=str(original_task.get("instruction", original_task.get("task_description", ""))),
            task_text=str(result.get("task_text", "")),
            repeat_index=original_task.get("repeat_index"),
            repeat_total=original_task.get("repeat_total"),
            success=bool(result.get("success", False)),
            error=str(result.get("error", "")),
            has_trajectory=has_trajectory,
            task_output_dir=str(result.get("task_output_dir", "")),
            validation=dict(result.get("validation") or {}),
            aggregate_validation=dict(aggregate_validation or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SyntheticTaskAggregateRecord:
    """Aggregate validation record at the synthetic-task level."""

    synthetic_task_index: int
    instruction: str
    run_task_ids: List[str]
    run_count: int
    expected_runs: int
    diagnostic_mode: str
    min_feasibility_score: int
    require_mixed_outcomes: bool
    feasibility_scores: List[int]
    validation_result_counts: Dict[str, int]
    success_count: int
    failure_count: int
    invalid_count: int
    feasibility_gate_pass: bool
    outcome_diversity_pass: bool
    aggregate_pass: Optional[bool]
    execution_mode: str = ""
    task_generation_category: str = ""
    aggregate_fail_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProposerMemoryRecord:
    """Task-memory record used as Proposer memory for future generation."""

    cycle: int
    synthetic_task_index: int
    instruction: str
    diagnostic_mode: str
    execution_mode: str
    success_count: int
    failure_count: int
    invalid_count: int
    aggregate_pass: Optional[bool]
    task_generation_category: str = ""
    aggregate_fail_reasons: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    feasibility_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
