"""Proposer-side memory store for PrePing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .records import ProposerMemoryRecord


class PrePingProposerMemory:
    """Accumulate synthetic-task outcomes for future task generation."""

    def __init__(self) -> None:
        self._entries: List[ProposerMemoryRecord] = []

    def update_from_cycle(
        self,
        *,
        cycle_index: int,
        cycle_results: List[Dict[str, Any]],
        task_diagnostics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        aggregate_map = (task_diagnostics.get("by_synthetic_task_index") or {})
        grouped_results: Dict[str, List[Dict[str, Any]]] = {}
        for result in cycle_results:
            synthetic_idx = (result.get("original_task") or {}).get("synthetic_task_index")
            if synthetic_idx is None:
                continue
            grouped_results.setdefault(str(synthetic_idx), []).append(result)

        new_entries: List[ProposerMemoryRecord] = []
        for synthetic_idx, aggregate_item in sorted(aggregate_map.items(), key=lambda pair: int(pair[0])):
            group_results = grouped_results.get(synthetic_idx, [])
            failure_reasons = self._collect_unique_validation_reasons(
                group_results,
                field="task_completion_reason",
                validation_results={"failure", "invalid"},
            )
            feasibility_reasons = self._collect_unique_validation_reasons(group_results, field="feasibility_reason")
            new_entries.append(
                ProposerMemoryRecord(
                    cycle=cycle_index,
                    synthetic_task_index=int(synthetic_idx),
                    instruction=str(aggregate_item.get("instruction", "")),
                    diagnostic_mode=str(aggregate_item.get("diagnostic_mode", "")),
                    execution_mode=str(aggregate_item.get("execution_mode", "")),
                    success_count=int(aggregate_item.get("success_count", 0) or 0),
                    failure_count=int(aggregate_item.get("failure_count", 0) or 0),
                    invalid_count=int(aggregate_item.get("invalid_count", 0) or 0),
                    aggregate_pass=aggregate_item.get("aggregate_pass"),
                    task_generation_category=str(aggregate_item.get("task_generation_category", "")),
                    aggregate_fail_reasons=list(aggregate_item.get("aggregate_fail_reasons", [])),
                    failure_reasons=failure_reasons,
                    feasibility_reasons=feasibility_reasons,
                )
            )

        self._entries.extend(new_entries)
        return [entry.to_dict() for entry in new_entries]

    def get_generation_summary(self) -> Dict[str, Any]:
        no_validator_tasks: List[Dict[str, Any]] = []
        success_tasks: List[Dict[str, Any]] = []
        failure_tasks: List[Dict[str, Any]] = []
        trivial_tasks: List[Dict[str, Any]] = []
        boundary_tasks: List[Dict[str, Any]] = []
        unsolved_feasible_tasks: List[Dict[str, Any]] = []
        infeasible_tasks: List[Dict[str, Any]] = []

        for entry in self._entries:
            if entry.diagnostic_mode == "no_validator":
                no_validator_tasks.append(self._build_generation_item(entry))
            if entry.task_generation_category == "success":
                success_tasks.append(self._build_generation_item(entry))
            if entry.task_generation_category == "failure":
                failure_tasks.append(
                    self._build_generation_item(
                        entry,
                        representative_failure_reason=(entry.failure_reasons[0] if entry.failure_reasons else None),
                    )
                )
            if entry.task_generation_category == "trivial":
                trivial_tasks.append(self._build_generation_item(entry))
            if entry.task_generation_category == "boundary":
                boundary_tasks.append(
                    self._build_generation_item(
                        entry,
                        representative_failure_reason=(entry.failure_reasons[0] if entry.failure_reasons else None),
                    )
                )
            if entry.task_generation_category == "unsolved_feasible":
                unsolved_feasible_tasks.append(
                    self._build_generation_item(
                        entry,
                        representative_failure_reason=(entry.failure_reasons[0] if entry.failure_reasons else None),
                    )
                )
            if entry.task_generation_category == "infeasible":
                infeasible_tasks.append(
                    self._build_generation_item(
                        entry,
                        representative_infeasible_reason=self._pick_representative_infeasible_reason(entry),
                    )
                )

        return {
            "summary": {
                "num_entries": len(self._entries),
                "no_validator_count": len(no_validator_tasks),
                "success_count": len(success_tasks),
                "failure_count": len(failure_tasks),
                "trivial_count": len(trivial_tasks),
                "boundary_count": len(boundary_tasks),
                "unsolved_feasible_count": len(unsolved_feasible_tasks),
                "infeasible_count": len(infeasible_tasks),
            },
            "no_validator_tasks": no_validator_tasks,
            "success_tasks": success_tasks,
            "failure_tasks": failure_tasks,
            "trivial_tasks": trivial_tasks,
            "boundary_tasks": boundary_tasks,
            "unsolved_feasible_tasks": unsolved_feasible_tasks,
            "infeasible_tasks": infeasible_tasks,
        }

    def snapshot(self, max_entries: int = 20) -> Dict[str, Any]:
        entries = self._entries[-max_entries:] if max_entries > 0 else self._entries
        return {
            "num_entries": len(self._entries),
            "recent_entries": [entry.to_dict() for entry in entries],
            "generation_summary": self.get_generation_summary(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_entries": len(self._entries),
            "entries": [entry.to_dict() for entry in self._entries],
            "generation_summary": self.get_generation_summary(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PrePingProposerMemory":
        memory = cls()
        entries = payload.get("entries") or payload.get("recent_entries") or []
        memory._entries = [
            ProposerMemoryRecord(
                cycle=int(entry.get("cycle", 0) or 0),
                synthetic_task_index=int(entry.get("synthetic_task_index", 0) or 0),
                instruction=str(entry.get("instruction", "")),
                diagnostic_mode=str(entry.get("diagnostic_mode", "")),
                execution_mode=str(entry.get("execution_mode", "")),
                success_count=int(entry.get("success_count", 0) or 0),
                failure_count=int(entry.get("failure_count", 0) or 0),
                invalid_count=int(entry.get("invalid_count", 0) or 0),
                aggregate_pass=entry.get("aggregate_pass"),
                task_generation_category=str(entry.get("task_generation_category", "")),
                aggregate_fail_reasons=list(entry.get("aggregate_fail_reasons", [])),
                failure_reasons=list(entry.get("failure_reasons", [])),
                feasibility_reasons=list(entry.get("feasibility_reasons", [])),
            )
            for entry in entries
        ]
        return memory

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "PrePingProposerMemory":
        input_path = Path(path)
        with input_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return cls.from_dict(payload)

    @staticmethod
    def _collect_unique_validation_reasons(
        group_results: List[Dict[str, Any]],
        *,
        field: str,
        validation_results: set[str] | None = None,
        max_reasons: int = 3,
    ) -> List[str]:
        reasons: List[str] = []
        for result in group_results:
            validation = result.get("validation") or {}
            if validation_results is not None and str(validation.get("validation_result", "")) not in validation_results:
                continue
            reason = str(validation.get(field, "")).strip()
            if not reason or reason in reasons:
                continue
            reasons.append(reason)
            if len(reasons) >= max_reasons:
                break
        return reasons

    @classmethod
    def _build_generation_item(
        cls,
        entry: ProposerMemoryRecord,
        *,
        representative_failure_reason: str | None = None,
        representative_infeasible_reason: str | None = None,
    ) -> Dict[str, Any]:
        item = {
            "cycle": entry.cycle,
            "synthetic_task_index": entry.synthetic_task_index,
            "instruction": entry.instruction,
            "diagnostic_mode": entry.diagnostic_mode,
            "execution_mode": entry.execution_mode,
            "success_count": entry.success_count,
            "failure_count": entry.failure_count,
            "invalid_count": entry.invalid_count,
        }
        if representative_failure_reason:
            item["representative_failure_reason"] = representative_failure_reason
        if representative_infeasible_reason:
            item["representative_infeasible_reason"] = representative_infeasible_reason
        return item

    @classmethod
    def _pick_representative_infeasible_reason(cls, entry: ProposerMemoryRecord) -> str:
        if entry.feasibility_reasons:
            return entry.feasibility_reasons[0]
        if entry.aggregate_fail_reasons:
            return entry.aggregate_fail_reasons[0]
        return ""
