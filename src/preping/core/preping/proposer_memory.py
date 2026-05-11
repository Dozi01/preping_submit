"""Proposer-side memory store for PrePing."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .records import ProposerMemoryRecord


class PrePingProposerMemory:
    """Accumulate synthetic-task outcomes for future task generation."""

    _APPWORLD_API_CALL_RE = re.compile(r"\bapis\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*\(")
    _SUPPORT_APPS = {"api_docs", "supervisor"}

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
            original_task = self._first_original_task(group_results)
            involved_apps = self._collect_task_list(original_task, "involved_apps", "servers")
            involved_apis = self._collect_task_list(original_task, "involved_apis", "intended_functions")
            used_apis = self._extract_invoked_apis(group_results)
            used_apps = self._apps_from_apis(used_apis)
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
                    involved_apps=involved_apps,
                    involved_apis=involved_apis,
                    used_apps=used_apps,
                    used_apis=used_apis,
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
            "usage_summary": self._build_usage_summary(),
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
        generation_summary = self.get_generation_summary()
        return {
            "num_entries": len(self._entries),
            "entries": [entry.to_dict() for entry in self._entries],
            "practice_history_view": generation_summary,
            "generation_summary": generation_summary,
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
                involved_apps=list(entry.get("involved_apps", [])),
                involved_apis=list(entry.get("involved_apis", [])),
                used_apps=list(entry.get("used_apps", [])),
                used_apis=list(entry.get("used_apis", [])),
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
            if (
                validation_results is not None
                and str(validation.get("validation_result", "")) not in validation_results
            ):
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
        if entry.involved_apps:
            item["involved_apps"] = list(entry.involved_apps)
        if entry.involved_apis:
            item["involved_apis"] = list(entry.involved_apis)
        if entry.used_apps:
            item["used_apps"] = list(entry.used_apps)
        if entry.used_apis:
            item["used_apis"] = list(entry.used_apis)
        return item

    @classmethod
    def _pick_representative_infeasible_reason(cls, entry: ProposerMemoryRecord) -> str:
        if entry.feasibility_reasons:
            return entry.feasibility_reasons[0]
        if entry.aggregate_fail_reasons:
            return entry.aggregate_fail_reasons[0]
        return ""

    @staticmethod
    def _first_original_task(group_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        for result in group_results:
            original_task = result.get("original_task")
            if isinstance(original_task, dict):
                return original_task
        return {}

    @classmethod
    def _collect_task_list(cls, original_task: Dict[str, Any], *keys: str) -> List[str]:
        values: List[str] = []
        for key in keys:
            raw_items = original_task.get(key)
            if not isinstance(raw_items, list):
                continue
            values.extend(str(item).strip() for item in raw_items if str(item).strip())
        return cls._unique_preserve_order(values)

    @classmethod
    def _extract_invoked_apis(cls, group_results: List[Dict[str, Any]]) -> List[str]:
        invoked: List[str] = []
        for result in group_results:
            for text in cls._iter_trajectory_texts(result):
                for app_name, api_name in cls._APPWORLD_API_CALL_RE.findall(text):
                    if app_name in cls._SUPPORT_APPS:
                        continue
                    invoked.append(f"{app_name}.{api_name}")
        return cls._unique_preserve_order(invoked)

    @staticmethod
    def _iter_trajectory_texts(result: Dict[str, Any]) -> List[str]:
        texts: List[str] = []

        trajectory = result.get("trajectory") or []
        if isinstance(trajectory, list):
            for step in trajectory:
                if isinstance(step, dict) and step.get("action"):
                    texts.append(str(step["action"]))

        llm_history = result.get("llm_history") or []
        if isinstance(llm_history, list):
            for message in llm_history:
                if not isinstance(message, dict):
                    continue
                if str(message.get("role", "")).lower() != "assistant":
                    continue
                content = message.get("content")
                if content:
                    texts.append(str(content))

        return texts

    @classmethod
    def _apps_from_apis(cls, api_names: List[str]) -> List[str]:
        app_names = [api_name.split(".", 1)[0] for api_name in api_names if "." in api_name]
        return cls._unique_preserve_order(app_names)

    @staticmethod
    def _unique_preserve_order(items: List[str]) -> List[str]:
        seen = set()
        unique_items: List[str] = []
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            unique_items.append(item)
        return unique_items

    def _build_usage_summary(self, *, recent_entries: int = 50, max_items: int = 10) -> Dict[str, Any]:
        entries = self._entries[-recent_entries:] if recent_entries > 0 else self._entries
        app_counts: Counter[str] = Counter()
        api_counts: Counter[str] = Counter()
        for entry in entries:
            app_counts.update(entry.used_apps)
            api_counts.update(entry.used_apis)

        return {
            "recent_entry_count": len(entries),
            "recent_overused_apps": self._format_counter_items(app_counts, max_items=max_items),
            "recent_overused_apis": self._format_counter_items(api_counts, max_items=max_items),
        }

    @staticmethod
    def _format_counter_items(counter: Counter[str], *, max_items: int) -> List[Dict[str, Any]]:
        return [
            {"name": name, "count": count}
            for name, count in counter.most_common(max_items)
            if name and count > 0
        ]
