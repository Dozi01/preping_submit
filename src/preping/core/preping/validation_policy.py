"""Selection and aggregation policy for PrePing task cycles."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from .records import SyntheticTaskAggregateRecord


class PrePingValidationPolicy:
    """Pure policy layer for repeated validation and task-history selection."""

    MEMORY_SELECTION_MODES = {
        "default",
        "single_run_include_failure",
    }

    @staticmethod
    def post_process_validation_results(
        *,
        cycle_results: List[Dict[str, Any]],
        validation_stats: Dict[str, Any],
        runs_per_task: int,
        repeat_eval_min_feasibility_score: int,
        repeat_eval_require_mixed_outcomes: bool,
        memory_selection_mode: str = "default",
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if validation_stats:
            task_diagnostics = PrePingValidationPolicy.build_task_diagnostics(
                cycle_results,
                min_feasibility_score=repeat_eval_min_feasibility_score,
                require_mixed_outcomes=repeat_eval_require_mixed_outcomes,
                runs_per_task=runs_per_task,
                memory_selection_mode=memory_selection_mode,
            )
            aggregate_pass_results = PrePingValidationPolicy.filter_results_by_task_diagnostics(
                cycle_results,
                task_diagnostics,
                require_aggregate_pass=True,
            )
            memory_input_results = PrePingValidationPolicy.select_memory_results(
                aggregate_pass_results,
                validation_stats=validation_stats,
                runs_per_task=runs_per_task,
                memory_selection_mode=memory_selection_mode,
            )
            return task_diagnostics, memory_input_results

        task_diagnostics = PrePingValidationPolicy.aggregate_no_validator_task_runs(
            cycle_results,
            runs_per_task=runs_per_task,
        )
        return task_diagnostics, cycle_results

    @staticmethod
    def build_task_diagnostics(
        results: List[Dict[str, Any]],
        *,
        min_feasibility_score: int = 5,
        require_mixed_outcomes: bool = True,
        runs_per_task: int = 1,
        memory_selection_mode: str = "default",
    ) -> Dict[str, Any]:
        if runs_per_task > 1:
            return PrePingValidationPolicy.aggregate_repeated_task_validations(
                results,
                min_feasibility_score=min_feasibility_score,
                require_mixed_outcomes=require_mixed_outcomes,
            )
        return PrePingValidationPolicy.aggregate_single_run_task_validations(
            results,
            min_feasibility_score=min_feasibility_score,
            memory_selection_mode=memory_selection_mode,
        )

    @staticmethod
    def attach_validation_to_tasks(
        *,
        tasks_to_update: List[Dict[str, Any]],
        cycle_results: List[Dict[str, Any]],
        task_diagnostics: Dict[str, Any],
        validation_stats: Dict[str, Any],
    ) -> None:
        result_map = {result.get("task_id"): result for result in cycle_results}
        task_diagnostics_map = task_diagnostics.get("by_synthetic_task_index", {})

        for task in tasks_to_update:
            result = result_map.get(task.get("task_id"))
            if not result:
                continue

            if validation_stats and "validation" in result:
                task["validation"] = result["validation"]

            synthetic_idx = task.get("synthetic_task_index")
            if synthetic_idx is None:
                continue

            diagnostic_item = task_diagnostics_map.get(str(synthetic_idx))
            if not diagnostic_item:
                continue

            task_diagnostic_payload = {
                "diagnostic_mode": diagnostic_item.get("diagnostic_mode", ""),
                "execution_mode": diagnostic_item.get("execution_mode", ""),
                "aggregate_pass": diagnostic_item.get("aggregate_pass"),
                "feasibility_gate_pass": diagnostic_item.get("feasibility_gate_pass"),
                "outcome_diversity_pass": diagnostic_item.get("outcome_diversity_pass"),
                "task_generation_category": diagnostic_item.get("task_generation_category", ""),
                "aggregate_fail_reasons": diagnostic_item.get("aggregate_fail_reasons", []),
                "validation_result_counts": diagnostic_item.get("validation_result_counts", {}),
            }
            task["task_diagnostics"] = task_diagnostic_payload

    @staticmethod
    def aggregate_no_validator_task_runs(
        results: List[Dict[str, Any]],
        *,
        runs_per_task: int,
    ) -> Dict[str, Any]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for result in results:
            synthetic_idx = (result.get("original_task") or {}).get("synthetic_task_index")
            if synthetic_idx is not None:
                grouped[str(synthetic_idx)].append(result)

        tasks: List[Dict[str, Any]] = []
        by_synthetic_task_index: Dict[str, Dict[str, Any]] = {}
        diagnostic_mode = "repeated_eval" if runs_per_task > 1 else "single_run"

        for synthetic_idx, group_results in sorted(grouped.items(), key=lambda pair: int(pair[0])):
            first_task = (group_results[0].get("original_task") or {}) if group_results else {}
            item = SyntheticTaskAggregateRecord(
                synthetic_task_index=int(synthetic_idx),
                instruction=first_task.get("instruction", ""),
                run_task_ids=[str(result.get("task_id", "")) for result in group_results],
                run_count=len(group_results),
                expected_runs=int(first_task.get("repeat_total", len(group_results)) or len(group_results)),
                diagnostic_mode="no_validator",
                min_feasibility_score=0,
                require_mixed_outcomes=False,
                feasibility_scores=[],
                validation_result_counts={},
                success_count=0,
                failure_count=0,
                invalid_count=0,
                feasibility_gate_pass=False,
                outcome_diversity_pass=False,
                aggregate_pass=None,
                execution_mode=diagnostic_mode,
                task_generation_category="",
                aggregate_fail_reasons=[],
            ).to_dict()
            tasks.append(item)
            by_synthetic_task_index[synthetic_idx] = item

        groups_total = len(tasks)
        summary = {
            "diagnostic_mode": "no_validator",
            "groups_total": groups_total,
            "groups_passed": None,
            "groups_failed": None,
            "pass_rate": None,
            "criteria": {
                "runs_per_task": runs_per_task,
            },
            "tasks": tasks,
        }
        return {"summary": summary, "by_synthetic_task_index": by_synthetic_task_index}

    @staticmethod
    def select_representative_trajectory_results(cycle_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for result in cycle_results:
            original_task = result.get("original_task") or {}
            synthetic_idx = original_task.get("synthetic_task_index")
            key = str(synthetic_idx) if synthetic_idx is not None else f"__task_id__{result.get('task_id', 'unknown')}"
            grouped[key].append(result)

        def has_trajectory(item: Dict[str, Any]) -> bool:
            llm_history = item.get("llm_history")
            trajectory = item.get("trajectory")
            return isinstance(llm_history, list) and bool(llm_history) or isinstance(trajectory, list) and bool(trajectory)

        def existing_trajectory(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [item for item in items if has_trajectory(item)]

        def pick(items: List[Dict[str, Any]], validation_result: str) -> Optional[Dict[str, Any]]:
            for item in items:
                validation = item.get("validation") or {}
                if validation.get("validation_result") == validation_result:
                    return item
            return None

        selected_results: List[Dict[str, Any]] = []
        for _, group in sorted(grouped.items(), key=lambda pair: pair[0]):
            existing = existing_trajectory(group)
            if not existing:
                continue

            selected = pick(existing, "success")
            if not selected:
                selected = pick(existing, "failure")
            if not selected:
                selected = next((item for item in existing if item.get("success", False)), None)
            if not selected:
                selected = existing[0]
            selected_results.append(selected)

        return selected_results

    @staticmethod
    def _group_memory_candidate_results(
        results: List[Dict[str, Any]],
        *,
        allowed_validation_results: set[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for result in results:
            validation = result.get("validation") or {}
            if validation.get("validation_result") not in allowed_validation_results:
                continue
            original_task = result.get("original_task") or {}
            synthetic_idx = original_task.get("synthetic_task_index")
            key = str(synthetic_idx) if synthetic_idx is not None else f"__task_id__{result.get('task_id', '')}"
            grouped[key].append(result)
        return grouped

    @staticmethod
    def _sort_memory_result_key(item: Dict[str, Any]) -> tuple[int, str]:
        original_task = item.get("original_task") or {}
        repeat_index = int(original_task.get("repeat_index", 10**9) or 10**9)
        return (repeat_index, str(item.get("task_id", "")))

    @staticmethod
    def _pick_representative_memory_result(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        successes = [
            item for item in group if (item.get("validation") or {}).get("validation_result") == "success"
        ]
        if successes:
            return sorted(successes, key=PrePingValidationPolicy._sort_memory_result_key)[0]

        failures = [
            item for item in group if (item.get("validation") or {}).get("validation_result") == "failure"
        ]
        if failures:
            return sorted(failures, key=PrePingValidationPolicy._sort_memory_result_key)[0]

        return sorted(group, key=PrePingValidationPolicy._sort_memory_result_key)[0]

    @staticmethod
    def select_successful_memory_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped = PrePingValidationPolicy._group_memory_candidate_results(
            results,
            allowed_validation_results={"success"},
        )
        return [sorted(group, key=PrePingValidationPolicy._sort_memory_result_key)[0] for _, group in sorted(grouped.items(), key=lambda pair: pair[0])]

    @staticmethod
    def select_single_run_memory_results(
        results: List[Dict[str, Any]],
        *,
        memory_selection_mode: str,
    ) -> List[Dict[str, Any]]:
        if memory_selection_mode == "default":
            return PrePingValidationPolicy.select_successful_memory_results(results)
        if memory_selection_mode == "single_run_include_failure":
            grouped = PrePingValidationPolicy._group_memory_candidate_results(
                results,
                allowed_validation_results={"success", "failure"},
            )
            return [
                PrePingValidationPolicy._pick_representative_memory_result(group)
                for _, group in sorted(grouped.items(), key=lambda pair: pair[0])
            ]
        raise ValueError(f"Unsupported memory_selection_mode: {memory_selection_mode}")

    @staticmethod
    def select_memory_results(
        results: List[Dict[str, Any]],
        *,
        validation_stats: Dict[str, Any],
        runs_per_task: int,
        memory_selection_mode: str,
    ) -> List[Dict[str, Any]]:
        if not validation_stats:
            return results
        if runs_per_task > 1:
            if memory_selection_mode not in PrePingValidationPolicy.MEMORY_SELECTION_MODES:
                raise ValueError(f"Unsupported memory_selection_mode: {memory_selection_mode}")
            return PrePingValidationPolicy.select_successful_memory_results(results)
        return PrePingValidationPolicy.select_single_run_memory_results(
            results,
            memory_selection_mode=memory_selection_mode,
        )

    @staticmethod
    def aggregate_repeated_task_validations(
        results: List[Dict[str, Any]],
        *,
        min_feasibility_score: int = 5,
        require_mixed_outcomes: bool = True,
    ) -> Dict[str, Any]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for result in results:
            synthetic_idx = (result.get("original_task") or {}).get("synthetic_task_index")
            if synthetic_idx is not None:
                grouped[str(synthetic_idx)].append(result)

        tasks: List[Dict[str, Any]] = []
        by_synthetic_task_index: Dict[str, Dict[str, Any]] = {}

        for synthetic_idx, group_results in sorted(grouped.items(), key=lambda pair: int(pair[0])):
            validations = [result.get("validation") or {} for result in group_results]
            label_counts = Counter(str(validation.get("validation_result", "unknown")) for validation in validations)
            feasibility_scores = [int(validation.get("feasibility_score", 0) or 0) for validation in validations]
            success_count = label_counts.get("success", 0)
            failure_count = label_counts.get("failure", 0)
            invalid_count = label_counts.get("invalid", 0)
            has_mixed = success_count > 0 and failure_count > 0
            feasibility_gate_pass = all(score >= min_feasibility_score for score in feasibility_scores)
            outcome_diversity_pass = has_mixed or not require_mixed_outcomes
            aggregate_pass = feasibility_gate_pass and outcome_diversity_pass and invalid_count == 0
            task_generation_category = PrePingValidationPolicy._categorize_task_group(
                success_count=success_count,
                failure_count=failure_count,
                invalid_count=invalid_count,
                feasibility_gate_pass=feasibility_gate_pass,
            )

            reasons: List[str] = []
            if not feasibility_gate_pass:
                reasons.append(f"feasibility_score below {min_feasibility_score} exists")
            if invalid_count > 0:
                reasons.append("contains invalid validation_result")
            if require_mixed_outcomes and not has_mixed:
                reasons.append("missing mixed outcomes (success+failure)")

            first_task = (group_results[0].get("original_task") or {}) if group_results else {}
            item = SyntheticTaskAggregateRecord(
                synthetic_task_index=int(synthetic_idx),
                instruction=first_task.get("instruction", ""),
                run_task_ids=[str(result.get("task_id", "")) for result in group_results],
                run_count=len(group_results),
                expected_runs=int(first_task.get("repeat_total", len(group_results)) or len(group_results)),
                diagnostic_mode="repeated_eval",
                min_feasibility_score=min_feasibility_score,
                require_mixed_outcomes=require_mixed_outcomes,
                feasibility_scores=feasibility_scores,
                validation_result_counts=dict(label_counts),
                success_count=success_count,
                failure_count=failure_count,
                invalid_count=invalid_count,
                feasibility_gate_pass=feasibility_gate_pass,
                outcome_diversity_pass=outcome_diversity_pass,
                aggregate_pass=aggregate_pass,
                execution_mode="repeated_eval",
                task_generation_category=task_generation_category,
                aggregate_fail_reasons=reasons,
            ).to_dict()
            tasks.append(item)
            by_synthetic_task_index[synthetic_idx] = item

        groups_total = len(tasks)
        groups_passed = sum(1 for task in tasks if task.get("aggregate_pass"))
        summary = {
            "diagnostic_mode": "repeated_eval",
            "groups_total": groups_total,
            "groups_passed": groups_passed,
            "groups_failed": groups_total - groups_passed,
            "pass_rate": (groups_passed / groups_total) if groups_total else None,
            "criteria": {
                "min_feasibility_score": min_feasibility_score,
                "require_mixed_outcomes": require_mixed_outcomes,
                "runs_per_task": None,
            },
            "tasks": tasks,
        }
        return {"summary": summary, "by_synthetic_task_index": by_synthetic_task_index}

    @staticmethod
    def aggregate_single_run_task_validations(
        results: List[Dict[str, Any]],
        *,
        min_feasibility_score: int = 5,
        memory_selection_mode: str = "default",
    ) -> Dict[str, Any]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for result in results:
            synthetic_idx = (result.get("original_task") or {}).get("synthetic_task_index")
            if synthetic_idx is not None:
                grouped[str(synthetic_idx)].append(result)

        include_failure_memory = memory_selection_mode == "single_run_include_failure"

        tasks: List[Dict[str, Any]] = []
        by_synthetic_task_index: Dict[str, Dict[str, Any]] = {}

        for synthetic_idx, group_results in sorted(grouped.items(), key=lambda pair: int(pair[0])):
            validations = [result.get("validation") or {} for result in group_results]
            label_counts = Counter(str(validation.get("validation_result", "unknown")) for validation in validations)
            feasibility_scores = [int(validation.get("feasibility_score", 0) or 0) for validation in validations]
            success_count = label_counts.get("success", 0)
            failure_count = label_counts.get("failure", 0)
            invalid_count = label_counts.get("invalid", 0)
            feasibility_gate_pass = all(score >= min_feasibility_score for score in feasibility_scores)
            aggregate_pass = feasibility_gate_pass and invalid_count == 0
            if not include_failure_memory:
                aggregate_pass = aggregate_pass and success_count > 0
            task_generation_category = PrePingValidationPolicy._categorize_single_run_group(
                validations=validations,
                success_count=success_count,
                failure_count=failure_count,
                invalid_count=invalid_count,
                feasibility_gate_pass=feasibility_gate_pass,
            )

            reasons: List[str] = []
            if not feasibility_gate_pass:
                reasons.append(f"feasibility_score below {min_feasibility_score} exists")
            if invalid_count > 0:
                reasons.append("contains invalid validation_result")
            if success_count == 0 and not include_failure_memory:
                reasons.append("single-run success required")

            first_task = (group_results[0].get("original_task") or {}) if group_results else {}
            item = SyntheticTaskAggregateRecord(
                synthetic_task_index=int(synthetic_idx),
                instruction=first_task.get("instruction", ""),
                run_task_ids=[str(result.get("task_id", "")) for result in group_results],
                run_count=len(group_results),
                expected_runs=int(first_task.get("repeat_total", len(group_results)) or len(group_results)),
                diagnostic_mode="single_run",
                min_feasibility_score=min_feasibility_score,
                require_mixed_outcomes=False,
                feasibility_scores=feasibility_scores,
                validation_result_counts=dict(label_counts),
                success_count=success_count,
                failure_count=failure_count,
                invalid_count=invalid_count,
                feasibility_gate_pass=feasibility_gate_pass,
                outcome_diversity_pass=True,
                aggregate_pass=aggregate_pass,
                execution_mode="single_run",
                task_generation_category=task_generation_category,
                aggregate_fail_reasons=reasons,
            ).to_dict()
            tasks.append(item)
            by_synthetic_task_index[synthetic_idx] = item

        groups_total = len(tasks)
        groups_passed = sum(1 for task in tasks if task.get("aggregate_pass"))
        summary = {
            "diagnostic_mode": "single_run",
            "groups_total": groups_total,
            "groups_passed": groups_passed,
            "groups_failed": groups_total - groups_passed,
            "pass_rate": (groups_passed / groups_total) if groups_total else None,
            "criteria": {
                "min_feasibility_score": min_feasibility_score,
                "require_mixed_outcomes": False,
                "runs_per_task": 1,
            },
            "tasks": tasks,
        }
        return {"summary": summary, "by_synthetic_task_index": by_synthetic_task_index}

    @staticmethod
    def _categorize_task_group(
        *,
        success_count: int,
        failure_count: int,
        invalid_count: int,
        feasibility_gate_pass: bool,
    ) -> str:
        if invalid_count > 0 or not feasibility_gate_pass:
            return "infeasible"
        if success_count == 0 and failure_count > 0:
            return "unsolved_feasible"
        if failure_count > 0:
            return "boundary"
        return "trivial"


    @staticmethod
    def filter_results_by_task_diagnostics(
        results: List[Dict[str, Any]],
        task_diagnostics: Dict[str, Any],
        *,
        require_aggregate_pass: bool = True,
    ) -> List[Dict[str, Any]]:
        index_map = task_diagnostics.get("by_synthetic_task_index", {})
        filtered: List[Dict[str, Any]] = []
        for result in results:
            synthetic_idx = (result.get("original_task") or {}).get("synthetic_task_index")
            if synthetic_idx is None:
                continue
            aggregate_item = index_map.get(str(synthetic_idx))
            if not aggregate_item:
                continue
            if require_aggregate_pass and not aggregate_item.get("aggregate_pass", False):
                continue
            filtered.append(result)
        return filtered

    @staticmethod
    def _categorize_single_run_group(
        *,
        validations: List[Dict[str, Any]],
        success_count: int,
        failure_count: int,
        invalid_count: int,
        feasibility_gate_pass: bool,
    ) -> str:
        del validations, failure_count
        if invalid_count > 0 or not feasibility_gate_pass:
            return "infeasible"
        return "success" if success_count > 0 else "failure"
