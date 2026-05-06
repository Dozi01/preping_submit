"""Task-local grounded environment summaries for PrePing."""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Callable, Dict, List, Optional

from .environment_info import PrePingEnvironmentInfo


logger = logging.getLogger(__name__)


class PrePingGroundedEnvironmentSummaries:
    """Summarize representative trajectories into prompt-ready grounded observations."""

    def __init__(
        self,
        *,
        llm_provider: Callable[[], Any],
        summary_prompt: str,
        verbose: bool = False,
    ) -> None:
        self._llm_provider = llm_provider
        self.summary_prompt = summary_prompt
        self.verbose = verbose

    @property
    def llm(self) -> Any:
        return self._llm_provider()

    @staticmethod
    def _normalize_summary(summary: Any) -> str:
        text = str(summary or "").strip()
        if not text:
            return ""
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    @classmethod
    def parse_summary_from_response(cls, response_text: str) -> str:
        """Parse the grounded summary payload from an LLM response."""
        payload = PrePingEnvironmentInfo.parse_json_from_response(response_text)
        if isinstance(payload, dict):
            summary = payload.get("summary", "")
        else:
            summary = payload
        normalized = cls._normalize_summary(summary)
        if not normalized:
            raise ValueError("Grounded environment summary response is empty.")
        return normalized

    def summarize_from_results(
        self,
        *,
        cycle_index: int,
        representative_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build one grounded summary per representative task result."""
        prompts: List[str] = []
        metadata: List[Dict[str, Any]] = []
        for result in representative_results:
            payload = PrePingEnvironmentInfo.build_extraction_payload_from_result(result)
            if payload is None:
                continue
            instruction = str(payload.get("task_instruction", "")).strip() or "Unknown"
            formatted_trajectory = PrePingEnvironmentInfo.format_trajectory_for_prompt(payload)
            prompts.append(
                self.summary_prompt.format(
                    task_instruction=instruction,
                    trajectory=formatted_trajectory,
                )
            )
            original_task = result.get("original_task") or {}
            metadata.append(
                {
                    "cycle": cycle_index,
                    "task_id": str(result.get("task_id", "")),
                    "synthetic_task_index": original_task.get("synthetic_task_index"),
                    "instruction": instruction,
                }
            )

        if not prompts:
            return []

        responses = self.llm.generate_batch(prompts)
        entries: List[Dict[str, Any]] = []
        for meta, response in zip(metadata, responses):
            response_text = response if isinstance(response, str) else str(response)
            if self.verbose:
                logger.info("\n[GROUNDED ENVIRONMENT SUMMARY RESPONSE]\n%s\n%s", response_text, "=" * 60)
            try:
                summary = self.parse_summary_from_response(response_text)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Failed to parse grounded environment summary for task_id=%s: %s", meta["task_id"], exc)
                continue

            entry = dict(meta)
            entry["grounded_summary"] = summary
            entries.append(entry)
        return entries

    @staticmethod
    def merge_summary_entries(
        existing_entries: List[Dict[str, Any]],
        new_entries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Append summary entries while preserving insertion order and de-duplicating exact repeats."""
        merged: List[Dict[str, Any]] = []
        seen = set()
        for entry in existing_entries + new_entries:
            key = json.dumps(
                {
                    "cycle": entry.get("cycle"),
                    "synthetic_task_index": entry.get("synthetic_task_index"),
                    "instruction": entry.get("instruction"),
                    "grounded_summary": entry.get("grounded_summary"),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry)
        return merged

    @staticmethod
    def prepare_for_generation(
        *,
        summary_entries: List[Dict[str, Any]],
        use_environment_info: bool,
        seed: int,
        cycle_num: int,
        max_summaries: int = 10,
    ) -> Optional[str]:
        """Sample grounded summaries and render them for the generation prompt."""
        if not use_environment_info or not summary_entries:
            return None

        sample_count = min(max(0, max_summaries), len(summary_entries))
        if sample_count == 0:
            return None

        rng = random.Random(f"{cycle_num}:grounded_environment_summaries")
        sampled_entries = rng.sample(summary_entries, sample_count) if sample_count < len(summary_entries) else list(summary_entries)

        blocks: List[str] = []
        for idx, entry in enumerate(sampled_entries, start=1):
            instruction = str(entry.get("instruction", "")).strip()
            summary = str(entry.get("grounded_summary", "")).strip()
            if not summary:
                continue
            lines = [f"Observation {idx}:", f"- Source task: {instruction}"]
            for line in summary.splitlines():
                lines.append(f"  {line}")
            blocks.append("\n".join(lines))

        if not blocks:
            return None
        return "\n\n".join(blocks)
