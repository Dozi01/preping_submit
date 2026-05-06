"""Persistence helpers for PrePing cycle artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict


logger = logging.getLogger(__name__)


class PrePingCycleLogger:
    """Persist normalized PrePing cycle artifacts."""

    def save_validation_result(self, result: Dict[str, Any]) -> None:
        task_output_dir = result.get("task_output_dir")
        validation = result.get("validation")
        if not task_output_dir or not validation:
            return
        task_output_path = Path(task_output_dir)
        self._write_json(task_output_path / "validation_result.json", validation)

        validation_prompt = result.get("validation_prompt")
        if isinstance(validation_prompt, str) and validation_prompt.strip():
            self._write_text(task_output_path / "validation_prompt.txt", validation_prompt)

    def save_proposer_memory(self, *, path: str, payload: Dict[str, Any]) -> None:
        self._write_json(Path(path), payload)

    def save_memory_update(self, *, cycle_dir: str, added_count: int, candidate_count: int) -> None:
        if candidate_count <= 0:
            return
        self._write_json(
            Path(cycle_dir) / "memory_update.json",
            {
                "candidate_count": candidate_count,
                "added_count": added_count,
            },
        )

    def save_grounded_environment_summaries(self, *, cycle_dir: str, payload: Any) -> None:
        if not payload:
            return
        self._write_json(Path(cycle_dir) / "grounded_environment_summaries.json", payload)

    def save_generation_debug_artifact(self, *, cycle_dir: str, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        self._write_json(Path(cycle_dir) / "generation_prompts.json", payload)

        prompts = payload.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            return
        first_prompt = prompts[0].get("prompt")
        if not isinstance(first_prompt, str) or not first_prompt.strip():
            return
        self._write_text(Path(cycle_dir) / "generation_prompt.txt", first_prompt)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
        except Exception as error:  # pragma: no cover - defensive runtime path
            logger.warning("Failed to save PrePing artifact %s: %s", path, error)

    @staticmethod
    def _write_text(path: Path, payload: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
        except Exception as error:  # pragma: no cover - defensive runtime path
            logger.warning("Failed to save PrePing artifact %s: %s", path, error)
