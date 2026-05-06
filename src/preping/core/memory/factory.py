"""Core memory factory.

This module is the stable import surface for creating PrePing memory stores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from .interface import MemoryManager


def create_memory_store(
    memory_type: Optional[str] = None,
    memory_path: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    use_thinking: bool = False,
    embedding_model: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    include_task_context: bool = True,
    use_ground_truth: bool = True,
    playbook_max_bullets: Optional[int] = None,
    use_reflector_curator_examples: bool = True,
    trajectory_formatters: Mapping[str, object] | None = None,
) -> Optional[MemoryManager]:
    """Create a MemoryManager from generic config options."""
    if not memory_type or not memory_path:
        return None

    if memory_type != "playbook":
        raise ValueError(f"Unsupported memory_type: {memory_type}")

    path = Path(memory_path)
    from preping.core.memory.playbook import PrePingMemoryManager
    from preping.core.memory.playbook.prompts import (
        CURATOR_SYSTEM_PROMPT,
        CURATOR_SYSTEM_PROMPT_NO_EXAMPLES,
        REFLECTOR_SYSTEM_PROMPT,
        REFLECTOR_SYSTEM_PROMPT_NO_EXAMPLES,
    )

    if bool(embedding_model) != bool(embedding_base_url):
        raise ValueError("Playbook semantic retrieval requires both embedding_model and embedding_base_url.")

    embedding_client = None
    if embedding_model and embedding_base_url:
        try:
            from preping.embedding_model import EmbeddingClient

            embedding_client = EmbeddingClient(
                model=embedding_model,
                base_url=embedding_base_url,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to initialize requested embedding client "
                f"(model={embedding_model}, base_url={embedding_base_url})."
            ) from exc

    return PrePingMemoryManager(
        model_name=model_name or "gpt-4o",
        playbook_path=path,
        temperature=temperature,
        use_thinking=use_thinking,
        embedding_client=embedding_client,
        include_task_context=include_task_context,
        use_ground_truth=use_ground_truth,
        reflector_system_prompt=(
            REFLECTOR_SYSTEM_PROMPT if use_reflector_curator_examples else REFLECTOR_SYSTEM_PROMPT_NO_EXAMPLES
        ),
        curator_system_prompt=(
            CURATOR_SYSTEM_PROMPT if use_reflector_curator_examples else CURATOR_SYSTEM_PROMPT_NO_EXAMPLES
        ),
        trajectory_formatters=trajectory_formatters,
    )
