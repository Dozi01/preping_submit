"""AppWorld-specific memory adapter and factory wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from preping.core.memory import MemoryManager, create_memory_store


@dataclass
class AppWorldMemoryAdapter:
    """Thin adapter to make memory usage explicit at the AppWorld boundary."""

    store: MemoryManager

    def get_memory(self, task_description: Optional[str] = None, **kwargs) -> List[str]:
        """Return memory context for an AppWorld task."""
        return self.store.get_memory(task_description=task_description, **kwargs)

    def get_context_sections(self, task_description: Optional[str] = None, **kwargs) -> List[str]:
        """Return memory context for prompt injection."""
        return self.get_memory(task_description=task_description, **kwargs)

    def process_episode(self, task_description: str, trajectory: List[Dict[str, Any]], **kwargs) -> Any:
        """Update memory using one completed AppWorld episode."""
        return self.store.process_episode(task_description=task_description, trajectory=trajectory, **kwargs)

    def get_token_usage_summary(self) -> Dict[str, Any]:
        """Expose memory token/cost summary for experiment reporting."""
        return self.store.get_token_usage_summary()


def create_appworld_memory_adapter(
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
) -> Optional[AppWorldMemoryAdapter]:
    """Create an AppWorld memory adapter from the shared core memory factory."""
    store = create_memory_store(
        memory_type=memory_type,
        memory_path=memory_path,
        model_name=model_name,
        temperature=temperature,
        use_thinking=use_thinking,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        include_task_context=include_task_context,
        use_ground_truth=use_ground_truth,
        playbook_max_bullets=playbook_max_bullets,
        use_reflector_curator_examples=use_reflector_curator_examples,
    )
    if store is None:
        return None
    return AppWorldMemoryAdapter(store=store)
