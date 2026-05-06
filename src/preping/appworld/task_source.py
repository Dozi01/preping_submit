"""AppWorld task source for split-based task enumeration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class AppWorldTaskSource:
    """Load AppWorld task IDs from the official experiment utility."""

    def load_task_ids(self, split: str) -> List[str]:
        """Return task IDs for a split (e.g., train/test_normal)."""
        try:
            from experiments.appworld.experiment_utils import load_task_ids_from_split
        except ModuleNotFoundError:
            from experiment_utils import load_task_ids_from_split  # type: ignore[no-redef]

        return load_task_ids_from_split(split)
