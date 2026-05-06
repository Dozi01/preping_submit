import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from preping.appworld.exploration.base import BaseExplorerAgent


logger = logging.getLogger(__name__)


@dataclass
class TrajectoryMemory:
    """Shared memory for tracking exploration across tasks."""

    visited_apis: Set[Tuple[str, str]] = field(default_factory=set)
    api_call_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    error_patterns: List[Dict[str, Any]] = field(default_factory=list)
    successful_calls: List[Dict[str, Any]] = field(default_factory=list)

    def add_visit(self, app: str, api: str, success: bool = True, output: str = ""):
        """Record a visit to an API."""
        key = (app, api)
        self.visited_apis.add(key)
        self.api_call_counts[key] = self.api_call_counts.get(key, 0) + 1

        record = {
            "app": app,
            "api": api,
            "success": success,
            "output_snippet": output[:200] if output else "",
        }

        if success:
            self.successful_calls.append(record)
        else:
            self.error_patterns.append(record)

    def get_visit_count(self, app: str, api: str) -> int:
        """Get number of times an API has been visited."""
        return self.api_call_counts.get((app, api), 0)

    def get_visited_apis_list(self) -> List[str]:
        """Get list of visited APIs as strings."""
        return [f"{app}.{api}" for app, api in sorted(self.visited_apis)]

    def get_visit_summary(self) -> str:
        """Get formatted summary of visited APIs."""
        if not self.visited_apis:
            return "No APIs visited yet."

        summary_parts = []
        for (app, api), count in sorted(self.api_call_counts.items()):
            summary_parts.append(f"- {app}.{api}: {count} times")

        return "\n".join(summary_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about exploration."""
        return {
            "unique_apis_visited": len(self.visited_apis),
            "total_api_calls": sum(self.api_call_counts.values()),
            "apps_explored": len(set(app for app, _ in self.visited_apis)),
            "successful_calls": len(self.successful_calls),
            "error_calls": len(self.error_patterns),
        }

    def save(self, path: str):
        """Save memory to file."""
        data = {
            "visited_apis": list(self.visited_apis),
            "api_call_counts": {f"{key[0]}.{key[1]}": value for key, value in self.api_call_counts.items()},
            "stats": self.get_stats(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as file:
            json.dump(data, file, indent=2)
        logger.info("TrajectoryMemory saved to %s", path)

    def reset(self):
        """Reset memory - use sparingly."""
        self.visited_apis.clear()
        self.api_call_counts.clear()
        self.error_patterns.clear()
        self.successful_calls.clear()


class GuidedExplorerAgent(BaseExplorerAgent):
    """Guided exploration agent for AppWorld."""

    shared_memory: TrajectoryMemory = None

    @classmethod
    def get_shared_memory(cls) -> TrajectoryMemory:
        """Get or create shared memory."""
        if cls.shared_memory is None:
            cls.shared_memory = TrajectoryMemory()
        return cls.shared_memory

    @classmethod
    def reset_shared_memory(cls):
        """Reset shared memory - call before starting new experiment."""
        cls.shared_memory = TrajectoryMemory()
        logger.info("GuidedExplorerAgent shared memory reset")

    @classmethod
    def save_shared_memory(cls, path: str):
        """Save shared memory to file."""
        if cls.shared_memory:
            cls.shared_memory.save(path)

    def __init__(
        self,
        model_name: str,
        key: str,
        env,
        task_config: Dict[str, Any],
        llm_cache: Optional[Dict] = None,
        debug_mode: bool = False,
        exp_config: Optional[Any] = None,
        max_steps: int = 50,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            key=key,
            env=env,
            task_config=task_config,
            llm_cache=llm_cache,
            debug_mode=debug_mode,
            exp_config=exp_config,
            max_steps=max_steps,
            temperature=temperature,
            **kwargs,
        )

        self.memory = self.get_shared_memory()

        stats = self.memory.get_stats()
        logger.info(
            "GuidedExplorerAgent initialized. Shared memory: %s APIs visited, %s total calls",
            stats["unique_apis_visited"],
            stats["total_api_calls"],
        )

    def get_exploration_prompt(self) -> str:
        """Get guided exploration focused prompt."""
        visited_summary = self.memory.get_visit_summary()
        stats = self.memory.get_stats()

        instruction = f"""You are an AI assistant systematically exploring the AppWorld environment.

Your goal is to MAXIMIZE COVERAGE by visiting NEW, unexplored APIs.
You have access to a Python REPL environment where you can execute code.

EXPLORATION STRATEGY:
1. PRIORITIZE APIs you haven't visited before
2. Avoid repeating the same API calls unless testing new parameters
3. Be systematic - explore apps methodically
4. Note return formats, required parameters, and error conditions
5. When you feel you've explored enough NEW things, call: apis.supervisor.complete_task()

Here are three key APIs that you need to know to get more information

# To get a list of apps that are available to you.

```python
print(apis.api_docs.show_app_descriptions())
```

# To get the list of apis under any app listed above, e.g. spotify

```python
print(apis.api_docs.show_api_descriptions(app_name='spotify'))
```

# To get the specification of a particular api, e.g. spotify app's login api

```python
print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))
```

You will be given a list of ALREADY VISITED APIs - focus on exploring NEW ones.

=== EXPLORATION PROGRESS ===
Unique APIs visited so far: {stats['unique_apis_visited']}
Total API calls made: {stats['total_api_calls']}
Apps explored: {stats['apps_explored']}

=== ALREADY VISITED APIs ===
{visited_summary if visited_summary != "No APIs visited yet." else "None yet - you're the first explorer."}

=== YOUR MISSION ===
Focus on exploring NEW APIs that are NOT in the list above.
Start by checking available apps and find APIs you haven't tried yet.

Write Python code to explore.

**Key instructions**:
(1) Make sure to end code blocks with ``` followed by a newline(\n).

(2) Remember you can use the variables in your code in subsequent code blocks.

(3) Remember that the email addresses, access tokens and variables (e.g. spotify_password) in the example above are not valid anymore.

(4) You can use the "supervisor" app to get information about my accounts and use the "phone" app to get information about friends and family.

(5) Always look at API specifications (using apis.api_docs.show_api_doc) before calling an API.

(6) Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.

(7) Many APIs return items in "pages". Make sure to run through all the pages by looping over `page_index`.

(8) Once you have completed the task, make sure to call apis.supervisor.complete_task(). If the task asked for some information, return it as the answer argument, i.e. call apis.supervisor.complete_task(answer=<answer>). Many tasks do not require an answer, so in those cases, just call apis.supervisor.complete_task() i.e. do not pass any argument.
"""
        return instruction

    def forward(self, prompt):
        """Override forward to track API calls from action."""
        result = super().forward(prompt)
        action = result.action if hasattr(result, "action") else str(result)
        self._track_api_calls(action)
        return result

    def _track_api_calls(self, action: str):
        """Parse action code and track API calls."""
        import re

        pattern = r"apis\.(\w+)\.(\w+)\s*\("
        matches = re.findall(pattern, action)
        for app, api in matches:
            self.memory.add_visit(app, api)
            logger.debug("Tracked API call: %s.%s", app, api)
