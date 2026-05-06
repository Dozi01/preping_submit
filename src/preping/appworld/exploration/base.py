import logging
import re
from typing import Any, Dict, List, Optional

from preping.core.agents.base import BaseAgent
from preping.llm import LLMManager


logger = logging.getLogger(__name__)


class BaseExplorerAgent(BaseAgent):
    """Base class for AppWorld exploration agents."""

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
        max_tokens: int = 3000,
        use_thinking: bool = False,
        **kwargs,
    ):
        self.max_steps = max_steps
        self.exploration_temperature = temperature
        self.max_tokens = max_tokens
        self.use_thinking = use_thinking

        super().__init__(
            model_name=model_name,
            key=key,
            env=env,
            task_config=task_config,
            llm_cache=llm_cache,
            debug_mode=debug_mode,
            exp_config=exp_config,
            **kwargs,
        )

    def _initialize_system_message(self):
        """Empty system message - all content goes to user prompt."""
        self.system_message = ""

    def _initialize_llm(self):
        """Initialize LLM with exploration temperature."""
        self.llm = LLMManager.create_llm(
            model_name=self.model_name,
            key=self.key,
            system_message=self.system_message,
            temperature=self.exploration_temperature,
            max_tokens=self.max_tokens,
            use_thinking=self.use_thinking,
        )

    def get_exploration_prompt(self) -> str:
        raise NotImplementedError

    def build_prompt(self, env, context_sections: List[str]) -> List[Dict[str, str]]:
        """Build initial prompt for exploration."""
        messages = []
        user_content = self.get_exploration_prompt() + "\n\n"

        if hasattr(env, "task") and env.task:
            supervisor = getattr(env.task, "supervisor", None)
            if supervisor:
                first_name = getattr(supervisor, "first_name", "")
                last_name = getattr(supervisor, "last_name", "")
                email = getattr(supervisor, "email", "")
                phone = getattr(supervisor, "phone_number", "")
                user_content += (
                    f"My name is: {first_name} {last_name}. "
                    f"My personal email is {email} and phone number is {phone}.\n\n"
                )

        if context_sections:
            raise ValueError("Context Sections are not supported for exploration agents")

        messages.append({"role": "user", "content": user_content})
        return messages

    def extract_action(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if hasattr(response, "response"):
            response = response.response

        response = self._remove_think_markers(response)
        code, _ = self._extract_code_and_fix_content(response)
        action = code if code else response
        self._check_task_query(action)
        return action

    def _check_task_query(self, action: str):
        """Check if action contains task-revealing API calls."""
        patterns_to_detect = [
            r"apis\.supervisor\.show_active_task\s*\(",
            r"show_active_task\s*\(",
        ]

        self._task_query_intercepted = False
        for pattern in patterns_to_detect:
            if re.search(pattern, action):
                self._task_query_intercepted = True
                break

    def _get_observation(self, env) -> str:
        """Get observation from environment, with exploration message if task was queried."""
        obs = getattr(env, "observation", "")

        if getattr(self, "_task_query_intercepted", False):
            import json

            exploration_task = {
                "instruction": (
                    "Explore this environment and discover its capabilities. "
                    "Test various APIs, observe their behaviors, and learn how they work together."
                ),
                "status": None,
                "answer": "<<NOT_GIVEN>>",
            }
            exploration_msg = json.dumps(exploration_task, indent=2)
            obs = "Output: " + exploration_msg
            self._task_query_intercepted = False

        return obs

    def _remove_think_markers(self, text: str) -> str:
        """Remove <think> tags, keeping inner content."""
        pattern = r"</?think[^>]*>"
        return re.sub(pattern, "", text, flags=re.IGNORECASE)

    def _extract_code_and_fix_content(self, text: str, ignore_multiple_calls: bool = True) -> tuple:
        """Extract code from text with markdown code blocks."""
        code_block_pattern = r"```(?:python)?\s*([\s\S]*?)```"
        matches = re.findall(code_block_pattern, text)

        if matches:
            if ignore_multiple_calls:
                code = matches[0].strip()
            else:
                code = "\n".join(match.strip() for match in matches)
            return code, text

        partial_pattern = r"```(?:python)?\s*([\s\S]*)"
        partial_match = re.search(partial_pattern, text)
        if partial_match:
            return partial_match.group(1).strip(), text

        return "", text

    def run(self, env, max_iter: int = 50, context_sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run exploration - uses max_steps instead of max_iter."""
        effective_max = max_iter if max_iter is not None else self.max_steps
        return super().run(env, max_iter=effective_max, context_sections=context_sections)
