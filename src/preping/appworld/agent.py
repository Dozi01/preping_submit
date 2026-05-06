"""AppWorld agent and configuration."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, Template

from preping.core.agents.base import BaseAgent


logger = logging.getLogger(__name__)


@dataclass
class AppWorldAgentConfig:
    """Configuration for the AppWorld agent."""

    name: str = "appworld_agent"
    description: str = "Agent for the AppWorld environment"
    default_timeout: int = 60
    error_message: str = "Operation timed out in AppWorld agent"
    prompt_file: Optional[str] = None
    model_name: str = "deepseek/deepseek-chat"
    temperature: float = 0.0
    use_thinking: bool = False
    max_tokens: int = 3000
    seed: Optional[int] = 42
    extra_config: Dict[str, Any] = field(default_factory=dict)


def create_appworld_agent_config(**kwargs) -> AppWorldAgentConfig:
    """Create AppWorld agent config from parameters."""
    known_fields = set(AppWorldAgentConfig.__dataclass_fields__.keys())
    config_params: Dict[str, Any] = {}
    extra_config: Dict[str, Any] = {}

    for key, value in kwargs.items():
        if key in known_fields:
            config_params[key] = value
        else:
            extra_config[key] = value

    if extra_config:
        config_params["extra_config"] = extra_config

    return AppWorldAgentConfig(**config_params)


class AppWorldAgent(BaseAgent):
    """AppWorld agent for code-based task completion."""

    def __init__(
        self,
        model_name: str,
        key: str,
        env,
        task_config: Dict[str, Any],
        llm_cache: Optional[Dict] = None,
        debug_mode: bool = False,
        exp_config: Optional[AppWorldAgentConfig] = None,
        **kwargs,
    ):
        if exp_config is None:
            exp_config = AppWorldAgentConfig()
        assert isinstance(
            exp_config, AppWorldAgentConfig
        ), f"exp_config should be an AppWorldAgentConfig object, but got {type(exp_config)}"

        self._prompt_dict: Dict[str, str] = {}
        self._working_dir = os.getcwd()
        self._template_cache: Dict[str, Template] = {}

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

        self.trajectory = []
        self.step_count = 0

    def _post_initialization_hook(self):
        """AppWorld-specific post-initialization."""
        self._initialize_prompt_config()

    def _initialize_prompt_config(self):
        """Initialize prompt configuration from exp_config."""
        if not hasattr(self.exp_config, "prompt_file") or self.exp_config.prompt_file is None:
            raise ValueError("prompt_file is required in exp_config")

        prompt_file = self.exp_config.prompt_file
        if not prompt_file.endswith(".jinja"):
            raise ValueError(f"prompt_file must be a .jinja file, got: {prompt_file}")

        self._prompt_dict = {"main_prompt_template": prompt_file}

    def _load_jinja_template(self, template_path: str) -> Template:
        """Load and cache a Jinja template from file."""
        if not os.path.isabs(template_path):
            template_path = os.path.join(self._working_dir, template_path)
        template_path = os.path.normpath(template_path)

        if template_path in self._template_cache:
            return self._template_cache[template_path]

        template_dir = os.path.dirname(template_path)
        template_name = os.path.basename(template_path)

        jinja_env = Environment(loader=FileSystemLoader(template_dir))
        template = jinja_env.get_template(template_name)

        self._template_cache[template_path] = template
        return template

    def parse_prompt_to_messages(self, input_str: str) -> List[Dict[str, Any]]:
        """Parse a prompt string with SYSTEM:/USER:/ASSISTANT: markers into messages."""
        messages_json: List[Dict[str, Any]] = []
        last_start = 0

        for match in re.finditer(r"(USER|ASSISTANT|SYSTEM):\n", input_str, flags=re.IGNORECASE):
            last_end = match.span()[0]
            if len(messages_json) == 0:
                if last_end != 0:
                    raise ValueError(f"Start of the prompt has no assigned role: {input_str[:last_end]}")
            else:
                messages_json[-1]["content"] = input_str[last_start:last_end].strip()
            role = match.group(1).lower()
            messages_json.append({"role": role, "content": None})
            last_start = match.span()[1]

        if messages_json:
            messages_json[-1]["content"] = input_str[last_start:].strip()
        else:
            raise ValueError("No role markers found in prompt")

        return messages_json

    def _get_observation(self, env) -> str:
        """Get the current observation from AppWorld environment."""
        return env.observation if env.observation else ""

    def build_prompt(self, env, context_sections: List[str]) -> List[Dict[str, str]]:
        """Build the initial conversation messages for AppWorld task."""
        context = "".join(context_sections)

        task_instruction = env.task.instruction if env.task else "No task specified"
        supervisor = env.task.supervisor if env.task else {}

        template_path = self._prompt_dict.get("main_prompt_template")
        if not template_path:
            raise ValueError("main_prompt_template not found in prompt_dict")

        template = self._load_jinja_template(template_path)
        dictionary = {"supervisor": supervisor, "instruction": task_instruction, "context": context}
        rendered_prompt = template.render(dictionary)

        return self.parse_prompt_to_messages(rendered_prompt)

    PARTIAL_CODE_REGEX = r".*```python\n(.*)"
    FULL_CODE_REGEX = r"```python\n(.*?)```"

    def extract_action(self, response: str) -> str:
        """Extract Python code from LLM response."""
        response = self._remove_think_markers(response)
        code, _ = self._extract_code_and_fix_content(response)
        return code

    def _extract_code_and_fix_content(self, text: str, ignore_multiple_calls: bool = True) -> tuple[str, str]:
        """Extract code from text and fix content for clean display."""
        original_text = text
        output_code = ""
        match_end = 0

        for re_match in re.finditer(self.FULL_CODE_REGEX, original_text, flags=re.DOTALL):
            code = re_match.group(1).strip()
            if ignore_multiple_calls:
                text = original_text[: re_match.end()]
                return code, text
            output_code += code + "\n"
            match_end = re_match.end()

        partial_match = re.match(self.PARTIAL_CODE_REGEX, original_text[match_end:], flags=re.DOTALL)
        if partial_match:
            output_code += partial_match.group(1).strip()
            if not text.endswith("\n"):
                text = text + "\n"
            text = text + "```"

        if len(output_code) == 0:
            return "", text
        return output_code, text

    def _remove_think_markers(self, text: str) -> str:
        """Remove <think> and </think> tags, keeping inner content intact."""
        text = re.sub(r"<\s*think[^>]*>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<\s*/\s*think\s*>", "", text, flags=re.IGNORECASE)
        return text

    def _determine_success(self, env, reward: float, info: Dict) -> bool:
        """Determine if AppWorld task was successful."""
        if "success" in info:
            return bool(info["success"])
        return super()._determine_success(env, reward, info)


def create_appworld_agent(
    model_name: str,
    key: str,
    env,
    task_config: Dict[str, Any],
    **kwargs,
) -> AppWorldAgent:
    """Factory function to create an AppWorld agent."""
    exp_config = kwargs.pop("exp_config", AppWorldAgentConfig())
    return AppWorldAgent(
        model_name=model_name,
        key=key,
        env=env,
        task_config=task_config,
        exp_config=exp_config,
        **kwargs,
    )
