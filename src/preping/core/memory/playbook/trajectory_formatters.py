"""Shared trajectory-to-prompt formatters for playbook memory components."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping


TrajectoryPromptFormatter = Callable[[List[Dict[str, Any]]], str]


def format_llm_history_trajectory(trajectory: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, msg in enumerate(trajectory, 1):
        if "role" not in msg or "content" not in msg:
            raise ValueError(
                "Invalid llm_history trajectory entry at index "
                f"{idx}: expected keys 'role' and 'content', got {sorted(msg.keys())}"
            )
        role = str(msg["role"]).upper()
        content = msg["content"]
        lines.append(f"[{role}]: {content}")
        lines.append("")
    return "\n".join(lines)


def format_action_output_trajectory(trajectory: List[Dict[str, Any]]) -> str:
    lines = []
    for i, step in enumerate(trajectory, 1):
        if "action" not in step or "output" not in step:
            raise ValueError(
                "Invalid action_output trajectory entry at index "
                f"{i}: expected keys 'action' and 'output', got {sorted(step.keys())}"
            )
        lines.append(f"Step {i}:")
        lines.append(f"  Action: {step['action']}")
        lines.append(f"  Output: {step['output']}")
        lines.append("")
    return "\n".join(lines)


DEFAULT_TRAJECTORY_PROMPT_FORMATTERS: Dict[str, TrajectoryPromptFormatter] = {
    "llm_history": format_llm_history_trajectory,
    "action_output": format_action_output_trajectory,
}


def format_trajectory_for_prompt(
    trajectory: List[Dict[str, Any]],
    trajectory_format: str = "action_output",
    custom_formatters: Mapping[str, TrajectoryPromptFormatter] | None = None,
) -> str:
    formatters = dict(DEFAULT_TRAJECTORY_PROMPT_FORMATTERS)
    if custom_formatters:
        formatters.update(custom_formatters)

    formatter = formatters.get(trajectory_format)
    if formatter is None:
        available = ", ".join(sorted(formatters))
        raise ValueError(f"Unknown trajectory format '{trajectory_format}'. Available formats: {available}")

    formatted = formatter(trajectory)
    return formatted if formatted else "No trajectory available"
