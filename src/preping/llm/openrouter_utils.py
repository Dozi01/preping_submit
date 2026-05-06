"""OpenRouter-specific utility functions.

Handles provider routing, reasoning toggle detection, and extra_body payload construction
for OpenRouter API calls.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def deep_merge_dicts(
    base: Optional[Dict[str, Any]],
    override: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Recursively merge dictionaries without mutating the inputs."""
    merged: Dict[str, Any] = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_openrouter_provider_override(model_name: str) -> Optional[Dict[str, Any]]:
    """Return pinned OpenRouter provider routing for selected models."""
    # Avoid circular import: provider constants are defined on OpenRouterModel.
    # but we keep a standalone copy here for use by callers that don't need the full model.
    from preping.llm.providers import OpenRouterModel

    normalized_name = model_name.lower()
    if "deepseek/deepseek-v3.2" in normalized_name:
        return OpenRouterModel.DEEPSEEK_PROVIDER
    if "openai/gpt-oss-120b" in normalized_name:
        return OpenRouterModel.GPT_OSS_PROVIDER
    if "qwen/qwen3-235b-a22b-2507" in normalized_name:
        return OpenRouterModel.QWEN3_235B_PROVIDER
    return None


def openrouter_supports_reasoning_toggle(model_name: str) -> bool:
    """Return whether the OpenRouter model supports the shared reasoning.enabled toggle."""
    normalized_name = model_name.lower()
    return any(
        supported_name in normalized_name
        for supported_name in (
            "deepseek/deepseek-v3.2",
            "qwen/qwen3-235b-a22b-2507",
        )
    )


def is_openrouter_qwen_non_thinking_model(model_name: str) -> bool:
    """Return whether the OpenRouter model is a Qwen instruct variant without native thinking mode."""
    normalized_name = model_name.lower()
    return "qwen/qwen3-235b-a22b-2507" in normalized_name


def build_openrouter_extra_body(
    model_name: str,
    *,
    provider_override: Optional[Dict[str, Any]] = None,
    reasoning_config: Optional[Dict[str, Any]] = None,
    extra_body_override: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build a merged OpenRouter extra_body payload for routing and reasoning controls."""
    payload: Dict[str, Any] = {}
    provider = provider_override if provider_override is not None else get_openrouter_provider_override(model_name)
    if provider is not None:
        payload["provider"] = provider
    if reasoning_config is not None:
        payload["reasoning"] = reasoning_config
    payload = deep_merge_dicts(payload, extra_body_override)
    return payload or None
