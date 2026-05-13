"""
Thin LiteLLM wrapper. All LLM calls in the stem agent go through call_llm().

Supported env vars (checked in priority order):
  STEM_AGENT_MODEL  — preferred model override
  MODEL             — fallback model override (e.g. set in .env as MODEL=gpt-4o)

  OPENAI_API_KEY    — standard LiteLLM OpenAI key
  OPENAI_KEY        — alternative name (mapped to OPENAI_API_KEY automatically)
  ANTHROPIC_API_KEY — required for Claude models
"""

import os
import litellm

# Suppress litellm's verbose success logging
litellm.success_callback = []
litellm.failure_callback = []

# Map alternative env var names to the ones LiteLLM expects
if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]

DEFAULT_MODEL = "claude-3-5-sonnet-20241022"


def _get_model() -> str:
    """Return the active model, checking STEM_AGENT_MODEL then MODEL then default."""
    return (
        os.environ.get("STEM_AGENT_MODEL")
        or os.environ.get("MODEL")
        or DEFAULT_MODEL
    )


def call_llm(messages: list[dict], max_tokens: int = 2048) -> str:
    """
    Call the configured LLM and return the response text.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        max_tokens: Maximum tokens in the response.

    Returns:
        The assistant's reply as a plain string.
    """
    model = _get_model()
    response = litellm.completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
