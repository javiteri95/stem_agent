"""
LangChain-backed LLM wrapper.  All LLM calls in the stem agent go through
``call_llm()`` (plain text) or ``call_llm_json()`` (auto-parsed JSON).

LLM management is handled by LangChain's ``ChatLiteLLM``, which delegates to
the installed ``litellm`` package and therefore supports every model that
LiteLLM supports (OpenAI, Anthropic, Cohere, Gemini, etc.).

JSON output is parsed by LangChain's ``JsonOutputParser``, which handles
markdown fence stripping (```json … ```) before calling ``json.loads``.  A
secondary repair step trims trailing truncated text and re-tries parsing so
that slightly over-budget responses still succeed.

Supported env vars (checked in priority order):
  STEM_AGENT_MODEL  — preferred model override
  MODEL             — fallback model override (e.g. set in .env as MODEL=gpt-4o)

  OPENAI_API_KEY    — standard key
  OPENAI_KEY        — aliased to OPENAI_API_KEY automatically
  ANTHROPIC_API_KEY — required for Claude models
"""

import json
import os

from langchain_litellm import ChatLiteLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# Map alternative env var names before any LangChain initialisation
if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]

DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

# Models that use a shared reasoning+output token budget (max_completion_tokens).
# For these, we scale the requested budget up so thinking tokens don't eat all
# of the available space before the model can write a visible answer.
_REASONING_MODEL_SUBSTRINGS = ("o1", "o3", "o4", "gpt-5", "gpt5")
# Minimum output budget guaranteed to reasoning models regardless of caller value.
_REASONING_MIN_TOKENS = 16_000
# Multiplier applied on top of the caller's requested budget.
_REASONING_MULTIPLIER = 4

_json_parser = JsonOutputParser()


def _get_model() -> str:
    """Return the active model, checking STEM_AGENT_MODEL then MODEL then default."""
    return (
        os.environ.get("STEM_AGENT_MODEL")
        or os.environ.get("MODEL")
        or DEFAULT_MODEL
    )


def _is_reasoning_model(model: str) -> bool:
    """Return True if *model* is a reasoning/thinking model (o1, o3, gpt-5.x, …)."""
    m = model.lower()
    return any(s in m for s in _REASONING_MODEL_SUBSTRINGS)


def _to_lc_messages(messages: list[dict]) -> list:
    """Convert OpenAI-style message dicts to LangChain message objects."""
    result = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
        else:
            result.append(HumanMessage(content=content))
    return result


def get_llm(max_tokens: int = 2048) -> ChatLiteLLM:
    """Return a configured LangChain ChatLiteLLM instance for the active model.

    For reasoning models (o1, o3, gpt-5.x) the requested *max_tokens* is
    scaled up and passed as ``max_completion_tokens`` — the parameter OpenAI
    uses for the combined thinking+output budget — so internal chain-of-thought
    does not silently consume the entire quota before writing a visible answer.
    """
    model = _get_model()
    if _is_reasoning_model(model):
        budget = max(max_tokens * _REASONING_MULTIPLIER, _REASONING_MIN_TOKENS)
        return ChatLiteLLM(
            model=model,
            max_tokens=budget,
            model_kwargs={"max_completion_tokens": budget},
        )
    return ChatLiteLLM(model=model, max_tokens=max_tokens)


def call_llm(messages: list[dict], max_tokens: int = 2048) -> str:
    """
    Call the configured LLM and return the response text.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        max_tokens: Maximum tokens in the response.

    Returns:
        The assistant's reply as a plain string.
    """
    response = get_llm(max_tokens).invoke(_to_lc_messages(messages))
    return response.content


def call_llm_json(messages: list[dict], max_tokens: int = 2048) -> dict | list:
    """
    Call the configured LLM and return the response as a parsed Python object.

    Uses LangChain's ``JsonOutputParser`` which:
    * strips markdown fences (```json … ```)
    * calls ``json.loads`` on the cleaned text
    * raises ``OutputParserException`` on failure

    A secondary repair step attempts to recover from truncated responses by
    trimming everything after the last ``]`` or ``}`` and retrying.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        max_tokens: Maximum tokens in the response.

    Returns:
        The parsed JSON as a dict or list.

    Raises:
        OutputParserException: If parsing and repair both fail.
    """
    raw = call_llm(messages, max_tokens=max_tokens)
    try:
        return _json_parser.parse(raw)
    except OutputParserException:
        # Truncation repair: trim after the last closing bracket / brace
        repaired = raw.strip()
        last_bracket = repaired.rfind("]")
        last_brace = repaired.rfind("}")
        cut = max(last_bracket, last_brace)
        if cut != -1:
            try:
                return json.loads(repaired[: cut + 1])
            except json.JSONDecodeError:
                pass
        raise

