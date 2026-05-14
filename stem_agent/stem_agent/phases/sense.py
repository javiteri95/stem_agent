"""
sense.py — Phase 1: Sense the task domain.

Calls the LLM once to produce a structured breakdown of how the given
task domain is typically solved. Returns a `primitives` dict.
"""

import json
from stem_agent.core.llm import call_llm

SENSE_PROMPT = """You are analyzing how the task domain '{domain}' is typically approached by expert systems and human researchers.

Produce a structured JSON analysis with exactly these keys:

{{
  "common_steps": ["list of 5-8 steps that typically appear in this task class"],
  "architecture_patterns": ["list of 3-5 named architecture patterns used (e.g. 'reflection loop', 'multi-agent debate')"],
  "tools_commonly_used": ["list of tools/APIs/capabilities typically needed"],
  "evaluation_criteria": ["list of 4-6 criteria by which outputs of this task are judged"],
  "known_failure_modes": ["list of 3-5 common ways these agents fail"],
  "key_design_decisions": ["list of 3-5 architectural choices that most affect quality"]
}}

Respond with ONLY the JSON object. No preamble, no markdown fences."""


def sense(task_domain: str) -> dict:
    """
    Phase 1: analyze the task domain and return a structured primitives dict.

    Acceptance criterion: returned dict has all six keys populated.
    """
    prompt = SENSE_PROMPT.format(domain=task_domain)
    messages = [{"role": "user", "content": prompt}]
    raw = call_llm(messages, max_tokens=2048)

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # Strip stray trailing text after the closing brace
    last_brace = raw.rfind("}")
    if last_brace != -1:
        raw = raw[: last_brace + 1]

    primitives = json.loads(raw)

    required_keys = [
        "common_steps",
        "architecture_patterns",
        "tools_commonly_used",
        "evaluation_criteria",
        "known_failure_modes",
        "key_design_decisions",
    ]
    missing = [k for k in required_keys if k not in primitives]
    if missing:
        raise ValueError(f"Phase 1 primitives missing keys: {missing}")

    print("Primitives extracted:")
    for key, val in primitives.items():
        print(f"  {key}: {val}")

    return primitives
