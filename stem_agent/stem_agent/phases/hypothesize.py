"""
hypothesize.py — Phase 2: Design the initial AgentSpec from domain primitives.

Calls the LLM once with the primitives produced by Phase 1 and populates
an AgentSpec (version 0). Saves it as checkpoint 0.
"""

import json
from stem_agent.core.agent_spec import AgentSpec
from stem_agent.core.llm import call_llm
from stem_agent.core.checkpointer import save_checkpoint

HYPOTHESIZE_PROMPT = """You are designing the initial architecture for a Deep Research agent.

Based on this analysis of how the domain is typically approached:
{primitives_json}

Choose initial values for an agent spec. You MUST choose from the allowed values listed.

Respond with ONLY a JSON object with these fields:
{{
  "system_prompt": "a system prompt for the research agent (2-4 sentences, focused on the domain)",
  "architecture_type": "one of: linear_chain, reflection_loop, multi_step_search, adversarial_qa",
  "search_strategy": "one of: breadth_first, depth_first, iterative_refinement",
  "synthesis_strategy": "one of: extractive, abstractive, hybrid",
  "max_search_rounds": "integer between 2 and 6",
  "self_critique": "true or false",
  "citation_style": "one of: inline, footnote, none",
  "tools": ["list of tool names from: web_search, document_reader, calculator, code_executor"],
  "rationale": "2-3 sentences explaining why you chose this combination"
}}

No preamble. No markdown fences. Only the JSON."""

VALID_VALUES = {
    "architecture_type": {"linear_chain", "reflection_loop", "multi_step_search", "adversarial_qa"},
    "search_strategy": {"breadth_first", "depth_first", "iterative_refinement"},
    "synthesis_strategy": {"extractive", "abstractive", "hybrid"},
    "citation_style": {"inline", "footnote", "none"},
}


def hypothesize(primitives: dict) -> AgentSpec:
    """
    Phase 2: produce AgentSpec version 0 from domain primitives.
    Saves checkpoint 0 (score placeholder — no eval yet at this stage).
    """
    prompt = HYPOTHESIZE_PROMPT.format(primitives_json=json.dumps(primitives, indent=2))
    messages = [{"role": "user", "content": prompt}]
    raw = call_llm(messages, max_tokens=1024)

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)
    rationale = data.pop("rationale", "")

    # Validate enum fields
    for field, allowed in VALID_VALUES.items():
        if field in data and data[field] not in allowed:
            raise ValueError(
                f"Field '{field}' got invalid value '{data[field]}'. Allowed: {allowed}"
            )

    # Coerce types
    data["max_search_rounds"] = int(data.get("max_search_rounds", 3))
    self_critique = data.get("self_critique", False)
    if isinstance(self_critique, str):
        self_critique = self_critique.lower() == "true"
    data["self_critique"] = self_critique

    spec = AgentSpec(**{k: v for k, v in data.items() if k in AgentSpec.__dataclass_fields__})
    spec.version = 0

    print(f"Initial architecture: {spec.architecture_type}")
    print(f"Search strategy: {spec.search_strategy} | Synthesis: {spec.synthesis_strategy}")
    print(f"Rationale: {rationale}")
    print(f"Spec: {spec.to_json()}")

    return spec
