from dataclasses import dataclass, field, asdict
from typing import Optional
import json, uuid, datetime


@dataclass
class AgentSpec:
    spec_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    version: int = 0
    task_domain: str = "Deep Research"

    # Core architecture choices (mutated by Phase 3)
    system_prompt: str = ""
    architecture_type: str = ""   # "linear_chain" | "reflection_loop" | "multi_step_search" | "adversarial_qa"
    search_strategy: str = ""     # "breadth_first" | "depth_first" | "iterative_refinement"
    synthesis_strategy: str = ""  # "extractive" | "abstractive" | "hybrid"
    max_search_rounds: int = 3
    self_critique: bool = False
    citation_style: str = "inline"  # "inline" | "footnote" | "none"

    # Tools the agent can use (list of tool names as strings)
    tools: list = field(default_factory=list)

    # Eval history (append-only)
    eval_scores: list = field(default_factory=list)  # list of dicts: {version, scores, timestamp}

    # Mutation rationale (append-only log)
    mutation_log: list = field(default_factory=list)  # list of {from_version, to_version, rationale, changes}

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "AgentSpec":
        return cls(**json.loads(s))

    def clone_next_version(self) -> "AgentSpec":
        import copy
        new = copy.deepcopy(self)
        new.version += 1
        new.spec_id = str(uuid.uuid4())[:8]
        return new
