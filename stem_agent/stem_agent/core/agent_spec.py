from dataclasses import dataclass, field, asdict
from typing import Optional
import json, uuid, datetime


@dataclass
class AgentSpec:
    """
    Central artifact of the stem agent system.  Every phase reads or writes this
    object; the differentiation loop mutates it; checkpointing saves versions of it.

    The dataclass is intentionally flat and JSON-serialisable so that checkpoints
    are human-readable and can be diffed between iterations.

    Fields
    ------
    Identity
    ~~~~~~~~
    spec_id : str
        Random 8-character ID unique to this exact version of the spec.
        Regenerated every time ``clone_next_version()`` is called, so two
        consecutive versions have different IDs.
    version : int
        Integer counter starting at 0.  Incremented by 1 each time the
        differentiation loop accepts a mutation.  Lets you track how many
        improvements were applied.
    task_domain : str
        The problem class this agent is being grown for — e.g. ``"Deep Research"``
        or ``"Code Review"``.  Comes from ``--domain`` on the CLI and is stamped
        here so checkpoints are self-describing.

    Architecture (mutated by Phase 3)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    system_prompt : str
        The instructions prepended to every LLM call the candidate agent makes.
        Controls tone, thoroughness, and domain focus.
    architecture_type : str
        How the agent structures its reasoning.

        * ``"linear_chain"`` — single call, no iteration.
        * ``"reflection_loop"`` — answer, then self-critique and revise.
        * ``"multi_step_search"`` — multiple rounds of deepening search before synthesis.
        * ``"adversarial_qa"`` — argue both sides then synthesise a balanced answer.
    search_strategy : str
        Within ``multi_step_search``, how to allocate rounds across the question.

        * ``"breadth_first"`` — cover many sub-topics.
        * ``"depth_first"`` — dig deeply into one aspect.
        * ``"iterative_refinement"`` — each round fills gaps from the previous one.
    synthesis_strategy : str
        How the final answer is assembled from gathered material.

        * ``"extractive"`` — pull key facts verbatim.
        * ``"abstractive"`` — rewrite in own words.
        * ``"hybrid"`` — mix of both.
    max_search_rounds : int
        How many LLM passes ``multi_step_search`` makes before synthesising.
        Range 2–6.  Higher means more thorough but more expensive.
    self_critique : bool
        If ``True``, the ``reflection_loop`` architecture adds a second pass where
        the agent reads its own answer and produces a revised version.
        Has no effect on other architecture types.
    citation_style : str
        How the agent references its sources in the final answer.

        * ``"inline"`` — cite within the sentence.
        * ``"footnote"`` — numbered references at the end.
        * ``"none"`` — no citations.
    tools : list[str]
        Capability names the agent is allowed to use, e.g. ``["web_search"]``.
        Currently gates whether Tavily search is invoked inside
        ``multi_step_search``.

    Bookkeeping (append-only, never mutated by differentiation)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    eval_scores : list[dict]
        History of eval results attached to this spec across runs.
        Each entry is ``{version, scores, timestamp}``.
    mutation_log : list[dict]
        Audit trail of every change the differentiation loop proposed and
        accepted.  Each entry records ``from_version``, ``to_version``, the
        actual field ``changes``, and the LLM's ``rationale``.  Useful for
        the write-up and for debugging why the agent evolved the way it did.
    """

    # --- Identity ---
    spec_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    version: int = 0
    task_domain: str = "Deep Research"

    # --- Architecture choices (mutated by Phase 3) ---
    system_prompt: str = ""
    architecture_type: str = ""   # "linear_chain" | "reflection_loop" | "multi_step_search" | "adversarial_qa"
    search_strategy: str = ""     # "breadth_first" | "depth_first" | "iterative_refinement"
    synthesis_strategy: str = ""  # "extractive" | "abstractive" | "hybrid"
    max_search_rounds: int = 3
    self_critique: bool = False
    citation_style: str = "inline"  # "inline" | "footnote" | "none"

    # --- Tools ---
    tools: list = field(default_factory=list)

    # --- Bookkeeping (append-only) ---
    eval_scores: list = field(default_factory=list)  # list of dicts: {version, scores, timestamp}
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
