# Stem Agent — Agent-Executable Build Specification

> **How to use this document**
> You are an agent. Read every section in order. Each section tells you what to build, what inputs to consume, what outputs to produce, and what the acceptance criteria are. Do not skip sections. Do not reorder phases. When a section says STOP, stop and verify before proceeding.

---

## 0. Context and Goal

You are building a **stem agent**: a system that takes a task domain as input, automatically builds an evaluation question set for it, researches how that class of tasks is solved, designs itself an architecture, tests it, improves it iteratively, and emits a finalised specialised agent when it converges.

The output is NOT the stem agent itself. The output is the **specialised agent it grew into**.

The domain is configurable at runtime via `--domain`. There is no fixed domain baked into the code.

**Stack:**
- Python 3.11+
- **LiteLLM** (`litellm`) as the universal LLM wrapper — no provider SDK is imported directly
- **uv** as the package manager (`pyproject.toml`, not `requirements.txt`)
- `python-dotenv` for `.env` loading
- `ddgs` (DuckDuckGo search), `requests`, `beautifulsoup4` for Phase 0 web evidence gathering
- `tavily-python` (optional, for live web search inside the runner)

---

## 1. Repository Layout

Create the following structure before writing any logic:

```
stem_agent/
├── README.md
├── pyproject.toml
├── stem_agent/
│   ├── __init__.py
│   ├── main.py                   # entry point: uv run python -m stem_agent.main
│   ├── phases/
│   │   ├── groundtruth.py        # Phase 0: build or load eval question set
│   │   ├── sense.py              # Phase 1
│   │   ├── hypothesize.py        # Phase 2
│   │   ├── differentiate.py      # Phase 3 (the loop)
│   │   └── crystallize.py        # Phase 4
│   ├── core/
│   │   ├── agent_spec.py         # AgentSpec dataclass + serialisation
│   │   ├── llm.py                # thin LiteLLM wrapper (call_llm)
│   │   ├── paths.py              # PROJECT_ROOT anchor
│   │   ├── runner.py             # runs a candidate agent against one question
│   │   └── checkpointer.py      # save/restore/rollback AgentSpec versions
│   └── eval/
│       └── harness.py            # runs full eval suite + scores
├── eval_suite/
│   └── <domain_slug>.json        # auto-generated per domain (Phase 0 output)
└── outputs/
    └── <domain_slug>/
        ├── checkpoints/          # one JSON per checkpoint
        ├── final_agent.json
        ├── eval_results.json
        └── eval_by_tier.json
```

All files inside `stem_agent/` must have a corresponding `__init__.py`. Create all directories and empty `__init__.py` files before writing logic.

---

## 2. Package Manifest

**File:** `pyproject.toml`

```toml
[project]
name = "stem-agent"
version = "0.1.0"
description = "A stem agent that grows into a specialized agent via self-differentiation"
requires-python = ">=3.11"
dependencies = [
    "litellm>=1.40.0",
    "python-dotenv>=1.0.0",
    "ddgs>=7.0.0",
    "requests>=2.31.0",
    "beautifulsoup4>=4.12.0",
    "langchain>=1.3.0",
    "langchain-community>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-litellm>=0.6.5",
]

[project.optional-dependencies]
search = [
    "tavily-python>=0.3.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["stem_agent"]
```

Run `uv sync` after creating this file and after any dependency change.

---

## 3. Core Infrastructure

### 3a. Path anchor

**File:** `stem_agent/core/paths.py`

```python
from pathlib import Path

# stem_agent/stem_agent/core/paths.py → up three levels = project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
```

Import `PROJECT_ROOT` everywhere a file path is constructed. Never use bare relative strings like `"outputs/..."` — always prefix with `PROJECT_ROOT /`. This ensures the CLI works correctly regardless of the working directory from which it is invoked.

### 3b. LLM wrapper

**File:** `stem_agent/core/llm.py`

All LLM calls go through two functions: `call_llm(messages, max_tokens) -> str` for plain text and `call_llm_json(messages, max_tokens) -> dict` for structured JSON output. No module may import a provider SDK directly. The underlying engine is **LangChain's `ChatLiteLLM`**, which delegates to `litellm` and therefore supports every provider LiteLLM supports.

Reasoning/thinking models (OpenAI `o1`, `o3`, `o4`, `gpt-5.x` series) use a shared thinking+output token budget. The wrapper automatically detects them and scales up the token request so internal chain-of-thought does not exhaust the quota before the visible answer is written.

```python
import json
import os

from langchain_litellm import ChatLiteLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# Alias alternative env var name before any initialisation
if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]

DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

# Substrings that identify reasoning/thinking models
_REASONING_MODEL_SUBSTRINGS = ("o1", "o3", "o4", "gpt-5", "gpt5")
_REASONING_MIN_TOKENS = 16_000   # floor for reasoning models
_REASONING_MULTIPLIER = 4        # scale caller budget × 4

_json_parser = JsonOutputParser()


def _get_model() -> str:
    """Priority: STEM_AGENT_MODEL → MODEL → DEFAULT_MODEL."""
    return (
        os.environ.get("STEM_AGENT_MODEL")
        or os.environ.get("MODEL")
        or DEFAULT_MODEL
    )


def _is_reasoning_model(model: str) -> bool:
    return any(s in model.lower() for s in _REASONING_MODEL_SUBSTRINGS)


def _to_lc_messages(messages: list[dict]) -> list:
    """Convert OpenAI-style dicts to LangChain message objects."""
    mapping = {"system": SystemMessage, "assistant": AIMessage}
    return [mapping.get(m["role"], HumanMessage)(content=m["content"]) for m in messages]


def get_llm(max_tokens: int = 2048) -> ChatLiteLLM:
    """Return a configured ChatLiteLLM instance, with budget scaling for reasoning models."""
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
    """Call the LLM and return the response as a plain string."""
    llm = get_llm(max_tokens)
    response = llm.invoke(_to_lc_messages(messages))
    return response.content


def call_llm_json(messages: list[dict], max_tokens: int = 2048) -> dict:
    """
    Call the LLM and parse the response as JSON.
    Handles markdown fences automatically via JsonOutputParser.
    Falls back to a manual repair pass (trim truncated trailing text) before raising.
    """
    raw = call_llm(messages, max_tokens)
    try:
        return _json_parser.parse(raw)
    except (OutputParserException, Exception):
        # Repair: find the last closing brace and truncate
        last_brace = raw.rfind("}")
        if last_brace != -1:
            try:
                return json.loads(raw[: last_brace + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"call_llm_json: could not parse response as JSON.\nRaw:\n{raw[:500]}")
```

Supported environment variables (any combination works):

| Variable | Purpose |
|---|---|
| `STEM_AGENT_MODEL` | Primary model override |
| `MODEL` | Secondary model override (alternative name) |
| `OPENAI_API_KEY` | OpenAI / OpenAI-compatible key |
| `OPENAI_KEY` | Alternative name; automatically aliased to `OPENAI_API_KEY` |
| `ANTHROPIC_API_KEY` | Required for Claude models |
| `TAVILY_API_KEY` | Optional; enables live web search in `multi_step_search` |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional; raises Semantic Scholar rate limits in Phase 0 |

> **Important:** `call_llm` returns plain text. Use `call_llm_json` whenever the response is expected to be a JSON object (Phase 1, 2, 3 introspect/mutate, and the eval judge). Both functions are exported from `core/llm.py`.

---

## 4. Core Data Structure: AgentSpec

**File:** `stem_agent/core/agent_spec.py`

`AgentSpec` is the central artefact that the stem agent reads, writes, and mutates. It is a plain dataclass serialisable to JSON. The differentiation loop mutates this object; checkpointing saves versions of it.

```python
from dataclasses import dataclass, field, asdict
from typing import Optional
import json, uuid, datetime

@dataclass
class AgentSpec:
    """
    Identity
    --------
    spec_id   : short unique ID for this version
    version   : incremented on every accepted mutation
    task_domain: the domain this agent is specialised for

    Architecture (mutated by Phase 3)
    -----------------------------------
    system_prompt      : injected as the system message on every LLM call
    architecture_type  : "linear_chain" | "reflection_loop" | "multi_step_search" | "adversarial_qa"
    search_strategy    : "breadth_first" | "depth_first" | "iterative_refinement"
    synthesis_strategy : "extractive" | "abstractive" | "hybrid"
    max_search_rounds  : how many search/refinement rounds the runner executes
    self_critique      : whether the runner adds a self-critique pass
    citation_style     : "inline" | "footnote" | "none"
    tools              : list of tool names the runner may invoke

    Bookkeeping (append-only)
    --------------------------
    eval_scores  : list of {version, scores, timestamp} dicts
    mutation_log : list of {from_version, to_version, rationale, changes} dicts
    """
    spec_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    version: int = 0
    task_domain: str = "Deep Research"
    system_prompt: str = ""
    architecture_type: str = ""
    search_strategy: str = ""
    synthesis_strategy: str = ""
    max_search_rounds: int = 3
    self_critique: bool = False
    citation_style: str = "inline"
    tools: list = field(default_factory=list)
    eval_scores: list = field(default_factory=list)
    mutation_log: list = field(default_factory=list)

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
```

**Acceptance criterion:** `AgentSpec` round-trips through `to_json()` / `from_json()` without data loss. Write a 5-line test and run it before proceeding.

---

## 5. Phase 0 — Ground Truth Builder

**File:** `stem_agent/phases/groundtruth.py`

Phase 0 builds the eval question set for the target domain. It is fully automatic and domain-agnostic.

### 5a. Resolution order

1. `--questions <path>` was supplied by the user → load that file directly; skip Phase 0.
2. `eval_suite/<domain_slug>.json` already exists → log "reusing cached ground truth" and load it.
3. Neither → gather evidence, generate questions, write the file atomically, return the list.

### 5b. Evidence sources

Gather evidence from four sources in order. Each source wraps its network calls in a `try/except` that logs a warning and returns `[]` on failure — never crash.

| Source | Implementation |
|---|---|
| **DuckDuckGo** | `from ddgs import DDGS` — run 4 targeted queries (`domain`, `"{domain} fundamentals"`, `"{domain} latest advances 2024 2025"`, `"{domain} challenges"`) with `max_results=5` each. Add 0.5 s sleep between queries. |
| **arXiv** | `GET https://export.arxiv.org/api/query` with `search_query=all:{domain} AND (ti:survey OR ti:review)`, `max_results=6`. Parse XML with `BeautifulSoup(..., "xml")` to extract title, abstract (≤400 chars), and link. |
| **Semantic Scholar** | `GET https://api.semanticscholar.org/graph/v1/paper/search` with `fields=title,abstract,year,externalIds`, `limit=5`. If `SEMANTIC_SCHOLAR_API_KEY` is set, inject it as `x-api-key` header. |
| **Wikipedia** | `GET https://en.wikipedia.org/api/rest_v1/page/summary/{title}` — try the full domain string, then the first word as fallback. Extract `extract` (≤600 chars). |

Combine all results into a `context` string (≤12 000 chars) sectioned by source type.

### 5c. Question generation

Single `call_llm` call with `max_tokens=16000`:

```
GENERATE_PROMPT = """You are building a high-quality evaluation question set for a research agent
specialising in "{domain}".

Use ONLY facts that appear in the evidence below or are universally established. Any claim that
is speculative or contested must have its key_fact prefixed with "[disputed]".

SOURCE EVIDENCE:
{context}

Generate exactly 15 research questions about "{domain}" distributed as:
- 5 easy   (foundational concepts, well-established facts)
- 6 medium (mechanisms, tradeoffs, history, applications)
- 4 hard   (frontier research, open debates, critical evaluation of competing theories)

Each question must have:
- "question": a clear, self-contained research question (not yes/no)
- "ground_truth": 2-4 sentences that constitute a correct, grounded answer
- "key_facts": 3-6 specific facts the answer must contain; prefix speculative ones with "[disputed]"
- "sources": list of source objects used, each with "type" (web|arxiv|semantic_scholar|wikipedia|parametric)
  and "url" when available, or {"type": "parametric", "note": "LLM parametric knowledge"} otherwise

Respond with ONLY a valid JSON array of 15 objects with keys: id, tier, question, ground_truth, key_facts, sources
No preamble, no markdown fences, no trailing text."""
```

#### JSON parse strategy

1. **Primary**: `json.loads(raw)` after stripping markdown fences and trailing text.
2. **Fallback** (if truncated): character-by-character scan that extracts every complete `{…}` object that parses cleanly. Log `WARNING: JSON parse failed; attempting partial recovery…` and the count recovered. Only raise if zero objects could be salvaged.

### 5d. Atomic write

```python
tmp_fd, tmp_path = tempfile.mkstemp(dir=PROJECT_ROOT / "eval_suite", suffix=".json.tmp")
with os.fdopen(tmp_fd, "w") as f:
    json.dump(questions, f, indent=2)
os.replace(tmp_path, out_path)
```

### 5e. Public interface

```python
def build_ground_truth(domain: str, domain_slug: str) -> list[dict]:
    """Gather evidence, generate 15 questions, write atomically, return list."""

def load_or_build(domain: str, domain_slug: str) -> list[dict]:
    """Return cached questions if the file exists, else call build_ground_truth."""
```

---

## 6. Phase 1 — Sense

**File:** `stem_agent/phases/sense.py`

**Input:** `task_domain: str`
**Output:** `primitives: dict`

Single `call_llm_json` call. Ask the model to act as a research analyst and produce a structured JSON breakdown of how the domain is typically approached.

```python
SENSE_PROMPT = """You are analyzing how the task domain '{domain}' is typically approached
by expert systems and human researchers.

Produce a structured JSON analysis with exactly these keys:

{{
  "common_steps": ["list of 5-8 steps that typically appear in this task class"],
  "architecture_patterns": ["list of 3-5 named architecture patterns (e.g. 'reflection loop')"],
  "tools_commonly_used": ["list of tools/APIs/capabilities typically needed"],
  "evaluation_criteria": ["list of 4-6 criteria by which outputs are judged"],
  "known_failure_modes": ["list of 3-5 common ways these agents fail"],
  "key_design_decisions": ["list of 3-5 architectural choices that most affect quality"]
}}

Respond with ONLY the JSON object. No preamble, no markdown fences."""
```

**Acceptance criterion:** `primitives` is a valid dict with all six keys populated. STOP and verify before Phase 2.

---

## 7. Phase 2 — Hypothesize

**File:** `stem_agent/phases/hypothesize.py`

**Input:** `primitives: dict`, `task_domain: str`
**Output:** `AgentSpec` v0

Single `call_llm_json` call. Inject `primitives` and the domain name. Stamp `spec.task_domain = task_domain`.

```python
HYPOTHESIZE_PROMPT = """You are designing the initial architecture for a '{domain}' agent.

Based on this analysis of how the domain is typically approached:
{primitives_json}

Choose initial values for an agent spec. Only use the allowed values listed.

Respond with ONLY a JSON object:
{{
  "system_prompt": "2-4 sentence system prompt focused on the domain",
  "architecture_type": one of ["linear_chain", "reflection_loop", "multi_step_search", "adversarial_qa"],
  "search_strategy": one of ["breadth_first", "depth_first", "iterative_refinement"],
  "synthesis_strategy": one of ["extractive", "abstractive", "hybrid"],
  "max_search_rounds": integer 2-6,
  "self_critique": true or false,
  "citation_style": one of ["inline", "footnote", "none"],
  "tools": subset of ["web_search", "document_reader", "calculator", "code_executor"],
  "rationale": "2-3 sentences explaining this combination"
}}

No preamble. No markdown fences. Only the JSON."""
```

**Acceptance criterion:** `AgentSpec` is fully populated with valid enum values; `spec.to_json()` serialises cleanly. STOP and verify.

---

## 8. Phase 3 — Differentiate (the loop)

**File:** `stem_agent/phases/differentiate.py`

### 8a. Signature

```python
def differentiate(
    initial_spec: AgentSpec,
    initial_score: dict,
    questions: list[dict],
    output_dir: str = "outputs",
) -> tuple[AgentSpec, dict]:
```

### 8b. Loop parameters

```python
MAX_ITERATIONS = 8
CONVERGENCE_DELTA = 0.02
MIN_ITERATIONS = 3
```

### 8c. Loop structure (implement exactly)

```
best_spec = initial_spec
best_score = initial_score
consecutive_small_improvements = 0

for iteration in 1..MAX_ITERATIONS:
    failure_analysis = introspect(best_spec, best_score)
    candidate_spec   = propose_mutation(best_spec, failure_analysis)
    candidate_score  = run_eval(candidate_spec, questions)
    delta = candidate_score['aggregate']['composite'] - best_score['aggregate']['composite']

    if delta >= 0:
        best_spec, best_score = candidate_spec, candidate_score
        save_checkpoint(best_spec, best_score, iteration, output_dir)
        consecutive_small_improvements = (consecutive_small_improvements + 1) if delta < CONVERGENCE_DELTA else 0
    # else: keep best_spec unchanged (implicit rollback)

    if iteration >= MIN_ITERATIONS and consecutive_small_improvements >= 2:
        break

return best_spec, best_score
```

### 8d. Introspect prompt

```python
INTROSPECT_PROMPT = """A '{domain}' agent with the following spec produced these evaluation results.

AGENT SPEC:
{spec_json}

EVAL RESULTS (per-question scores):
{eval_summary}

Worst-performing questions:
{worst_questions}

Identify the 2-3 most likely root causes of underperformance. Be specific about which spec
fields are likely responsible.

Respond with ONLY a JSON object:
{{
  "root_causes": ["cause 1", "cause 2", "cause 3"],
  "fields_to_change": ["field_name_1", "field_name_2"],
  "rationale": "one paragraph"
}}"""
```

For `worst_questions` pass the 3 questions with lowest composite score, including per-dimension scores and answer text.

### 8e. Mutate prompt

```python
MUTATE_PROMPT = """Propose a targeted improvement to a '{domain}' agent spec.

CURRENT SPEC:
{spec_json}

FAILURE ANALYSIS:
{failure_analysis_json}

Propose exactly one mutation — change 1-3 fields. You MUST:
- Only change fields listed in failure_analysis.fields_to_change
- Stay within allowed values
- Explain why this specific change fixes the identified root cause

Respond with ONLY a JSON object:
{{
  "changes": {{"field_name": "new_value", ...}},
  "expected_improvement": "one sentence",
  "rationale": "2-3 sentences"
}}"""
```

Apply changes via `spec.clone_next_version()`. Append to `spec.mutation_log`.

---

## 9. Checkpointer

**File:** `stem_agent/core/checkpointer.py`

```python
from stem_agent.core.paths import PROJECT_ROOT

def save_checkpoint(spec: AgentSpec, score: dict, iteration: int, output_dir: str = "outputs") -> None:
    path = Path(output_dir) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    filename = path / f"v{spec.version}_iter{iteration}_score{score['aggregate']['composite']:.3f}.json"
    payload = {"spec": json.loads(spec.to_json()), "score": score, "iteration": iteration}
    filename.write_text(json.dumps(payload, indent=2))

def load_best_checkpoint(output_dir: str) -> tuple[AgentSpec, dict] | None:
    """Scan all checkpoints under output_dir and return (spec, score) for the highest composite."""
    ckpt_dir = Path(output_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    best = None
    best_composite = -1.0
    for f in ckpt_dir.glob("*.json"):
        data = json.loads(f.read_text())
        composite = data["score"]["aggregate"]["composite"]
        if composite > best_composite:
            best_composite = composite
            best = (AgentSpec.from_json(json.dumps(data["spec"])), data["score"])
    return best
```

---

## 10. The Candidate Agent Runner

**File:** `stem_agent/core/runner.py`

```python
from stem_agent.core.llm import call_llm

def run_candidate_agent(spec: AgentSpec, question: str) -> str:
    """Dispatch to the correct execution strategy. Returns the agent's answer."""
```

Implement four private functions — use `call_llm` (plain text) directly:

| Architecture | Implementation |
|---|---|
| `linear_chain` | Single `call_llm` with `spec.system_prompt`. |
| `reflection_loop` | Two calls — generate answer, then critique and revise. Falls back to `linear_chain` if `spec.self_critique == False`. |
| `multi_step_search` | `spec.max_search_rounds` calls; each may invoke Tavily if `"web_search"` in `spec.tools` and `TAVILY_API_KEY` is set. Synthesise a final answer from round outputs. |
| `adversarial_qa` | Three calls — "for" position, "against/counterpoint" position, synthesis. Best for comparative questions. |

---

## 11. The Eval Harness

**File:** `stem_agent/eval/harness.py`

```python
def run_eval(spec: AgentSpec, questions: list[dict]) -> dict:
    """
    Returns:
    {
        "per_question": [{"id": "q01", "tier": "easy", "scores": {...}, "composite": 0.0, "answer": "..."}, ...],
        "aggregate": {"factual_accuracy": 0.0, "coverage": 0.0, "coherence": 0.0, "source_diversity": 0.0, "composite": 0.0},
        "version": spec.version,
        "timestamp": "ISO-8601"
    }
    """
```

Score each answer on four dimensions (0.0–1.0) via a judge `call_llm_json` call (import it from `core/llm.py`). The judge prompt requests JSON directly so parsing is always structured:

| Dimension | What it measures |
|---|---|
| `factual_accuracy` | Key facts from ground truth that appear correctly |
| `coverage` | Fraction of `key_facts` addressed |
| `coherence` | Well-structured and internally consistent |
| `source_diversity` | Synthesises from multiple angles |

**Composite** = mean of four dimensions.

Judge prompt template:

```python
JUDGE_PROMPT = """You are a research quality evaluator.

QUESTION: {question}
GROUND TRUTH SUMMARY: {ground_truth}
KEY FACTS TO LOOK FOR: {key_facts}

ANSWER TO EVALUATE:
{answer}

Score each dimension 0.0-1.0:
- factual_accuracy: Are the facts correct? Do key facts from ground truth appear?
- coverage: What fraction of the key facts above does the answer address?
- coherence: Is the answer well-structured, logically consistent, and readable?
- source_diversity: Does it present multiple perspectives or synthesise from multiple angles?

Respond with ONLY a JSON object:
{{"factual_accuracy": 0.0, "coverage": 0.0, "coherence": 0.0, "source_diversity": 0.0, "reasoning": "one sentence"}}"""
```

Run questions sequentially. Add 1-second sleep between judge calls. Log each result to stdout as it completes.

---

## 12. Phase 4 — Crystallize

**File:** `stem_agent/phases/crystallize.py`

```python
def crystallize(best_spec: AgentSpec, best_score: dict, initial_score: dict, output_dir: str = "outputs") -> None:
```

Writes three files to `output_dir`:

1. **`final_agent.json`** — `best_spec.to_json()`
2. **`eval_results.json`** — before/after comparison:
   ```json
   {
     "before": { ...aggregate scores... },
     "after":  { ...aggregate scores... },
     "delta":  { ...per-dimension difference... },
     "iterations_run": best_spec.version,
     "mutation_log": [...]
   }
   ```
3. **`eval_by_tier.json`** — per-tier breakdown:
   ```json
   {
     "easy":   {"factual_accuracy": 0.0, "coverage": 0.0, "coherence": 0.0, "source_diversity": 0.0, "composite": 0.0, "n": 8},
     "medium": {...},
     "hard":   {...}
   }
   ```

Print a summary table to stdout on completion.

---

## 13. Main Entry Point

**File:** `stem_agent/main.py`

### 13a. CLI

```python
parser.add_argument("--domain",    default="Deep Research")
parser.add_argument("--questions", default=None,
    help="Path to questions JSON. When omitted, Phase 0 auto-generates or reuses cache.")
parser.add_argument("--resume",    action="store_true",
    help="Skip Phases 1-2, load best checkpoint, continue differentiation.")
```

### 13b. Phase 0 resolution block (always runs first)

```python
domain_slug = re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_")

if args.questions:
    questions = json.loads(Path(args.questions).read_text())
else:
    from stem_agent.phases.groundtruth import load_or_build
    questions = load_or_build(domain, domain_slug)

output_dir = PROJECT_ROOT / "outputs" / domain_slug
output_dir.mkdir(parents=True, exist_ok=True)
```

All file paths (`output_dir`, `eval_suite/`) must use `PROJECT_ROOT` so the CLI works from any working directory.

### 13c. Main flow

```
Phase 0  → resolve question set (always)
Phase 1  → sense(domain)                           ┐ skipped if --resume
Phase 2  → hypothesize(primitives, domain)         ┘ and checkpoint exists
           baseline eval
Phase 3  → differentiate(initial_spec, initial_score, questions, output_dir)
Phase 4  → crystallize(best_spec, best_score, initial_score, output_dir)
```

If `--resume` is set but no checkpoint exists, log a warning and fall through to the normal Phases 1–2 path.

---

## 14. Build Order

Execute in this exact sequence:

1. Create directory structure and all `__init__.py` files
2. Create `pyproject.toml`; run `uv sync`
3. Implement `core/paths.py`
4. Implement `core/llm.py`; smoke-test: `call_llm([{"role":"user","content":"ping"}])` returns a string and `call_llm_json([{"role":"user","content":"respond with {\"ok\":true}"}])` returns a dict
5. Implement `core/agent_spec.py`; round-trip test
6. Implement `core/runner.py` (all four architectures)
7. Implement `core/checkpointer.py`
8. Implement `eval/harness.py`
9. Implement `phases/sense.py`
10. Implement `phases/hypothesize.py`
11. Implement `phases/differentiate.py`
12. Implement `phases/crystallize.py`
13. Implement `phases/groundtruth.py`; smoke-test: `load_or_build("Code Review", "code_review")` returns a list
14. Wire `main.py`
15. Full run: `uv run python -m stem_agent.main --domain "Code Review"`

**Do not start step N+1 until step N produces the expected output.**

---

## 15. Acceptance Criteria

The submission passes if:

- [ ] `uv run python -m stem_agent.main --domain "Code Review"` runs to completion from any directory
- [ ] `eval_suite/code_review.json` exists with 15 questions (Phase 0 output)
- [ ] `outputs/code_review/final_agent.json` exists and is valid `AgentSpec` JSON
- [ ] `outputs/code_review/eval_results.json` shows a before/after comparison with different composite scores
- [ ] `outputs/code_review/eval_by_tier.json` shows per-tier (easy/medium/hard) breakdown
- [ ] At least one rollback is visible in checkpoints (a candidate composite lower than best was not saved as new best)
- [ ] `uv run python -m stem_agent.main --domain "Code Review" --resume` reuses the cached questions and checkpoint
- [ ] `uv run python -m stem_agent.main --domain "Code Review" --questions custom.json` uses the supplied file and skips Phase 0


---

