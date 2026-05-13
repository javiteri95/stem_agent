# Stem Agent — Agent-Executable Build Specification

> **How to use this document**
> You are an agent. Read every section in order. Each section tells you what to build, what inputs to consume, what outputs to produce, and what the acceptance criteria are. Do not skip sections. Do not reorder phases. When a section says STOP, stop and verify before proceeding.

---

## 0. Context and Goal

You are building a **stem agent**: a system that takes a class of tasks as input, researches how that class is solved, designs itself an architecture, tests it, improves it iteratively, and emits a finalized specialized agent when it converges.

The output is NOT the stem agent itself. The output is the **specialized agent it grew into**.

**Fixed domain for this implementation:** Deep Research  
Reason: outputs are measurable (factual accuracy, coverage, coherence, source diversity), good baselines exist, and architectural variation is meaningful (search strategies, synthesis approaches, citation handling differ substantially across implementations).

**Stack:** Python 3.11+, Anthropic Python SDK (`anthropic`), optionally `duckduckgo-search` or `tavily-python` for web search tool access inside the stem agent.

---

## 1. Repository Layout

Create the following directory structure before writing any logic:

```
stem_agent/
├── README.md                  # setup + run instructions (write last)
├── requirements.txt
├── stem_agent/
│   ├── __init__.py
│   ├── main.py                # entry point: python -m stem_agent.main
│   ├── phases/
│   │   ├── sense.py           # Phase 1
│   │   ├── hypothesize.py     # Phase 2
│   │   ├── differentiate.py   # Phase 3 (the loop)
│   │   └── crystallize.py     # Phase 4
│   ├── core/
│   │   ├── agent_spec.py      # AgentSpec dataclass + serialization
│   │   ├── runner.py          # runs a candidate agent against one question
│   │   └── checkpointer.py    # save/restore/rollback AgentSpec versions
│   └── eval/
│       └── harness.py         # runs full eval suite + scores
├── eval_suite/
│   └── questions.json         # 10 research questions with ground truth
├── outputs/
│   ├── checkpoints/           # one JSON per checkpoint
│   ├── final_agent.json       # the crystallized agent spec
│   └── eval_results.json      # before/after scores
└── writeup/
    └── writeup.md             # 4-page write-up (write last)
```

Create all directories and empty `__init__.py` files before writing logic.

---

## 2. Core Data Structure: AgentSpec

**File:** `stem_agent/core/agent_spec.py`

`AgentSpec` is the central artifact that the stem agent reads, writes, and mutates. It is a plain dataclass serializable to JSON. The differentiation loop mutates this object; checkpointing saves versions of it.

```python
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
    citation_style: str = "inline" # "inline" | "footnote" | "none"
    
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
```

**Acceptance criteria:** `AgentSpec` round-trips through `to_json()` / `from_json()` without data loss. Write a 5-line test and run it before proceeding.

---

## 3. The Eval Suite

**File:** `eval_suite/questions.json`

Create 10 research questions across three difficulty tiers. Each question has a ground truth summary (1–3 sentences) that the scorer uses as a reference. Questions must span different sub-domains within Deep Research.

```json
[
  {
    "id": "q01",
    "tier": "easy",
    "question": "What is retrieval-augmented generation (RAG) and why was it introduced?",
    "ground_truth": "RAG is a technique that augments language model generation by first retrieving relevant documents from an external corpus, then conditioning generation on those documents. It was introduced to reduce hallucination and extend the model's effective knowledge beyond its training data without full fine-tuning.",
    "key_facts": ["retrieval before generation", "external corpus", "reduces hallucination", "no full retraining needed"]
  },
  {
    "id": "q02",
    "tier": "easy",
    "question": "What caused the 2008 financial crisis?",
    "ground_truth": "The 2008 crisis was triggered by the collapse of the US housing bubble, which had been inflated by subprime mortgage lending and securitization of those loans into mortgage-backed securities. When housing prices fell, defaults cascaded through the financial system, causing bank failures and a global credit freeze.",
    "key_facts": ["housing bubble", "subprime mortgages", "mortgage-backed securities", "bank failures", "credit freeze"]
  },
  {
    "id": "q03",
    "tier": "easy",
    "question": "How does mRNA vaccine technology work?",
    "ground_truth": "mRNA vaccines deliver synthetic messenger RNA encoding a target antigen (such as a viral spike protein) into cells. The cells translate the mRNA into protein, which the immune system recognizes and builds a response against. No live virus or DNA is involved, and the mRNA degrades quickly.",
    "key_facts": ["synthetic mRNA", "antigen encoding", "immune response", "no live virus", "mRNA degrades"]
  },
  {
    "id": "q04",
    "tier": "medium",
    "question": "What are the main tradeoffs between transformer-based and state-space models for sequence modeling?",
    "ground_truth": "Transformers have quadratic attention complexity with sequence length, enabling rich global context but at high compute cost. State-space models like Mamba achieve linear-time inference through selective state updates, but may trade off some expressivity on tasks requiring precise long-range interactions. The practical frontier is hybrid architectures combining both.",
    "key_facts": ["quadratic vs linear complexity", "global context", "Mamba / selective state updates", "hybrid architectures"]
  },
  {
    "id": "q05",
    "tier": "medium",
    "question": "Explain the history and current state of fusion energy research.",
    "ground_truth": "Fusion research began in the 1950s with tokamak designs. Key milestones include ITER (under construction) and NIF's 2022 ignition achievement. Private companies like Commonwealth Fusion and Helion are pursuing compact reactor designs. Net energy gain remains the key commercial milestone not yet achieved at scale.",
    "key_facts": ["tokamak", "ITER", "NIF ignition 2022", "private companies", "net energy gain"]
  },
  {
    "id": "q06",
    "tier": "medium",
    "question": "What is the current scientific consensus on the microbiome's role in mental health?",
    "ground_truth": "The gut-brain axis hypothesis proposes that gut microbiota influence brain function via vagus nerve signaling, metabolite production (including short-chain fatty acids and neurotransmitter precursors), and immune modulation. Evidence from animal studies is robust; human clinical evidence remains correlational and the field is still establishing causal mechanisms.",
    "key_facts": ["gut-brain axis", "vagus nerve", "SCFAs", "animal vs human evidence", "causal mechanisms unestablished"]
  },
  {
    "id": "q07",
    "tier": "medium",
    "question": "How did the Apollo Guidance Computer work, and why was it significant?",
    "ground_truth": "The AGC was a 16-bit real-time computer with 4KB RAM and 72KB ROM, using integrated circuits at a time when they were unproven. It ran a priority-scheduled OS (Executive) and was significant for pioneering software reliability techniques and demonstrating that ICs could be trusted for mission-critical applications.",
    "key_facts": ["16-bit", "4KB RAM", "integrated circuits", "real-time OS", "priority scheduling", "IC reliability"]
  },
  {
    "id": "q08",
    "tier": "hard",
    "question": "What are the strongest arguments for and against the simulation hypothesis?",
    "ground_truth": "Proponents cite the computational tractability of simulating reality and the Copernican principle (we are unlikely to be in base reality). Critics argue the hypothesis is unfalsifiable, relies on contestable assumptions about computational substrate, and does not escape the problem of explaining the origin of the simulating civilization.",
    "key_facts": ["computational tractability", "Copernican principle", "unfalsifiability", "substrate assumptions", "regress problem"]
  },
  {
    "id": "q09",
    "tier": "hard",
    "question": "Compare and critically evaluate the leading theories of consciousness (IIT, Global Workspace, Higher-Order).",
    "ground_truth": "IIT quantifies consciousness as integrated information (phi) and is mathematically precise but predicts consciousness in simple systems and resists empirical falsification. Global Workspace Theory posits a 'broadcast' mechanism for access consciousness, supported by neuroimaging but criticized for explaining access rather than phenomenal consciousness. Higher-Order theories require a meta-representation, making them computationally plausible but philosophically contested.",
    "key_facts": ["IIT / phi", "unfalsifiability of IIT", "GWT broadcast", "access vs phenomenal", "higher-order meta-representation"]
  },
  {
    "id": "q10",
    "tier": "hard",
    "question": "What were the root causes of the collapse of the Soviet Union, and which factors do historians most dispute?",
    "ground_truth": "Structural causes include economic stagnation, the arms race burden, and ethnic nationalism in constituent republics. Gorbachev's reforms (glasnost, perestroika) accelerated disintegration by loosening control before institutions could adapt. Historians dispute the relative weight of economic versus ideological versus contingent (Chernobyl, Afghanistan) factors.",
    "key_facts": ["economic stagnation", "arms race", "nationalism", "glasnost/perestroika", "Chernobyl", "disputed weighting"]
  }
]
```

---

## 4. The Eval Harness

**File:** `stem_agent/eval/harness.py`

The harness runs a candidate `AgentSpec` against all 10 questions and returns a score dict. It does NOT know about the stem agent — it only knows about an `AgentSpec` and the runner.

### 4a. Scoring dimensions

Score each answer on four dimensions, each 0.0–1.0, using a Claude API call as the judge:

| Dimension | What it measures |
|---|---|
| `factual_accuracy` | Key facts from ground truth that appear correctly in the answer |
| `coverage` | What fraction of `key_facts` are addressed |
| `coherence` | Is the answer well-structured and internally consistent? |
| `source_diversity` | Does it synthesize from multiple angles or repeat one perspective? |

**Composite score** = mean of four dimensions.

### 4b. Judge prompt template

```python
JUDGE_PROMPT = """You are a research quality evaluator. Score the following research answer.

QUESTION: {question}
GROUND TRUTH SUMMARY: {ground_truth}
KEY FACTS TO LOOK FOR: {key_facts}

ANSWER TO EVALUATE:
{answer}

Score on each dimension from 0.0 to 1.0:
- factual_accuracy: Are the facts in the answer correct? Do key facts from ground truth appear?
- coverage: What fraction of the key facts listed above does the answer address?
- coherence: Is the answer well-structured, logically consistent, and readable?
- source_diversity: Does it present multiple perspectives or synthesize from multiple angles?

Respond with ONLY a JSON object in this exact format:
{{"factual_accuracy": 0.0, "coverage": 0.0, "coherence": 0.0, "source_diversity": 0.0, "reasoning": "one sentence"}}"""
```

### 4c. Harness interface

```python
def run_eval(spec: AgentSpec, questions: list[dict], client: anthropic.Anthropic) -> dict:
    """
    Returns:
    {
        "per_question": [{"id": "q01", "scores": {...}, "composite": 0.0, "answer": "..."}, ...],
        "aggregate": {"factual_accuracy": 0.0, "coverage": 0.0, "coherence": 0.0, "source_diversity": 0.0, "composite": 0.0},
        "version": spec.version
    }
    """
```

Run questions sequentially (not parallel) to avoid rate limits. Add 1-second sleep between calls. Log each question result to stdout as it completes.

---

## 5. Phase 1 — Sense

**File:** `stem_agent/phases/sense.py`

**Input:** `task_domain: str` (e.g. `"Deep Research"`)  
**Output:** `primitives: dict` — a structured summary of how this task class is typically solved

**What to do:**

Call the Claude API with a prompt that asks it to act as a research analyst and produce a structured breakdown of how Deep Research tasks are typically approached. This is a single API call (no tool use needed in Phase 1 — Claude's parametric knowledge is sufficient for this domain).

```python
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
```

Parse the JSON response. Store the result as `primitives`. Log it to stdout. This is the stem agent's "reading signals from its environment."

**Acceptance criterion:** `primitives` is a valid dict with all six keys populated. STOP and verify before Phase 2.

---

## 6. Phase 2 — Hypothesize

**File:** `stem_agent/phases/hypothesize.py`

**Input:** `primitives: dict` (from Phase 1)  
**Output:** `AgentSpec` — version 0, populated with initial architecture choices

**What to do:**

Call the Claude API with the primitives and ask it to design an initial agent spec for the domain. The prompt must constrain the output to valid `AgentSpec` field values so you can deserialize it directly.

```python
HYPOTHESIZE_PROMPT = """You are designing the initial architecture for a Deep Research agent.

Based on this analysis of how the domain is typically approached:
{primitives_json}

Choose initial values for an agent spec. You MUST choose from the allowed values listed.

Respond with ONLY a JSON object with these fields:
{{
  "system_prompt": "a system prompt for the research agent (2-4 sentences, focused on the domain)",
  "architecture_type": one of ["linear_chain", "reflection_loop", "multi_step_search", "adversarial_qa"],
  "search_strategy": one of ["breadth_first", "depth_first", "iterative_refinement"],
  "synthesis_strategy": one of ["extractive", "abstractive", "hybrid"],
  "max_search_rounds": integer between 2 and 6,
  "self_critique": true or false,
  "citation_style": one of ["inline", "footnote", "none"],
  "tools": list of tool names from ["web_search", "document_reader", "calculator", "code_executor"],
  "rationale": "2-3 sentences explaining why you chose this combination"
}}

No preamble. No markdown fences. Only the JSON."""
```

Populate an `AgentSpec` from the response. Log the rationale. Save as checkpoint version 0.

**Acceptance criterion:** `AgentSpec` is fully populated with valid enum values. Run `spec.to_json()` and confirm it serializes cleanly. STOP and verify.

---

## 7. Phase 3 — Differentiate (the loop)

**File:** `stem_agent/phases/differentiate.py`

This is the core loop. It runs the eval harness, introspects failures, proposes mutations, checkpoints improvements, and rolls back regressions.

### 7a. Loop parameters

```python
MAX_ITERATIONS = 8        # hard budget
CONVERGENCE_DELTA = 0.02  # if improvement < this for 2 consecutive rounds, stop early
MIN_ITERATIONS = 3        # always run at least this many
```

### 7b. Loop structure (pseudocode — implement exactly)

```
best_spec = spec from Phase 2
best_score = run_eval(best_spec)
save_checkpoint(best_spec, best_score, iteration=0)
consecutive_small_improvements = 0

for iteration in 1..MAX_ITERATIONS:
    print(f"--- Iteration {iteration} ---")
    print(f"Current best composite: {best_score['aggregate']['composite']:.3f}")

    # Step 1: Introspect failures
    failure_analysis = introspect(best_spec, best_score)  # see 7c

    # Step 2: Propose mutation
    candidate_spec = propose_mutation(best_spec, failure_analysis)  # see 7d

    # Step 3: Eval candidate
    candidate_score = run_eval(candidate_spec)
    delta = candidate_score['aggregate']['composite'] - best_score['aggregate']['composite']
    print(f"Candidate composite: {candidate_score['aggregate']['composite']:.3f} (delta: {delta:+.3f})")

    # Step 4: Checkpoint or rollback
    if delta >= 0:
        best_spec = candidate_spec
        best_score = candidate_score
        save_checkpoint(best_spec, best_score, iteration)
        print("✓ Improvement accepted")
        if delta < CONVERGENCE_DELTA:
            consecutive_small_improvements += 1
        else:
            consecutive_small_improvements = 0
    else:
        print("✗ Regression — rolling back")
        # best_spec unchanged

    # Step 5: Convergence check
    if iteration >= MIN_ITERATIONS and consecutive_small_improvements >= 2:
        print(f"Converged at iteration {iteration}. Exiting loop.")
        break

return best_spec, best_score
```

### 7c. Introspect function

Calls Claude with the current spec and its eval results to identify what's going wrong.

```python
INTROSPECT_PROMPT = """A Deep Research agent with the following spec produced these evaluation results.

AGENT SPEC:
{spec_json}

EVAL RESULTS (per-question scores):
{eval_summary}

Worst-performing questions:
{worst_questions}

Identify the 2-3 most likely root causes of underperformance. Be specific about which spec fields are likely responsible.

Respond with ONLY a JSON object:
{{
  "root_causes": ["cause 1", "cause 2", "cause 3"],
  "fields_to_change": ["field_name_1", "field_name_2"],
  "rationale": "one paragraph"
}}"""
```

For `worst_questions`, pass the 3 questions with the lowest composite score, including their per-dimension scores and the answer text.

### 7d. Propose mutation function

Calls Claude to suggest a concrete mutation based on the failure analysis.

```python
MUTATE_PROMPT = """You are proposing a targeted improvement to a Deep Research agent spec.

CURRENT SPEC:
{spec_json}

FAILURE ANALYSIS:
{failure_analysis_json}

Propose exactly one mutation — change 1 to 3 fields of the spec. You MUST:
- Only change fields listed in failure_analysis.fields_to_change
- Stay within allowed values for each field (see original spec constraints)
- Explain why this specific change should fix the identified root cause

Respond with ONLY a JSON object:
{{
  "changes": {{"field_name": "new_value", ...}},
  "expected_improvement": "one sentence",
  "rationale": "2-3 sentences"
}}"""
```

Apply changes to a cloned `AgentSpec` (use `spec.clone_next_version()`). Append to `spec.mutation_log`.

### 7e. Checkpointer

**File:** `stem_agent/core/checkpointer.py`

```python
def save_checkpoint(spec: AgentSpec, score: dict, iteration: int, path: str = "outputs/checkpoints/"):
    filename = f"{path}v{spec.version}_iter{iteration}_score{score['aggregate']['composite']:.3f}.json"
    payload = {"spec": spec.to_json(), "score": score, "iteration": iteration}
    # write to file
```

---

## 8. The Candidate Agent Runner

**File:** `stem_agent/core/runner.py`

The runner takes an `AgentSpec` and a single research question, executes the agent according to the spec's architecture, and returns a string answer.

```python
def run_candidate_agent(spec: AgentSpec, question: str, client: anthropic.Anthropic) -> str:
    """
    Dispatches to the right execution strategy based on spec.architecture_type.
    Returns the agent's answer as a plain string.
    """
    if spec.architecture_type == "linear_chain":
        return _run_linear_chain(spec, question, client)
    elif spec.architecture_type == "reflection_loop":
        return _run_reflection_loop(spec, question, client)
    elif spec.architecture_type == "multi_step_search":
        return _run_multi_step_search(spec, question, client)
    elif spec.architecture_type == "adversarial_qa":
        return _run_adversarial_qa(spec, question, client)
    else:
        raise ValueError(f"Unknown architecture_type: {spec.architecture_type}")
```

Implement each architecture as a separate private function. Minimum viable implementations:

**`_run_linear_chain`:** Single API call with `spec.system_prompt`. No iterations.

**`_run_reflection_loop`:** Two calls — first generates an answer, second critiques it and produces a revised answer. Only active if `spec.self_critique == True`, otherwise falls back to linear.

**`_run_multi_step_search`:** Calls Claude `spec.max_search_rounds` times. Each call can request web search tool use if `"web_search"` is in `spec.tools`. Synthesizes a final answer from round outputs.

**`_run_adversarial_qa`:** Two calls — one takes the "for" position on the question, one takes the "against" or "counterpoint" position. A third call synthesizes a balanced answer. Best for comparative or evaluative questions.

> **Note on web search:** If you have access to a web search tool (e.g. Tavily), wire it into the `multi_step_search` architecture. If not, the architecture still works — Claude uses parametric knowledge across multiple rounds, which is sufficient for the eval suite questions.

---

## 9. Phase 4 — Crystallize

**File:** `stem_agent/phases/crystallize.py`

**Input:** `best_spec: AgentSpec`, `best_score: dict`  
**Output:** Final JSON files written to `outputs/`

```python
def crystallize(best_spec: AgentSpec, best_score: dict, initial_score: dict):
    # 1. Write final agent spec
    with open("outputs/final_agent.json", "w") as f:
        f.write(best_spec.to_json())

    # 2. Write eval comparison
    comparison = {
        "before": initial_score["aggregate"],
        "after": best_score["aggregate"],
        "delta": {k: best_score["aggregate"][k] - initial_score["aggregate"][k]
                  for k in initial_score["aggregate"]},
        "iterations_run": best_spec.version,
        "mutation_log": best_spec.mutation_log
    }
    with open("outputs/eval_results.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # 3. Print summary
    print("\n=== CRYSTALLIZATION COMPLETE ===")
    print(f"Final composite score: {best_score['aggregate']['composite']:.3f}")
    print(f"Improvement over baseline: {comparison['delta']['composite']:+.3f}")
    print(f"Architecture settled on: {best_spec.architecture_type}")
    print(f"Mutations accepted: {best_spec.version}")
    print("Final agent written to outputs/final_agent.json")
```

---

## 10. Main Entry Point

**File:** `stem_agent/main.py`

```python
def main():
    import anthropic, json
    from pathlib import Path

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    
    questions = json.loads(Path("eval_suite/questions.json").read_text())
    
    print("=== STEM AGENT ===")
    print("Domain: Deep Research\n")

    # Phase 1
    print("Phase 1: Sense")
    primitives = sense("Deep Research", client)

    # Phase 2
    print("\nPhase 2: Hypothesize")
    initial_spec = hypothesize(primitives, client)
    
    # Baseline eval (before differentiation)
    print("\nRunning baseline eval...")
    initial_score = run_eval(initial_spec, questions, client)
    print(f"Baseline composite: {initial_score['aggregate']['composite']:.3f}")

    # Phase 3
    print("\nPhase 3: Differentiate")
    best_spec, best_score = differentiate(initial_spec, initial_score, questions, client)

    # Phase 4
    print("\nPhase 4: Crystallize")
    crystallize(best_spec, best_score, initial_score)

if __name__ == "__main__":
    main()
```

---

## 11. Requirements File

```
anthropic>=0.25.0
python-dotenv>=1.0.0
```

Optional (add if web search is available):
```
tavily-python>=0.3.0
```

---

## 12. Setup Instructions (for README.md)

Write these exactly into `README.md` when you create it:

```markdown
## Setup

1. Python 3.11+ required.
2. `pip install -r requirements.txt`
3. Set environment variable: `export ANTHROPIC_API_KEY=your_key_here`
4. (Optional) Set `TAVILY_API_KEY` for live web search in multi_step_search architecture.
5. Run: `python -m stem_agent.main`

## What to expect

Full run takes ~15-25 minutes (8 iterations × 10 questions × ~4 API calls per question).
Outputs land in `outputs/`. Progress logs to stdout.

## Cost estimate

~$2-5 USD depending on which architecture the agent selects and how many iterations run.
```

---

## 13. The Write-Up

**File:** `writeup/writeup.md`  
**Write this LAST, after the full run completes.**  
**Max 4 pages. No padding. Say things once.**

Use this exact structure:

### Page 1 — Approach

Answer these questions in flowing prose (no bullet points):
- What is the stem cell metaphor really asking for architecturally?
- Why Deep Research as the domain?
- What does the AgentSpec abstraction buy you — why represent the agent as a data structure?
- What was the riskiest design decision and why did you make it?

### Page 2 — Experiments

Write a chronological account of what happened when you ran it:
- What did Phase 1 surface that you didn't expect?
- What architecture did Phase 2 hypothesize, and was it surprising?
- Walk through iterations 1-N: what mutations were proposed, which were accepted, which rolled back, and why?
- Include the before/after score table (pull from `outputs/eval_results.json`)

Score table format:
| Dimension | Baseline | Final | Δ |
|---|---|---|---|
| factual_accuracy | | | |
| coverage | | | |
| coherence | | | |
| source_diversity | | | |
| **composite** | | | |

### Page 3 — What Failed

Be honest. Include at minimum:
- One thing the rollback mechanism caught (a mutation that hurt performance)
- One thing the stem agent got wrong that you had to manually observe (not catch automatically)
- One place where the eval metric was misleading (a high score that was actually low quality, or vice versa)
- What the convergence criterion missed

### Page 4 — What I'd Do With More Time

Three concrete directions, each with a specific technical proposal:
1. **Better mutation operators** — e.g. genetic crossover across checkpoints, not just point mutations
2. **Learned convergence** — e.g. train a small classifier on (spec, eval_result) pairs to predict whether a mutation will help
3. **Multi-domain generalization** — what changes in the AgentSpec schema would be needed to handle Security Auditing or QA as domains?

End with one paragraph: what does this build make you believe about the stem agent concept that you didn't believe before you started?

---

## 14. Build Order

Execute in this exact sequence:

1. Create directory structure and empty files
2. Implement `AgentSpec` + round-trip test
3. Create `eval_suite/questions.json`
4. Implement `runner.py` (all four architectures, even if stubs)
5. Implement `harness.py`
6. Implement `sense.py`
7. Implement `hypothesize.py`
8. Implement `checkpointer.py`
9. Implement `differentiate.py`
10. Implement `crystallize.py`
11. Wire `main.py`
12. Smoke test: run one question through the runner manually
13. Full run
14. Write `writeup.md` from actual outputs
15. Write `README.md`

**Do not start step N+1 until step N produces the expected output.**

---

## 15. Acceptance Criteria (final)

The submission passes if:
- [ ] `python -m stem_agent.main` runs to completion without error
- [ ] `outputs/final_agent.json` exists and is valid JSON matching `AgentSpec` schema
- [ ] `outputs/eval_results.json` exists and shows a before/after comparison
- [ ] At least one rollback appears in `outputs/checkpoints/` (score went down, then recovered)
- [ ] `writeup/writeup.md` references specific iteration numbers and actual scores from the run
- [ ] The before composite score and after composite score are different (the agent changed)
