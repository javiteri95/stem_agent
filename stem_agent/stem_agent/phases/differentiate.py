"""
differentiate.py — Phase 3: The self-improvement loop.

Runs eval → introspects failures → proposes mutation → checkpoints improvements
→ rolls back regressions. Iterates until convergence or budget exhausted.
"""

import json
from stem_agent.core.agent_spec import AgentSpec
from stem_agent.core.llm import call_llm
from stem_agent.core.checkpointer import save_checkpoint
from stem_agent.eval.harness import run_eval

MAX_ITERATIONS = 8
CONVERGENCE_DELTA = 0.02
MIN_ITERATIONS = 3

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

MUTATE_PROMPT = """You are proposing a targeted improvement to a Deep Research agent spec.

CURRENT SPEC:
{spec_json}

FAILURE ANALYSIS:
{failure_analysis_json}

Propose exactly one mutation — change 1 to 3 fields of the spec. You MUST:
- Only change fields listed in failure_analysis.fields_to_change
- Stay within allowed values for each field:
    architecture_type: linear_chain | reflection_loop | multi_step_search | adversarial_qa
    search_strategy: breadth_first | depth_first | iterative_refinement
    synthesis_strategy: extractive | abstractive | hybrid
    max_search_rounds: integer 2-6
    self_critique: true | false
    citation_style: inline | footnote | none
    tools: list from [web_search, document_reader, calculator, code_executor]
    system_prompt: any string (2-4 sentences)
- Explain why this specific change should fix the identified root cause

Respond with ONLY a JSON object:
{{
  "changes": {{"field_name": "new_value"}},
  "expected_improvement": "one sentence",
  "rationale": "2-3 sentences"
}}"""


def _build_eval_summary(eval_result: dict) -> str:
    lines = ["Per-question composite scores:"]
    for q in eval_result["per_question"]:
        lines.append(
            f"  {q['id']} ({q['tier']}): composite={q['composite']:.3f} | "
            f"accuracy={q['scores']['factual_accuracy']:.2f} coverage={q['scores']['coverage']:.2f} "
            f"coherence={q['scores']['coherence']:.2f} diversity={q['scores']['source_diversity']:.2f}"
        )
    agg = eval_result["aggregate"]
    lines.append(
        f"Aggregate: composite={agg['composite']:.3f} | "
        f"accuracy={agg['factual_accuracy']:.3f} coverage={agg['coverage']:.3f} "
        f"coherence={agg['coherence']:.3f} diversity={agg['source_diversity']:.3f}"
    )
    return "\n".join(lines)


def _worst_questions(eval_result: dict, n: int = 3) -> str:
    sorted_qs = sorted(eval_result["per_question"], key=lambda q: q["composite"])
    worst = sorted_qs[:n]
    parts = []
    for q in worst:
        parts.append(
            f"[{q['id']}] composite={q['composite']:.3f}\n"
            f"Answer excerpt: {q['answer'][:300]}..."
        )
    return "\n\n".join(parts)


def _introspect(spec: AgentSpec, eval_result: dict) -> dict:
    """Ask LLM to identify root causes of poor eval performance."""
    prompt = INTROSPECT_PROMPT.format(
        spec_json=spec.to_json(),
        eval_summary=_build_eval_summary(eval_result),
        worst_questions=_worst_questions(eval_result),
    )
    messages = [{"role": "user", "content": prompt}]
    raw = call_llm(messages, max_tokens=512)

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


def _propose_mutation(spec: AgentSpec, failure_analysis: dict) -> AgentSpec:
    """Ask LLM to propose a concrete mutation, apply it to a cloned spec."""
    prompt = MUTATE_PROMPT.format(
        spec_json=spec.to_json(),
        failure_analysis_json=json.dumps(failure_analysis, indent=2),
    )
    messages = [{"role": "user", "content": prompt}]
    raw = call_llm(messages, max_tokens=512)

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    mutation = json.loads(raw)
    changes = mutation.get("changes", {})
    rationale = mutation.get("rationale", "")
    expected = mutation.get("expected_improvement", "")

    candidate = spec.clone_next_version()

    # Apply changes
    for field, value in changes.items():
        if hasattr(candidate, field):
            # Coerce types as needed
            if field == "max_search_rounds":
                value = int(value)
            elif field == "self_critique":
                if isinstance(value, str):
                    value = value.lower() == "true"
            setattr(candidate, field, value)
        else:
            print(f"  Warning: unknown field '{field}' in mutation — skipping")

    candidate.mutation_log.append({
        "from_version": spec.version,
        "to_version": candidate.version,
        "rationale": rationale,
        "expected_improvement": expected,
        "changes": changes,
    })

    print(f"  Proposed mutation: {changes}")
    print(f"  Expected: {expected}")
    print(f"  Rationale: {rationale}")

    return candidate


def differentiate(
    initial_spec: AgentSpec,
    initial_score: dict,
    questions: list[dict],
    output_dir: str = "outputs",
) -> tuple[AgentSpec, dict]:
    """
    Phase 3: Run the differentiation loop.

    Returns (best_spec, best_score).
    """
    best_spec = initial_spec
    best_score = initial_score
    save_checkpoint(best_spec, best_score, iteration=0, output_dir=output_dir)
    consecutive_small_improvements = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        print(f"Current best composite: {best_score['aggregate']['composite']:.3f}")

        # Step 1: Introspect failures
        print("  Introspecting failures...")
        failure_analysis = _introspect(best_spec, best_score)
        print(f"  Root causes: {failure_analysis.get('root_causes', [])}")
        print(f"  Fields to change: {failure_analysis.get('fields_to_change', [])}")

        # Step 2: Propose mutation
        print("  Proposing mutation...")
        candidate_spec = _propose_mutation(best_spec, failure_analysis)

        # Step 3: Eval candidate
        print(f"  Running eval for candidate v{candidate_spec.version}...")
        candidate_score = run_eval(candidate_spec, questions)
        delta = (
            candidate_score["aggregate"]["composite"]
            - best_score["aggregate"]["composite"]
        )
        print(
            f"  Candidate composite: {candidate_score['aggregate']['composite']:.3f} "
            f"(delta: {delta:+.3f})"
        )

        # Step 4: Checkpoint or rollback
        if delta >= 0:
            best_spec = candidate_spec
            best_score = candidate_score
            save_checkpoint(best_spec, best_score, iteration, output_dir=output_dir)
            print("  ✓ Improvement accepted")
            if delta < CONVERGENCE_DELTA:
                consecutive_small_improvements += 1
            else:
                consecutive_small_improvements = 0
        else:
            # Save rollback checkpoint so we can observe it later
            save_checkpoint(candidate_spec, candidate_score, iteration, output_dir=output_dir)
            print("  ✗ Regression — rolling back")

        # Step 5: Convergence check
        if iteration >= MIN_ITERATIONS and consecutive_small_improvements >= 2:
            print(f"  Converged at iteration {iteration}. Exiting loop.")
            break

    return best_spec, best_score
