"""
harness.py — Run a full eval suite against a candidate AgentSpec and score results.

Scoring uses a separate LLM judge call per question on four dimensions:
  factual_accuracy, coverage, coherence, source_diversity (each 0.0-1.0).
Composite = mean of four dimensions.
"""

import json
import time
import datetime
from stem_agent.core.agent_spec import AgentSpec
from stem_agent.core.runner import run_candidate_agent
from stem_agent.core.llm import call_llm

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


def _score_answer(question: dict, answer: str) -> dict:
    """Call the LLM judge and return a scores dict."""
    prompt = JUDGE_PROMPT.format(
        question=question["question"],
        ground_truth=question["ground_truth"],
        key_facts=", ".join(question["key_facts"]),
        answer=answer,
    )
    messages = [{"role": "user", "content": prompt}]
    raw = call_llm(messages, max_tokens=256)

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    scores = json.loads(raw)
    dims = ["factual_accuracy", "coverage", "coherence", "source_diversity"]
    composite = sum(float(scores[d]) for d in dims) / len(dims)
    scores["composite"] = round(composite, 4)
    return scores


def _compute_aggregate(per_question: list[dict]) -> dict:
    dims = ["factual_accuracy", "coverage", "coherence", "source_diversity", "composite"]
    aggregate = {}
    for d in dims:
        aggregate[d] = round(
            sum(q["scores"][d] for q in per_question) / len(per_question), 4
        )
    return aggregate


def run_eval(spec: AgentSpec, questions: list[dict]) -> dict:
    """
    Run all questions through the candidate agent, judge each answer, aggregate.

    Returns:
    {
        "per_question": [{"id": "q01", "scores": {...}, "composite": 0.0, "answer": "..."}, ...],
        "aggregate": {"factual_accuracy": 0.0, "coverage": 0.0, "coherence": 0.0,
                      "source_diversity": 0.0, "composite": 0.0},
        "version": spec.version,
        "timestamp": "..."
    }
    """
    per_question = []

    for q in questions:
        print(f"  Evaluating {q['id']} ({q['tier']})...", end=" ", flush=True)

        # Run the candidate agent
        answer = run_candidate_agent(spec, q["question"])
        time.sleep(1)  # avoid rate limits between runner + judge calls

        # Judge the answer
        scores = _score_answer(q, answer)
        time.sleep(1)

        per_question.append({
            "id": q["id"],
            "tier": q["tier"],
            "scores": scores,
            "composite": scores["composite"],
            "answer": answer,
        })
        print(f"composite={scores['composite']:.3f}")

    aggregate = _compute_aggregate(per_question)
    return {
        "per_question": per_question,
        "aggregate": aggregate,
        "version": spec.version,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
