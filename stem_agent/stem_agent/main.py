"""
main.py — Entry point for the stem agent.

Usage:
    uv run python -m stem_agent.main --domain "Deep Research"
    uv run python -m stem_agent.main --domain "Code Review" --questions path/to/questions.json
    uv run python -m stem_agent.main --domain "Deep Research" --resume  # continue from best checkpoint

Arguments:
    --domain    Task domain to specialise for (default: "Deep Research").
                Outputs are isolated under outputs/<domain_slug>/.
    --questions Path to a JSON questions file (default: eval_suite/questions.json).
    --resume    If a checkpoint already exists for this domain, skip Phases 1-2 and
                continue the differentiation loop from the best saved checkpoint.

Environment variables:
    STEM_AGENT_MODEL  — any LiteLLM model string (default: claude-3-5-sonnet-20241022)
    MODEL             — fallback model override (from .env)
    ANTHROPIC_API_KEY — required if using Claude models
    OPENAI_API_KEY    — required if using OpenAI models
    OPENAI_KEY        — alternative name for OPENAI_API_KEY
    TAVILY_API_KEY    — optional, enables live web search in multi_step_search
"""

import argparse
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from stem_agent.phases.sense import sense
from stem_agent.phases.hypothesize import hypothesize
from stem_agent.phases.differentiate import differentiate
from stem_agent.phases.crystallize import crystallize
from stem_agent.eval.harness import run_eval
from stem_agent.core.checkpointer import save_checkpoint, load_best_checkpoint
from stem_agent.core.llm import _get_model


def _domain_slug(domain: str) -> str:
    """Convert a domain name to a safe directory name, e.g. 'Deep Research' -> 'deep_research'."""
    return re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stem_agent",
        description=(
            "Stem agent: grows a specialized agent for a given task domain via "
            "self-differentiation. Outputs are scoped per domain under outputs/<domain_slug>/."
        ),
    )
    parser.add_argument(
        "--domain",
        default="Deep Research",
        metavar="DOMAIN",
        help='Task domain to specialise for (default: "Deep Research").',
    )
    parser.add_argument(
        "--questions",
        default="eval_suite/questions.json",
        metavar="PATH",
        help="Path to a JSON file with evaluation questions (default: eval_suite/questions.json).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the best existing checkpoint for this domain. "
            "Skips Phases 1 & 2 and continues the differentiation loop, "
            "then re-evaluates and updates final_agent.json."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    domain: str = args.domain
    questions_path: str = args.questions
    resume: bool = args.resume

    questions = json.loads(Path(questions_path).read_text())
    output_dir = os.path.join("outputs", _domain_slug(domain))
    os.makedirs(output_dir, exist_ok=True)

    print("=== STEM AGENT ===")
    print(f"Domain:  {domain}")
    print(f"Model:   {_get_model()}")
    print(f"Outputs: {output_dir}/")
    print(f"Resume:  {resume}\n")

    if resume:
        result = load_best_checkpoint(output_dir)
        if result is None:
            print("No checkpoints found for this domain — starting fresh.\n")
            resume = False
        else:
            initial_spec, initial_score = result
            print(
                f"Resuming from checkpoint v{initial_spec.version} "
                f"(composite={initial_score['aggregate']['composite']:.3f})\n"
            )

    if not resume:
        # Phase 1
        print("Phase 1: Sense")
        primitives = sense(domain)

        # Phase 2
        print("\nPhase 2: Hypothesize")
        initial_spec = hypothesize(primitives, task_domain=domain)

        # Baseline eval
        print("\nRunning baseline eval...")
        initial_score = run_eval(initial_spec, questions)
        print(f"Baseline composite: {initial_score['aggregate']['composite']:.3f}")

        # Print per-tier breakdown of baseline
        _print_tier_summary(initial_score["per_question"])

        save_checkpoint(initial_spec, initial_score, iteration=0, output_dir=output_dir)

    # Phase 3
    print("\nPhase 3: Differentiate")
    best_spec, best_score = differentiate(initial_spec, initial_score, questions, output_dir=output_dir)

    # Phase 4
    print("\nPhase 4: Crystallize")
    crystallize(best_spec, best_score, initial_score, output_dir=output_dir)


def _print_tier_summary(per_question: list[dict]) -> None:
    """Print a compact per-tier score table to stdout."""
    from collections import defaultdict
    tiers: dict[str, list] = defaultdict(list)
    for q in per_question:
        tiers[q["tier"]].append(q["composite"])
    print("  Baseline by tier:")
    for tier in ("easy", "medium", "hard"):
        if tier in tiers:
            scores = tiers[tier]
            print(f"    {tier:6s}  avg={sum(scores)/len(scores):.3f}  (n={len(scores)})")


if __name__ == "__main__":
    main()

