"""
main.py — Entry point for the stem agent.

Usage:
    uv run python -m stem_agent.main

Environment variables:
    STEM_AGENT_MODEL  — any LiteLLM model string (default: claude-3-5-sonnet-20241022)
    ANTHROPIC_API_KEY — required if using Claude models
    OPENAI_API_KEY    — required if using OpenAI models
    TAVILY_API_KEY    — optional, enables live web search in multi_step_search
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from stem_agent.phases.sense import sense
from stem_agent.phases.hypothesize import hypothesize
from stem_agent.phases.differentiate import differentiate
from stem_agent.phases.crystallize import crystallize
from stem_agent.eval.harness import run_eval
from stem_agent.core.checkpointer import save_checkpoint


def main():
    questions = json.loads(Path("eval_suite/questions.json").read_text())

    print("=== STEM AGENT ===")
    print("Domain: Deep Research")
    model = os.environ.get("STEM_AGENT_MODEL", "claude-3-5-sonnet-20241022")
    print(f"Model:  {model}\n")

    # Phase 1
    print("Phase 1: Sense")
    primitives = sense("Deep Research")

    # Phase 2
    print("\nPhase 2: Hypothesize")
    initial_spec = hypothesize(primitives)

    # Baseline eval (before differentiation)
    print("\nRunning baseline eval...")
    initial_score = run_eval(initial_spec, questions)
    print(f"Baseline composite: {initial_score['aggregate']['composite']:.3f}")

    # Save baseline checkpoint (v0 with actual score)
    save_checkpoint(initial_spec, initial_score, iteration=0)

    # Phase 3
    print("\nPhase 3: Differentiate")
    best_spec, best_score = differentiate(initial_spec, initial_score, questions)

    # Phase 4
    print("\nPhase 4: Crystallize")
    crystallize(best_spec, best_score, initial_score)


if __name__ == "__main__":
    main()
