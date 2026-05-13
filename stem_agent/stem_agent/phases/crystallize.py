"""
crystallize.py — Phase 4: Write final outputs to disk.
"""

import json
import os
from stem_agent.core.agent_spec import AgentSpec


def crystallize(best_spec: AgentSpec, best_score: dict, initial_score: dict) -> None:
    """
    Write final_agent.json and eval_results.json to outputs/.
    Print a summary to stdout.
    """
    os.makedirs("outputs", exist_ok=True)

    # 1. Write final agent spec
    with open("outputs/final_agent.json", "w") as f:
        f.write(best_spec.to_json())

    # 2. Write eval comparison
    agg_before = initial_score["aggregate"]
    agg_after = best_score["aggregate"]
    comparison = {
        "before": agg_before,
        "after": agg_after,
        "delta": {k: round(agg_after[k] - agg_before[k], 4) for k in agg_before},
        "iterations_run": best_spec.version,
        "mutation_log": best_spec.mutation_log,
    }
    with open("outputs/eval_results.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # 3. Print summary
    print("\n=== CRYSTALLIZATION COMPLETE ===")
    print(f"Final composite score:        {agg_after['composite']:.3f}")
    print(f"Improvement over baseline:    {comparison['delta']['composite']:+.3f}")
    print(f"Architecture settled on:      {best_spec.architecture_type}")
    print(f"Mutations accepted:           {best_spec.version}")
    print("Final agent written to        outputs/final_agent.json")
    print("Eval comparison written to    outputs/eval_results.json")
