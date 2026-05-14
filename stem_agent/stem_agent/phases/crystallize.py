"""
crystallize.py — Phase 4: Write final outputs to disk.

Outputs are scoped to <output_dir>/:
  final_agent.json   — the winning AgentSpec (overwritten on re-runs)
  eval_results.json  — before/after aggregate scores + mutation log
  eval_by_tier.json  — per-tier breakdown (easy / medium / hard)
"""

import json
import os
from collections import defaultdict
from stem_agent.core.agent_spec import AgentSpec


def _tier_breakdown(per_question: list[dict]) -> dict:
    """Aggregate scores grouped by question tier."""
    tiers: dict[str, list[dict]] = defaultdict(list)
    for q in per_question:
        tiers[q["tier"]].append(q["scores"])

    result = {}
    dims = ["factual_accuracy", "coverage", "coherence", "source_diversity", "composite"]
    for tier, scores_list in sorted(tiers.items()):
        result[tier] = {
            d: round(sum(s[d] for s in scores_list) / len(scores_list), 4)
            for d in dims
        }
        result[tier]["n"] = len(scores_list)
    return result


def crystallize(
    best_spec: AgentSpec,
    best_score: dict,
    initial_score: dict,
    output_dir: str = "outputs",
) -> None:
    """
    Write final_agent.json, eval_results.json, and eval_by_tier.json to output_dir.
    Files are overwritten if they already exist (supports re-runs on the same domain).
    Print a summary to stdout.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Write final agent spec (overwrite — re-run updates the winner)
    final_agent_path = os.path.join(output_dir, "final_agent.json")
    with open(final_agent_path, "w") as f:
        f.write(best_spec.to_json())

    # 2. Write aggregate eval comparison
    agg_before = initial_score["aggregate"]
    agg_after = best_score["aggregate"]
    comparison = {
        "domain": best_spec.task_domain,
        "before": agg_before,
        "after": agg_after,
        "delta": {k: round(agg_after[k] - agg_before[k], 4) for k in agg_before},
        "iterations_run": best_spec.version,
        "mutation_log": best_spec.mutation_log,
    }
    eval_results_path = os.path.join(output_dir, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # 3. Write per-tier breakdown
    tier_path = os.path.join(output_dir, "eval_by_tier.json")
    tier_breakdown = {
        "domain": best_spec.task_domain,
        "architecture": best_spec.architecture_type,
        "tiers": _tier_breakdown(best_score["per_question"]),
    }
    with open(tier_path, "w") as f:
        json.dump(tier_breakdown, f, indent=2)

    # 4. Print summary
    print(f"\n=== CRYSTALLIZATION COMPLETE [{best_spec.task_domain}] ===")
    print(f"Final composite score:        {agg_after['composite']:.3f}")
    print(f"Improvement over baseline:    {comparison['delta']['composite']:+.3f}")
    print(f"Architecture settled on:      {best_spec.architecture_type}")
    print(f"Mutations accepted:           {best_spec.version}")
    print(f"\nPer-tier breakdown:")
    for tier, scores in tier_breakdown["tiers"].items():
        print(
            f"  {tier:6s}  composite={scores['composite']:.3f}  "
            f"accuracy={scores['factual_accuracy']:.3f}  "
            f"coverage={scores['coverage']:.3f}  "
            f"coherence={scores['coherence']:.3f}  "
            f"diversity={scores['source_diversity']:.3f}  "
            f"(n={scores['n']})"
        )
    print(f"\nOutputs written to {output_dir}/")
    print(f"  {final_agent_path}")
    print(f"  {eval_results_path}")
    print(f"  {tier_path}")

