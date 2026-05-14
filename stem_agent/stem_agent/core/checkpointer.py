"""
checkpointer.py — Save and load AgentSpec versions from domain-scoped directories.

Checkpoint files live under:
    outputs/<domain_slug>/checkpoints/v<version>_iter<n>_score<x.xxx>.json
"""

import glob
import json
import os
from stem_agent.core.agent_spec import AgentSpec


def save_checkpoint(
    spec: AgentSpec,
    score: dict,
    iteration: int,
    output_dir: str = "outputs",
) -> str:
    """
    Persist a checkpoint to <output_dir>/checkpoints/.

    Returns the filepath written.
    """
    path = os.path.join(output_dir, "checkpoints")
    os.makedirs(path, exist_ok=True)
    composite = score["aggregate"]["composite"]
    filename = os.path.join(path, f"v{spec.version}_iter{iteration}_score{composite:.3f}.json")
    payload = {
        "spec": json.loads(spec.to_json()),
        "score": score,
        "iteration": iteration,
    }
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    return filename


def load_best_checkpoint(output_dir: str) -> tuple[AgentSpec, dict] | None:
    """
    Scan <output_dir>/checkpoints/ and return (spec, score) for the checkpoint
    with the highest composite score.  Returns None if no checkpoints exist.
    """
    pattern = os.path.join(output_dir, "checkpoints", "*.json")
    files = glob.glob(pattern)
    if not files:
        return None

    best_file = None
    best_composite = -1.0

    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            composite = data["score"]["aggregate"]["composite"]
            if composite > best_composite:
                best_composite = composite
                best_file = f
                best_data = data
        except Exception:
            continue

    if best_file is None:
        return None

    spec = AgentSpec.from_json(json.dumps(best_data["spec"]))
    score = best_data["score"]
    return spec, score
