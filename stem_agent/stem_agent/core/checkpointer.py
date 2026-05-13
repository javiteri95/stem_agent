"""
checkpointer.py — Save AgentSpec versions to disk.
"""

import json
import os
from stem_agent.core.agent_spec import AgentSpec


def save_checkpoint(
    spec: AgentSpec,
    score: dict,
    iteration: int,
    path: str = "outputs/checkpoints/",
) -> str:
    """
    Persist a checkpoint to disk.

    Returns the filepath written.
    """
    os.makedirs(path, exist_ok=True)
    composite = score["aggregate"]["composite"]
    filename = f"{path}v{spec.version}_iter{iteration}_score{composite:.3f}.json"
    payload = {
        "spec": json.loads(spec.to_json()),
        "score": score,
        "iteration": iteration,
    }
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    return filename
