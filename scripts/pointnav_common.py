"""Pure helpers for strict published-policy loading and finite episode summaries."""

import math
from typing import Any, Dict, List


def policy_state_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    """Remove only the published updater prefix, preserving every policy tensor."""
    if not state:
        raise ValueError("Empty checkpoint state_dict")
    if not all(key.startswith("actor_critic.") for key in state):
        raise ValueError("Unexpected checkpoint keys; refusing partial policy loading")
    return {key[len("actor_critic."):]: value for key, value in state.items()}


def mean_metrics(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    """Average required navigation metrics, rejecting missing or non-finite values."""
    if not episodes:
        raise ValueError("No completed episodes")
    result = {}
    for key in ("success", "spl", "distance_to_goal", "num_steps"):
        values = [float(episode[key]) for episode in episodes]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite metric: " + key)
        result[key] = sum(values) / len(values)
    return result
