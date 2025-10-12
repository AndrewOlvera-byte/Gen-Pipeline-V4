from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np


@dataclass
class EpisodeStats:
    length: int
    return_sum: float
    success: float


def summarize_dataset_quality(episodes: List[EpisodeStats], obs_dim: int, act_dim: int) -> Dict[str, Any]:

    num_episodes = len(episodes)
    lengths = np.array([e.length for e in episodes], dtype=np.float32) if episodes else np.array([0.0])
    returns = np.array([e.return_sum for e in episodes], dtype=np.float32) if episodes else np.array([0.0])
    successes = np.array([e.success for e in episodes], dtype=np.float32) if episodes else np.array([0.0])

    total_steps = int(lengths.sum())
    avg_len = float(lengths.mean()) if num_episodes > 0 else 0.0
    avg_ret = float(returns.mean()) if num_episodes > 0 else 0.0
    success_rate = float(successes.mean()) if num_episodes > 0 else 0.0

    # Heuristics for "good enough BC dataset"
    # - at least ~20k transitions
    # - success rate high enough (>= 0.85)
    # - non-trivial episode length (>= 10)
    good_enough = (total_steps >= 20_000) and (success_rate >= 0.85) and (avg_len >= 10)

    return {
        "episodes": num_episodes,
        "total_steps": total_steps,
        "avg_episode_length": avg_len,
        "avg_return": avg_ret,
        "success_rate": success_rate,
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "good_for_bc": bool(good_enough),
    }

