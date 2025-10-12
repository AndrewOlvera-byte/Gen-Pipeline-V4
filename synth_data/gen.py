from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import gymnasium as gym
import minari
import numpy as np

from synth_data.envs import make_fetchreach_env, HeuristicFetchReachExpert
from synth_data.utils import (
    create_progress,
    episode_progress,
    summarize_dataset_quality,
    resolve_repo_root,
    ensure_data_dir,
)


def _collect_episode(env: gym.Env, expert: HeuristicFetchReachExpert, max_steps: Optional[int] = None) -> Dict[str, np.ndarray]:

    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    rew_list: List[float] = []
    terminated = False
    truncated = False
    steps = 0

    obs, _ = env.reset()
    while True:
        obs_list.append(np.asarray(obs, dtype=np.float32))
        action, _ = expert.predict(obs, deterministic=True)
        act_list.append(np.asarray(action, dtype=np.float32))
        next_obs, reward, terminated, truncated, info = env.step(action)
        rew_list.append(float(reward))
        obs = next_obs
        steps += 1
        if max_steps is not None and steps >= max_steps:
            truncated = True
        if terminated or truncated:
            obs_list.append(np.asarray(next_obs, dtype=np.float32))
            break

    # Convert to arrays with proper alignment for Minari expectations
    observations = np.stack(obs_list, axis=0)
    actions = np.stack(act_list, axis=0)
    rewards = np.asarray(rew_list, dtype=np.float32)
    terminations = np.zeros_like(rewards, dtype=bool)
    truncations = np.zeros_like(rewards, dtype=bool)
    if len(rewards) > 0:
        terminations[-1] = bool(terminated)
        truncations[-1] = bool(truncated)

    # Attempt to extract success metric
    success = 0.0
    try:
        # Gymnasium robotics often exposes 'is_success' in info at last step
        success = float(info.get("is_success", 0.0)) if isinstance(info, dict) else 0.0
    except Exception:
        success = 0.0

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminations": terminations,
        "truncations": truncations,
        "success": success,
    }


def main():

    dataset_id = os.environ.get("DATASET_ID", "fetchreach_v2_expert_bc_v1")
    env_id = os.environ.get("ENV_ID", "FetchReach-v2")
    seed = int(os.environ.get("SEED", "0"))
    episodes = int(os.environ.get("EPISODES", "400"))
    max_steps_per_ep: Optional[int] = None

    # Build env and expert (CPU-only)
    env = make_fetchreach_env(env_id, seed=seed)
    expert = HeuristicFetchReachExpert(env)

    # Resolve output directory under repo_root/data
    repo_root: Path = resolve_repo_root()
    data_dir: Path = ensure_data_dir(repo_root)
    dataset_dir: Path = data_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Force Minari to save datasets strictly under repo_root/data
    os.environ["MINARI_DATASET_PATH"] = str(data_dir)

    # Progress bars
    ep_bar, step_bar = create_progress(total_episodes=episodes)

    # Collect episodes
    collected: List[Dict[str, Any]] = []
    episode_stats = []
    for ep_idx in range(episodes):
        ep = _collect_episode(env, expert, max_steps=max_steps_per_ep)
        collected.append({
            "observations": ep["observations"],
            "actions": ep["actions"],
            "rewards": ep["rewards"],
            "terminations": ep["terminations"],
            "truncations": ep["truncations"],
        })
        episode_stats.append(type("S", (), {"length": len(ep["rewards"]), "return_sum": float(ep["rewards"].sum()), "success": float(ep["success"])})())

        # Update progress text
        ep_bar.set_postfix_str(episode_progress(current_step=len(ep["rewards"]), max_steps=max_steps_per_ep))
        ep_bar.update(1)

    # Save with Minari using the new buffers API only (no fallback)
    if not hasattr(minari, "create_dataset_from_buffers"):
        raise RuntimeError(
            "Minari create_dataset_from_buffers is required but not available. "
            "Please install a Minari version that provides this API."
        )
    minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        buffer=collected,
        env=env_id,
        algorithm_name="HeuristicPD",
        code_permalink="",
        author="local",
    )

    # Compute dims from collected buffers (no dataset reload)
    ds_path = str(dataset_dir)
    obs_dim = int(collected[0]["observations"].shape[-1])
    act_dim = int(collected[0]["actions"].shape[-1])

    # Quality summary
    q = summarize_dataset_quality(episode_stats, obs_dim=obs_dim, act_dim=act_dim)

    # Final output
    print("=== Minari dataset saved ===")
    print(f"Dataset ID: {dataset_id}")
    print(f"Path: {ds_path}")
    print(f"Episodes: {q['episodes']}")
    print(f"Total steps: {q['total_steps']}")
    print(f"Avg episode length: {q['avg_episode_length']:.2f}")
    print(f"Avg return: {q['avg_return']:.3f}")
    print(f"Success rate: {q['success_rate']*100:.1f}%")
    print(f"Obs dim: {q['obs_dim']}  Act dim: {q['act_dim']}")
    print(f"Good for BC: {q['good_for_bc']}")


if __name__ == "__main__":
    main()


