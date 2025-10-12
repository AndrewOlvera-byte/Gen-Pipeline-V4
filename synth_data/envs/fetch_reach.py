from __future__ import annotations
from typing import Optional
import gymnasium as gym
import numpy as np


class ObservationKeyWrapper(gym.ObservationWrapper):
    """
    Selects a single key from a Dict observation and exposes it as the observation.
    Intended for robotics envs where obs is a dict containing 'observation', 'achieved_goal', 'desired_goal'.
    """
    def __init__(self, env: gym.Env, key: str = "observation"):
        super().__init__(env)
        assert hasattr(env.observation_space, "spaces"), "ObservationKeyWrapper requires a Dict observation space"
        assert key in env.observation_space.spaces, f"Key '{key}' not in observation space"
        self.key = key
        self.observation_space = env.observation_space.spaces[self.key]

    def observation(self, observation):
        return observation[self.key]


def make_fetchreach_env(id: str = "FetchReach-v3", seed: Optional[int] = 0) -> gym.Env:
    env = gym.make(id)
    if seed is not None:
        # Gymnasium reset handles seeding at reset time; keep for completeness
        try:
            env.reset(seed=int(seed))
        except Exception:
            pass
    # Reduce obs to the 'observation' vector only for BC compatibility
    env = ObservationKeyWrapper(env, key="observation")
    # Flatten to 1D just in case it is multi-dim
    env = gym.wrappers.FlattenObservation(env)
    return env


def _find_fetch_base_env(env: gym.Env):
    """Walk down wrappers to the base Fetch env that exposes _get_obs and goal."""
    cur = env
    max_depth = 10
    for _ in range(max_depth):
        # Heuristic: stop when we find attributes used by FetchEnv
        if hasattr(cur, "_get_obs") and hasattr(cur, "goal"):
            return cur
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    return None


class HeuristicFetchReachExpert:
    """
    Simple PD-like controller for FetchReach: move gripper towards desired goal.
    Assumes action space is 4D: dx, dy, dz, gripper (we keep gripper neutral).
    """
    def __init__(self, env: gym.Env, kp: float = 5.0, max_delta: float = 1.0):
        self.env = env
        self.kp = float(kp)
        self.max_delta = float(max_delta)

        base = _find_fetch_base_env(env)
        if base is None:
            raise RuntimeError("Could not locate underlying FetchEnv for heuristic expert.")
        self._base = base
        self._act_low = getattr(env.action_space, "low", np.array([-1, -1, -1, -1], dtype=np.float32))
        self._act_high = getattr(env.action_space, "high", np.array([1, 1, 1, 1], dtype=np.float32))

    def predict(self, obs, deterministic: bool = True):
        raw = self._base._get_obs()
        ag = np.asarray(raw["achieved_goal"], dtype=np.float32)
        dg = np.asarray(raw["desired_goal"], dtype=np.float32)
        delta = dg - ag
        ctrl = np.clip(self.kp * delta, -self.max_delta, self.max_delta)
        if ctrl.shape[0] == 3:
            ctrl = np.concatenate([ctrl, np.array([0.0], dtype=np.float32)], axis=0)
        ctrl = np.clip(ctrl, self._act_low, self._act_high)
        return ctrl, None

