from __future__ import annotations
from typing import Any, Dict
import torch
from torchrl.envs import ParallelEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import TransformedEnv, DoubleToFloat, CatTensors, FlattenObservation
from src.core.registry import register


def _make_single_env(env_id: str, kwargs: Dict[str, Any] | None = None):
    base = GymEnv(env_id, **(kwargs or {}))
    env = TransformedEnv(base, DoubleToFloat())
    env.append_transform(FlattenObservation(-1))
    return env


@register("env", "torchrl_vectorized")
class TorchRLVectorizedEnvFactory:
    def build(self, cfg_node, context):
        num_envs = int(getattr(cfg_node, "num_envs", 8))
        env_id = getattr(cfg_node, "id", None)
        if not env_id:
            raise ValueError("env.id required for torchrl_vectorized")
        kwargs = getattr(cfg_node, "kwargs", {})
        def make():
            return _make_single_env(env_id, kwargs)
        penv = ParallelEnv(num_envs, make, shared_memory=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {"env": penv.to(device)}


