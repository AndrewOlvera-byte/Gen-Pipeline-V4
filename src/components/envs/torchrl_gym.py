from __future__ import annotations
import gymnasium as gym
from src.core.registry import register

from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import Compose, DoubleToFloat, FlattenObservation, StepCounter

@register("env", "torchrl_gym")
class TorchRLGymFactory:
    def build(self, cfg_node, context):
        """
        Builds a TorchRL TransformedEnv wrapping a Gymnasium env.
        Keys follow TorchRL defaults: "observation", "action", "reward", "done".
        """
        base = GymEnv(cfg_node.id, frame_skip=1, from_pixels=False, gbs=True, **cfg_node.kwargs)
        transforms = [DoubleToFloat()]
        # Flatten common dict observations into a single vector
        transforms += [FlattenObservation(keys=["observation"])]
        transforms += [StepCounter()]  # adds "step_count" if you ever need truncation by steps
        env = TransformedEnv(base, Compose(*transforms))
        # expose specs and dims
        action_spec = env.action_spec
        obs_spec = env.observation_spec
        obs_dim = obs_spec["observation"].shape[-1]
        act_dim = action_spec.shape[-1]
        return {"env": env, "obs_dim": obs_dim, "act_dim": act_dim, "action_spec": action_spec}
