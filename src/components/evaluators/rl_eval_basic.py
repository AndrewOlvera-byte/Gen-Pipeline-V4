from __future__ import annotations
import numpy as np, torch
from src.core.registry import register
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.data import TensorDict

@register("evaluator", "rl_eval_basic")
class RLEvalFactory:
    def build(self, cfg_node, context):
        env = context["env"]["env"]
        actor = context["model"]["actor"]
        episodes = int(cfg_node.episodes)
        device = next(actor.parameters()).device

        @torch.no_grad()
        def run(_model):
            actor.eval()
            set_exploration_type(ExplorationType.MODE)  # deterministic (mean)
            returns = []
            for _ in range(episodes):
                tensordict = env.reset()
                done = False
                ep_ret = 0.0
                while not done:
                    tensordict = actor(tensordict.to(device))
                    tensordict = env.step(tensordict)
                    ep_ret += float(tensordict["reward"].cpu().item())
                    done = bool(tensordict.get(("next","done"), tensordict["done"]).cpu().item())
                    tensordict = tensordict["next"]
                returns.append(ep_ret)
            return {"eval/return_mean": float(np.mean(returns)), "eval/return_std": float(np.std(returns))}
        return run
