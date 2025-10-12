from __future__ import annotations
import torch
from torchrl.collectors import SyncDataCollector
from src.core.registry import register

def _scale_to_env_bounds(tanh_action, low, high):
    # maps [-1,1] → [low, high]
    return (0.5 * (tanh_action + 1.0)) * (torch.as_tensor(high, device=tanh_action.device) - torch.as_tensor(low, device=tanh_action.device)) + torch.as_tensor(low, device=tanh_action.device)

@register("collector", "sync")
class SyncCollectorFactory:
    def build(self, cfg_node, context):
        """
        Returns a SyncDataCollector that yields batches (list of transitions).
        We store old logp and value alongside actions to support PPO.
        """
        env = context["env"]["env"]
        model = context["model"]
        low = context["env"]["act_low"]; high = context["env"]["act_high"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).train()

        def policy_step(obs_np):
            # obs_np is numpy from env; convert to torch
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).view(1, -1)
            dist = model.policy(obs)
            action_tanh = dist.rsample()
            logp = dist.log_prob(action_tanh)
            value = model.value(obs)
            # scale to env bounds
            action_env = _scale_to_env_bounds(action_tanh, low, high)
            return action_env.squeeze(0).detach().cpu().numpy(), {
                "logp": logp.detach().cpu().numpy()[0],
                "value": value.detach().cpu().numpy()[0],
                "mu": dist._mu.detach().cpu().numpy()[0],
                "std": dist._std.detach().cpu().numpy()[0],
            }

        # TorchRL's generic collectors operate over env wrappers; here we keep it simple:
        # we provide a minimal SyncDataCollector-like iterator using the gym env.
        # If you prefer real TorchRL tensordicts, you can replace this with EnvBase later.
        class SimpleSyncCollector:
            def __init__(self, env, frames_per_batch, total_frames):
                self.env = env
                self.fpb = int(frames_per_batch)
                self.total = int(total_frames)

            def __iter__(self):
                frames = 0
                while frames < self.total:
                    batch = []
                    obs, _ = env.reset()
                    done = False
                    while not done and len(batch) < self.fpb:
                        act, info = policy_step(obs)
                        next_obs, reward, terminated, truncated, _ = env.step(act)
                        done = terminated or truncated
                        batch.append({
                            "obs": obs,
                            "act": act,
                            "rew": reward,
                            "done": float(done),
                            "next_obs": next_obs,
                            "logp_old": info["logp"],
                            "value_old": info["value"],
                            "mu": info["mu"], "std": info["std"],
                        })
                        obs = next_obs
                    frames += len(batch)
                    yield batch

        return SimpleSyncCollector(env=env,
                                   frames_per_batch=cfg_node.frames_per_batch,
                                   total_frames=cfg_node.total_frames or 1_000_000)
