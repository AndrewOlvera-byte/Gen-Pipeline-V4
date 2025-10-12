from __future__ import annotations
import torch
from src.core.registry import register

@register("loss", "bc_gaussian_nll")
class BCLossFactory:
    def build(self, cfg_node, context):
        lam = float(cfg_node.args.get("action_smooth_coef", 0.0))
        def loss_fn(model, batch):
            obs, act = batch["obs"], batch["act"]
            dist = model.policy(obs)
            nll = -dist.log_prob(act).mean()
            reg = torch.zeros((), device=obs.device)
            if lam > 0 and "prev_act" in batch:
                reg = lam * ((act - batch["prev_act"]) ** 2).mean()
            total = nll + reg
            return {"loss": total, "bc/nll": nll.detach(), "bc/reg": reg.detach(),
                    "bc/std_mean": dist._std.mean().detach(), "bc/std_min": dist._std.min().detach()}
        return loss_fn
