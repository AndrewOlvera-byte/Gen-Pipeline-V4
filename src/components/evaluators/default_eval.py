from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from src.core.registry import register

@register("evaluator", "default_eval")
class DefaultEvaluatorFactory:
    def build(self, cfg_node, context):
        valid_ds = context["dataset"]["valid"]
        bs = context["cfg"].dataset.batch_size
        dl = DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=0)
        @torch.no_grad()
        def run(model):
            model.eval()
            total_nll, total_n = 0.0, 0
            for batch in dl:
                batch = {k: (v.cuda(non_blocking=True) if v.__class__.__name__=="Tensor" and torch.cuda.is_available() else v)
                         for k, v in batch.items()}
                dist = model.policy(batch["obs"])
                nll = -dist.log_prob(batch["act"])  # (B,)
                total_nll += nll.sum().item()
                total_n += nll.numel()
            model.train()
            return {"eval/nll_mean": total_nll / max(1,total_n)}
        return run
