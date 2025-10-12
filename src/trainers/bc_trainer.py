from __future__ import annotations
from typing import Dict, Any
import torch
from src.core.registry import register
from src.core.trainer_base import BaseTrainer

@register("trainer", "bc")
class BCTrainer(BaseTrainer):
    """
    Offline BC / SL-style trainer:
      required components:
        - dataset: {"train": Dataset, "valid": Dataset}
        - datacoll: {"train_loader", "valid_loader"}  (optional; will build loaders if absent)
        - model: nn.Module with .policy(obs) -> Distribution
        - loss: callable(model, batch) -> dict {"loss": Tensor, ...}
        - optimizer: torch.optim.Optimizer
        - logger: BaseLogger (.log, .should_save, .finish)
        - evaluator: callable(model) -> dict  (optional)
    """
    required_components = ["dataset", "model", "loss", "optimizer", "logger", "evaluator", "datacoll"]

    def fit(self):
        cfg = self.cfg
        model = self.components["model"].to(self.device)
        loss_fn = self.components["loss"]
        optim = self.components["optimizer"]
        evaluator = self.components.get("evaluator", None)

        # Data loaders: use datacoll if provided, otherwise build quick loaders here.
        if self.components.get("datacoll", None) is not None:
            loaders = self.components["datacoll"]
            train_loader = loaders["train_loader"]
        else:
            from torch.utils.data import DataLoader
            ds = self.components["dataset"]["train"]
            train_loader = DataLoader(
                ds, batch_size=cfg.dataset.batch_size, shuffle=True,
                num_workers=cfg.dataset.num_workers, pin_memory=True, drop_last=True
            )

        # AMP setup
        prec = str(getattr(cfg.speed, "precision", "bf16")).lower() if hasattr(cfg, "speed") else "bf16"
        use_amp = torch.cuda.is_available() and prec in ("bf16", "fp16")
        amp_dtype = torch.bfloat16 if prec == "bf16" else (torch.float16 if prec == "fp16" else None)
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and prec == "fp16"))

        max_steps = int(cfg.trainer.max_steps)
        log_int = int(getattr(cfg.logger, "log_interval", 100))
        step = 0
        model.train()

        while step < max_steps:
            for batch in train_loader:
                if step >= max_steps:
                    break
                batch = self.to_device(batch)

                # forward + loss
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    ldict: Dict[str, Any] = loss_fn(model, batch)
                    loss = ldict["loss"]

                # backward
                optim.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    self.clip_grad(model.parameters(), getattr(cfg, "grad_clip", 1.0))
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    self.clip_grad(model.parameters(), getattr(cfg, "grad_clip", 1.0))
                    optim.step()

                step += 1

                # logging
                if step % log_int == 0:
                    out = {k: (v.item() if hasattr(v, "item") else v) for k, v in ldict.items()}
                    out["step"] = step
                    self.logger.log(out)

                # eval
                if step % cfg.trainer.eval_every == 0 and evaluator is not None:
                    eval_metrics = evaluator(model) or {}
                    eval_metrics["step"] = step
                    self.logger.log(eval_metrics)

                # checkpoint
                if step % cfg.trainer.save_every == 0 and self.logger.should_save():
                    self.save_checkpoint(step)

        # final save
        if self.logger.should_save():
            self.save_checkpoint(max_steps)
        self.logger.finish()
