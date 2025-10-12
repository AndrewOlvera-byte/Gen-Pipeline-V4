from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import shutil
import torch
from src.core.registry import register
from src.core.trainer_base import BaseTrainer
from src.core.progress import ProgressLogger
from src.core.run_io import dump_full_config, write_summary

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
    required_components = ["dataset", "model", "loss", "optimizer", "evaluator", "datacoll"]

    def fit(self):
        cfg = self.cfg
        model = self.components["model"].to(self.device)
        loss_fn = self.components["loss"]
        optim = self.components["optimizer"]
        evaluator = self.components.get("evaluator", None)

        # Per-run output organization
        run_dir = Path.cwd()
        reports_dir = run_dir / "reports"
        dump_full_config(cfg, reports_dir)

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

        # Progress bar & tracking
        prog = ProgressLogger(total=max_steps, desc="BC Train", outdir=reports_dir, unit="step")
        num_checkpoints = 0
        last_ckpt_path: str | None = None
        best_metric_name = "eval/nll_mean"
        best_metric_value = float("inf")
        best_step: int | None = None
        best_ckpt_path: str | None = None

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

                # progress tick
                prog.update(1)

                # logging
                if step % log_int == 0:
                    out = {k: (v.item() if hasattr(v, "item") else v) for k, v in ldict.items()}
                    out["step"] = step
                    self.logger.log(out)
                    # record series and refresh display
                    if "loss" in out:
                        prog.add_metric_point("loss", step, float(out["loss"]))
                    if "bc/nll" in out:
                        prog.add_metric_point("bc/nll", step, float(out["bc/nll"]))
                    if "bc/reg" in out:
                        prog.add_metric_point("bc/reg", step, float(out["bc/reg"]))
                    prog.update(0, loss=out.get("loss", 0.0), bc_nll=out.get("bc/nll", 0.0))

                # eval
                if step % cfg.trainer.eval_every == 0 and evaluator is not None:
                    eval_metrics = evaluator(model) or {}
                    eval_metrics["step"] = step
                    self.logger.log(eval_metrics)
                    # record eval metric and update best
                    ev = float(eval_metrics.get("eval/nll_mean", float("nan")))
                    if ev == ev:  # not NaN
                        prog.add_metric_point("eval/nll_mean", step, ev)
                        prog.update(0, eval_nll=ev)
                        if ev < best_metric_value:
                            best_metric_value = ev
                            best_step = step
                            if self.logger.should_save():
                                p = self.save_checkpoint(step)
                                num_checkpoints += 1
                                last_ckpt_path = p
                                best_ckpt = self.ckpt_dir / "best.pt"
                                try:
                                    shutil.copyfile(p, best_ckpt)
                                    best_ckpt_path = str(best_ckpt)
                                except Exception:
                                    best_ckpt_path = p

                # checkpoint
                if step % cfg.trainer.save_every == 0 and self.logger.should_save():
                    last_ckpt_path = self.save_checkpoint(step)
                    num_checkpoints += 1

        # final save
        if self.logger.should_save():
            last_ckpt_path = self.save_checkpoint(max_steps)
            num_checkpoints += 1
        prog.close()
        # save plot and summary
        plot_path = prog.save_plot("graph.png", keys=["loss", "bc/nll", "eval/nll_mean"]) or ""
        summary = {
            "exp_name": getattr(cfg.exp, "name", ""),
            "mode": getattr(cfg, "mode", ""),
            "trainer": "bc",
            "total_time_sec": prog.elapsed_seconds(),
            "num_checkpoints": num_checkpoints,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value if best_step is not None else None,
            "best_step": best_step,
            "best_checkpoint_path": best_ckpt_path,
            "final_checkpoint_path": last_ckpt_path,
            "graph_path": plot_path,
            "metrics_displayed": ["loss", "bc/nll", "eval/nll_mean"],
        }
        write_summary(reports_dir, summary, filename="summary.json")
        self.logger.finish()
