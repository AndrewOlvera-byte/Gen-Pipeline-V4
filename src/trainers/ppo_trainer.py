from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler
from pathlib import Path
from typing import Dict, Any, Tuple
from src.core.registry import register
from src.core.trainer_base import BaseTrainer
from src.core.progress import ProgressLogger
from src.core.run_io import dump_full_config, write_summary


@register("trainer", "ppo")
class PPOTrainer(BaseTrainer):
    required_components = ["env", "model", "loss", "collector", "optimizer", "evaluator"]

    # ===== Private modular methods =====
    def _prepare_modules(self):
        actor = self.components["model"]["actor"].to(self.device)
        critic = self.components["model"]["critic"].to(self.device)
        loss_module = self.components["loss"].to(self.device)
        collector = self.components["collector"]
        optimizer = self.components["optimizer"]
        evaluator = self.components["evaluator"]
        return actor, critic, loss_module, collector, optimizer, evaluator

    def _ppo_hparams(self) -> Tuple[int, int, int, int, int, int]:
        cfg = self.cfg
        frames_per_batch = int(getattr(getattr(cfg, "collector", {}), "frames_per_batch", 8192))
        ppo_cfg = getattr(cfg, "ppo", {})
        ppo_epochs = int(getattr(ppo_cfg, "epochs", 4))
        minibatch_size = int(getattr(ppo_cfg, "minibatch_size", max(frames_per_batch // 4, 1024)))
        max_env_frames = int(getattr(getattr(cfg, "trainer", {}), "max_steps", 1_000_000))
        log_int = int(getattr(getattr(cfg, "logger", {}), "log_interval", 100))
        return frames_per_batch, ppo_epochs, minibatch_size, max_env_frames, log_int, int(getattr(getattr(cfg, "trainer", {}), "save_every", 10000))

    def _eval_every(self) -> int:
        return int(getattr(getattr(self.cfg, "trainer", {}), "eval_every", 1000))

    def _compute_advantages(self, loss_module, batch):
        with torch.no_grad():
            loss_module.value_estimator(batch)

    def _ppo_update(self, actor, critic, loss_module, optimizer, scaler, batch, ppo_epochs: int, minibatch_size: int, use_amp: bool, amp_dtype):
        N = batch.batch_size[0]
        idxs = np.arange(N)
        last_losses: Dict[str, torch.Tensor] = {}
        for _ in range(ppo_epochs):
            sampler = BatchSampler(SubsetRandomSampler(idxs), minibatch_size, drop_last=True)
            for mb_idx in sampler:
                mb = batch[mb_idx]
                mb = self.apply_normalization(mb)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    losses = loss_module(mb)
                    total_loss = (
                        losses["loss_objective"] + losses["loss_critic"] -
                        loss_module.entropy_coef * losses.get("entropy", torch.tensor(0.0, device=mb.device))
                    )
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                    self.clip_grad(list(actor.parameters()) + list(critic.parameters()), getattr(self.cfg, "grad_clip", 1.0))
                    scaler.step(optimizer); scaler.update()
                else:
                    total_loss.backward()
                    self.clip_grad(list(actor.parameters()) + list(critic.parameters()), getattr(self.cfg, "grad_clip", 1.0))
                    optimizer.step()
                last_losses = losses
        return int(N), last_losses

    def _log_step(self, env_frames: int, frames_per_batch: int, log_int: int, losses: Dict[str, torch.Tensor], actor, critic, optimizer, prog: ProgressLogger):
        if env_frames % log_int >= frames_per_batch:
            return
        log_vals = {
            "step_env": env_frames,
            "ppo/loss_objective": float(losses["loss_objective"].detach().cpu().item()),
            "ppo/loss_critic": float(losses["loss_critic"].detach().cpu().item()),
            "ppo/entropy": float(losses.get("entropy", torch.tensor(0.0)).detach().cpu().item()),
        }
        log_vals["samples_per_sec"] = prog.samples_per_sec(env_frames)
        try:
            total_norm_g = 0.0
            total_norm_p = 0.0
            for p in list(actor.parameters()) + list(critic.parameters()):
                if p.grad is not None:
                    total_norm_g += float(p.grad.detach().norm(2).item() ** 2)
                total_norm_p += float(p.detach().norm(2).item() ** 2)
            log_vals["grad_norm"] = float(total_norm_g ** 0.5)
            log_vals["param_norm"] = float(total_norm_p ** 0.5)
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                log_vals["gpu_mem_gb"] = float(torch.cuda.memory_allocated() / (1024 ** 3))
        except Exception:
            pass
        try:
            log_vals["lr"] = float(optimizer.param_groups[0]["lr"])
        except Exception:
            pass
        self.logger.log(log_vals)
        prog.add_metric_point("ppo/loss_objective", env_frames, log_vals["ppo/loss_objective"])
        prog.add_metric_point("ppo/loss_critic", env_frames, log_vals["ppo/loss_critic"]) 
        prog.add_metric_point("ppo/entropy", env_frames, log_vals["ppo/entropy"]) 
        prog.update(0, loss_obj=log_vals["ppo/loss_objective"], loss_critic=log_vals["ppo/loss_critic"], entropy=log_vals["ppo/entropy"]) 

    def _maybe_eval(self, env_frames: int, frames_per_batch: int, evaluator, prog: ProgressLogger, best: Dict[str, Any]):
        if env_frames % self._eval_every() >= frames_per_batch:
            return
        em = evaluator(self.components["model"])  # evaluator accesses actor inside model
        em["step_env"] = env_frames
        self.logger.log(em)
        ret = float(em.get("eval/return_mean", float("nan")))
        if ret == ret:  # not NaN
            prog.add_metric_point("eval/return_mean", env_frames, ret)
            prog.update(0, eval_return=ret)
            if ret > best["value"]:
                best["value"] = ret
                best["step"] = env_frames
                if self.logger.should_save():
                    p = self.save_checkpoint(env_frames, objective=ret)
                    best["num_ckpts"] += 1
                    best["last_ckpt"] = p
                    best_file = Path.cwd() / "reports" / "best.pt"
                    best["best_path"] = str(best_file) if best_file.exists() else p

    def _maybe_checkpoint(self, env_frames: int, frames_per_batch: int, best: Dict[str, Any]):
        save_every = int(getattr(getattr(self.cfg, "trainer", {}), "save_every", 10000))
        if env_frames % save_every >= frames_per_batch:
            return
        if self.logger.should_save():
            p = self.save_checkpoint(env_frames, objective=(best["value"] if best["step"] == env_frames else None))
            best["last_ckpt"] = p
            best["num_ckpts"] += 1

    # ===== Orchestration =====
    def fit(self):
        cfg = self.cfg
        actor, critic, loss_module, collector, optimizer, evaluator = self._prepare_modules()
        frames_per_batch, ppo_epochs, minibatch_size, max_env_frames, log_int, save_every = self._ppo_hparams()
        use_amp, amp_dtype, scaler = self.get_amp_settings()

        run_dir = Path.cwd()
        reports_dir = run_dir / "reports"
        dump_full_config(cfg, reports_dir)

        # Optional LR scheduler
        sched = self.build_lr_scheduler(optimizer, total_steps=max_env_frames)

        prog = ProgressLogger(total=max_env_frames, desc="PPO Train", outdir=reports_dir, unit="frame")
        best = {"name": "eval/return_mean", "value": float("-inf"), "step": None, "best_path": None, "last_ckpt": None, "num_ckpts": 0}
        env_frames = 0

        for data in collector:
            batch = data.clone()
            self._compute_advantages(loss_module, batch)
            N, last_losses = self._ppo_update(actor, critic, loss_module, optimizer, scaler, batch, ppo_epochs, minibatch_size, use_amp, amp_dtype)

            # scheduler step after each batch update cycle
            try:
                if sched is not None:
                    sched.step()
            except Exception:
                pass

            env_frames += int(N)
            prog.update(int(N))
            self._log_step(env_frames, frames_per_batch, log_int, last_losses, actor, critic, optimizer, prog)
            self._maybe_eval(env_frames, frames_per_batch, evaluator, prog, best)
            self._maybe_checkpoint(env_frames, frames_per_batch, best)

            if env_frames >= max_env_frames:
                break

        if self.logger.should_save():
            p = self.save_checkpoint(env_frames, objective=(best["value"] if best["step"] == env_frames else None))
            best["last_ckpt"] = p
            best["num_ckpts"] += 1
        prog.close()

        plot_path = prog.save_plot("graph.png", keys=["ppo/loss_objective", "ppo/loss_critic", "eval/return_mean"]) or ""
        summary = {
            "exp_name": getattr(cfg.exp, "name", ""),
            "mode": getattr(cfg, "mode", ""),
            "trainer": "ppo",
            "total_time_sec": prog.elapsed_seconds(),
            "num_checkpoints": best["num_ckpts"],
            "best_metric_name": best["name"],
            "best_metric_value": best["value"] if best["step"] is not None else None,
            "best_step": best["step"],
            "best_checkpoint_path": best["best_path"],
            "final_checkpoint_path": best["last_ckpt"],
            "graph_path": plot_path,
            "metrics_displayed": ["ppo/loss_objective", "ppo/loss_critic", "eval/return_mean"],
        }
        write_summary(reports_dir, summary, filename="summary.json")
        self.logger.finish()
