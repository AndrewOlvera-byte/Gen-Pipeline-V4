from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler
from pathlib import Path
import shutil
from src.core.registry import register
from src.core.trainer_base import BaseTrainer
from src.core.progress import ProgressLogger
from src.core.run_io import dump_full_config, write_summary

@register("trainer", "ppo")
class PPOTrainer(BaseTrainer):
    required_components = ["env", "model", "loss", "collector", "optimizer", "evaluator"]

    def fit(self):
        cfg = self.cfg
        env = self.components["env"]["env"]
        actor = self.components["model"]["actor"].to(self.device)
        critic = self.components["model"]["critic"].to(self.device)
        loss_module = self.components["loss"].to(self.device)   # PPOLoss
        collector = self.components["collector"]                 # SyncDataCollector
        optimizer = self.components["optimizer"]
        evaluator = self.components["evaluator"]

        # AMP setup
        prec = str(getattr(cfg.speed, "precision", "bf16")).lower() if hasattr(cfg, "speed") else "bf16"
        use_amp = torch.cuda.is_available() and prec in ("bf16", "fp16")
        amp_dtype = torch.bfloat16 if prec == "bf16" else (torch.float16 if prec == "fp16" else None)
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and prec == "fp16"))

        frames_per_batch = int(cfg.collector.frames_per_batch)
        ppo_epochs = int(cfg.get("ppo", {}).get("epochs", 4))
        minibatch_size = int(cfg.get("ppo", {}).get("minibatch_size", max(frames_per_batch // 4, 1024)))

        max_env_frames = int(cfg.trainer.max_steps)
        log_int = int(getattr(cfg.logger, "log_interval", 100))
        env_frames = 0

        # Per-run output organization
        run_dir = Path.cwd()
        reports_dir = run_dir / "reports"
        dump_full_config(cfg, reports_dir)

        # Progress bar & tracking
        prog = ProgressLogger(total=max_env_frames, desc="PPO Train", outdir=reports_dir, unit="frame")
        num_checkpoints = 0
        last_ckpt_path: str | None = None
        best_metric_name = "eval/return_mean"
        best_metric_value = float("-inf")
        best_step: int | None = None
        best_ckpt_path: str | None = None

        for data in collector:
            # data is a TensorDict batch on device with "observation","action","reward","done","state_value","sample_log_prob", and "next"
            batch = data.clone()

            # 1) compute advantage / value targets in-place via loss module's GAE
            with torch.no_grad():
                loss_module.value_estimator(batch)  # writes "advantage","value_target"

            # 2) PPO epochs over minibatches
            N = batch.batch_size[0]
            idxs = np.arange(N)
            for _ in range(ppo_epochs):
                sampler = BatchSampler(SubsetRandomSampler(idxs), minibatch_size, drop_last=True)
                for mb_idx in sampler:
                    mb = batch[mb_idx]
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        losses = loss_module(mb)  # dict with "loss_objective","loss_critic","entropy" etc.
                        total_loss = (
                            losses["loss_objective"] + losses["loss_critic"] -  # PPOLoss returns positive loss terms
                            loss_module.entropy_coef * losses.get("entropy", torch.tensor(0.0, device=mb.device))
                        )

                    optimizer.zero_grad(set_to_none=True)
                    if scaler.is_enabled():
                        scaler.scale(total_loss).backward()
                        self.clip_grad(list(actor.parameters()) + list(critic.parameters()), getattr(cfg, "grad_clip", 1.0))
                        scaler.step(optimizer); scaler.update()
                    else:
                        total_loss.backward()
                        self.clip_grad(list(actor.parameters()) + list(critic.parameters()), getattr(cfg, "grad_clip", 1.0))
                        optimizer.step()

            env_frames += int(N)

            # progress tick
            prog.update(int(N))

            # 3) Logging
            if env_frames % log_int < frames_per_batch:
                log_vals = {
                    "step_env": env_frames,
                    "ppo/loss_objective": float(losses["loss_objective"].detach().cpu().item()),
                    "ppo/loss_critic": float(losses["loss_critic"].detach().cpu().item()),
                    "ppo/entropy": float(losses.get("entropy", torch.tensor(0.0)).detach().cpu().item()),
                }
                self.logger.log(log_vals)
                # record series and refresh display
                prog.add_metric_point("ppo/loss_objective", env_frames, log_vals["ppo/loss_objective"])
                prog.add_metric_point("ppo/loss_critic", env_frames, log_vals["ppo/loss_critic"]) 
                prog.add_metric_point("ppo/entropy", env_frames, log_vals["ppo/entropy"]) 
                prog.update(0, loss_obj=log_vals["ppo/loss_objective"], loss_critic=log_vals["ppo/loss_critic"], entropy=log_vals["ppo/entropy"]) 

            # 4) Eval
            if env_frames % cfg.trainer.eval_every < frames_per_batch:
                em = evaluator(self.components["model"])  # evaluator accesses actor inside model
                em["step_env"] = env_frames
                self.logger.log(em)
                # record eval metric and update best
                ret = float(em.get("eval/return_mean", float("nan")))
                if ret == ret:  # not NaN
                    prog.add_metric_point("eval/return_mean", env_frames, ret)
                    prog.update(0, eval_return=ret)
                    if ret > best_metric_value:
                        best_metric_value = ret
                        best_step = env_frames
                        if self.logger.should_save():
                            p = self.save_checkpoint(env_frames)
                            num_checkpoints += 1
                            last_ckpt_path = p
                            best_ckpt = self.ckpt_dir / "best.pt"
                            try:
                                shutil.copyfile(p, best_ckpt)
                                best_ckpt_path = str(best_ckpt)
                            except Exception:
                                best_ckpt_path = p

            # 5) Checkpoint
            if env_frames % cfg.trainer.save_every < frames_per_batch and self.logger.should_save():
                last_ckpt_path = self.save_checkpoint(env_frames)
                num_checkpoints += 1

            if env_frames >= max_env_frames:
                break

        if self.logger.should_save():
            last_ckpt_path = self.save_checkpoint(env_frames)
            num_checkpoints += 1
        prog.close()
        # save plot and summary
        plot_path = prog.save_plot("graph.png", keys=["ppo/loss_objective", "ppo/loss_critic", "eval/return_mean"]) or ""
        summary = {
            "exp_name": getattr(cfg.exp, "name", ""),
            "mode": getattr(cfg, "mode", ""),
            "trainer": "ppo",
            "total_time_sec": prog.elapsed_seconds(),
            "num_checkpoints": num_checkpoints,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value if best_step is not None else None,
            "best_step": best_step,
            "best_checkpoint_path": best_ckpt_path,
            "final_checkpoint_path": last_ckpt_path,
            "graph_path": plot_path,
            "metrics_displayed": ["ppo/loss_objective", "ppo/loss_critic", "eval/return_mean"],
        }
        write_summary(reports_dir, summary, filename="summary.json")
        self.logger.finish()
