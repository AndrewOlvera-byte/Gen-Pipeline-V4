from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler
from src.core.registry import register
from src.core.trainer_base import BaseTrainer

@register("trainer", "ppo")
class PPOTrainer(BaseTrainer):
    required_components = ["env", "model", "loss", "collector", "optimizer", "logger", "evaluator"]

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

            # 3) Logging
            if env_frames % log_int < frames_per_batch:
                log_vals = {
                    "step_env": env_frames,
                    "ppo/loss_objective": float(losses["loss_objective"].detach().cpu().item()),
                    "ppo/loss_critic": float(losses["loss_critic"].detach().cpu().item()),
                    "ppo/entropy": float(losses.get("entropy", torch.tensor(0.0)).detach().cpu().item()),
                }
                self.logger.log(log_vals)

            # 4) Eval
            if env_frames % cfg.trainer.eval_every < frames_per_batch:
                em = evaluator(self.components["model"])  # evaluator accesses actor inside model
                em["step_env"] = env_frames
                self.logger.log(em)

            # 5) Checkpoint
            if env_frames % cfg.trainer.save_every < frames_per_batch and self.logger.should_save():
                self.save_checkpoint(env_frames)

            if env_frames >= max_env_frames:
                break

        if self.logger.should_save():
            self.save_checkpoint(env_frames)
        self.logger.finish()
