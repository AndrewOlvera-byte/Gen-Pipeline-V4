from __future__ import annotations
from typing import Dict, Any, Tuple
from pathlib import Path
import torch
from src.core.registry import register
from src.core.trainer_base import BaseTrainer
from src.core.progress import ProgressLogger
from src.core.run_io import dump_full_config, write_summary


@register("trainer", "bc")
class BCTrainer(BaseTrainer):
    """
    Offline BC / SL trainer.
    required components: dataset, model, loss, optimizer, evaluator, datacoll (optional)
    """
    required_components = ["dataset", "model", "loss", "optimizer", "evaluator", "datacoll"]

    # ===== Private modular methods =====
    def _resolve_cfg_nodes(self) -> Tuple[Any, Any]:
        def _maybe_get(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                return None
        def _get_node(cfg_root, key):
            node = _maybe_get(cfg_root, key)
            if node is not None:
                return node
            exp = _maybe_get(cfg_root, "exp")
            if exp is not None:
                node = _maybe_get(exp, key)
                if node is not None:
                    inner = _maybe_get(node, key)
                    return inner or node
            return None
        return _get_node(self.cfg, "trainer"), _get_node(self.cfg, "logger")

    def _prepare_model_and_io(self):
        model = self.components["model"].to(self.device)
        loss_fn = self.components["loss"]
        optimizer = self.components["optimizer"]
        evaluator = self.components.get("evaluator", None)
        return model, loss_fn, optimizer, evaluator

    def _build_train_loader(self):
        if self.components.get("datacoll", None) is not None:
            return self.components["datacoll"]["train_loader"]
        from torch.utils.data import DataLoader
        ds = self.components["dataset"]["train"]
        # resolve dataset cfg (flat or nested)
        def _maybe_get(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                return None
        def _get_dataset_cfg(cfg):
            ds = _maybe_get(cfg, "dataset")
            if ds is not None:
                return ds
            exp = _maybe_get(cfg, "exp")
            if exp is not None:
                ds = _maybe_get(exp, "dataset")
                if ds is not None:
                    return ds
            return None
        ds_cfg = _get_dataset_cfg(self.cfg)
        bs = int(getattr(ds_cfg, "batch_size", 32)) if ds_cfg is not None else 32
        nw = int(getattr(ds_cfg, "num_workers", 0)) if ds_cfg is not None else 0
        return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)

    def _forward_loss(self, model, loss_fn, batch, amp_dtype, use_amp) -> Dict[str, Any]:
        # augmentation and normalization hooks
        batch = self.apply_augmentations(batch)
        batch = self.apply_normalization(batch)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            ldict: Dict[str, Any] = loss_fn(model, batch)
            # optional extra regularization term
            extra_reg = self.compute_regularization()
            if torch.is_tensor(extra_reg) and extra_reg.ne(0).any():
                ldict["loss"] = ldict["loss"] + extra_reg
                ldict.setdefault("extra/reg", extra_reg)
        return ldict

    def _backward_step(self, model, optimizer, loss: torch.Tensor, scaler, grad_clip: float):
        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            self.clip_grad(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            self.clip_grad(model.parameters(), grad_clip)
            optimizer.step()

    def _log_step(self, step: int, processed: int, ldict: Dict[str, Any], model, optimizer, prog: ProgressLogger):
        out = {k: (v.item() if hasattr(v, "item") else v) for k, v in ldict.items()}
        out["step"] = step
        out["samples_per_sec"] = prog.samples_per_sec(processed)
        with torch.no_grad():
            try:
                g2 = 0.0
                p2 = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        g = p.grad.detach()
                        g2 += float(g.norm(2).item() ** 2)
                    d = p.detach()
                    p2 += float(d.norm(2).item() ** 2)
                out["grad_norm"] = float(g2 ** 0.5)
                out["param_norm"] = float(p2 ** 0.5)
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    out["gpu_mem_gb"] = float(torch.cuda.memory_allocated() / (1024 ** 3))
            except Exception:
                pass
        out["wall_clock"] = float(prog.elapsed_seconds())
        try:
            out["lr"] = float(optimizer.param_groups[0]["lr"])
        except Exception:
            pass
        self.logger.log(out)
        if "loss" in out:
            prog.add_metric_point("loss", step, float(out["loss"]))
        if "bc/nll" in out:
            prog.add_metric_point("bc/nll", step, float(out["bc/nll"]))
        if "bc/reg" in out:
            prog.add_metric_point("bc/reg", step, float(out["bc/reg"]))
        prog.update(0, loss=out.get("loss", 0.0), bc_nll=out.get("bc/nll", 0.0))

    def _maybe_eval(self, step: int, eval_every: int, evaluator, model, prog: ProgressLogger, best: Dict[str, Any]):
        if evaluator is None:
            return
        if step % eval_every != 0:
            return
        eval_metrics = evaluator(model) or {}
        eval_metrics["step"] = step
        self.logger.log(eval_metrics)
        ev = float(eval_metrics.get("eval/nll_mean", float("nan")))
        if ev == ev:  # not NaN
            prog.add_metric_point("eval/nll_mean", step, ev)
            prog.update(0, eval_nll=ev)
            if ev < best["value"]:
                best["value"] = ev
                best["step"] = step
                if self.logger.should_save():
                    p = self.save_checkpoint(step, objective=ev)
                    best["num_ckpts"] += 1
                    best["last_ckpt"] = p
                    best_file = self.reports_dir / "best.pt"
                    best["best_path"] = str(best_file) if best_file.exists() else p

    def _maybe_checkpoint(self, step: int, save_every: int, best: Dict[str, Any]):
        if step % save_every != 0:
            return
        if self.logger.should_save():
            p = self.save_checkpoint(step, objective=(best["value"] if best["step"] == step else None))
            best["last_ckpt"] = p
            best["num_ckpts"] += 1

    # ===== Orchestration =====
    def fit(self):
        cfg = self.cfg
        model, loss_fn, optim, evaluator = self._prepare_model_and_io()
        tr_cfg, lg_cfg = self._resolve_cfg_nodes()
        max_steps = int(getattr(tr_cfg, "max_steps", 1000))
        log_int = int(getattr(lg_cfg, "log_interval", 100))
        save_every = int(getattr(tr_cfg, "save_every", 10000))
        eval_every = int(getattr(tr_cfg, "eval_every", 1000))

        run_dir = Path.cwd()
        reports_dir = run_dir / "reports"
        dump_full_config(cfg, reports_dir)

        train_loader = self._build_train_loader()
        use_amp, amp_dtype, scaler = self.get_amp_settings()
        sched = self.build_lr_scheduler(optim, total_steps=max_steps)

        step = 0
        processed = 0
        model.train()
        prog = ProgressLogger(total=max_steps, desc="BC Train", outdir=reports_dir, unit="step")
        best = {"name": "eval/nll_mean", "value": float("inf"), "step": None, "best_path": None, "last_ckpt": None, "num_ckpts": 0}

        for batch in train_loader:
            if step >= max_steps:
                break
            batch = self.to_device(batch)
            ldict = self._forward_loss(model, loss_fn, batch, amp_dtype, use_amp)
            loss = ldict["loss"]
            self._backward_step(model, optim, loss, scaler, getattr(cfg, "grad_clip", 1.0))
            if sched is not None:
                try:
                    sched.step()
                except Exception:
                    pass

            step += 1
            processed += len(batch.get("obs", [])) if isinstance(batch, dict) else 1
            prog.update(1)

            if step % log_int == 0:
                self._log_step(step, processed, ldict, model, optim, prog)

            self._maybe_eval(step, eval_every, evaluator, model, prog, best)
            self._maybe_checkpoint(step, save_every, best)

        if self.logger.should_save():
            p = self.save_checkpoint(max_steps, objective=(best["value"] if best["step"] == max_steps else None))
            best["last_ckpt"] = p
            best["num_ckpts"] += 1
        prog.close()

        plot_path = prog.save_plot("graph.png", keys=["loss", "bc/nll", "eval/nll_mean"]) or ""
        summary = {
            "exp_name": getattr(cfg.exp, "name", ""),
            "mode": getattr(cfg, "mode", ""),
            "trainer": "bc",
            "total_time_sec": prog.elapsed_seconds(),
            "num_checkpoints": best["num_ckpts"],
            "best_metric_name": best["name"],
            "best_metric_value": best["value"] if best["step"] is not None else None,
            "best_step": best["step"],
            "best_checkpoint_path": best["best_path"],
            "final_checkpoint_path": best["last_ckpt"],
            "graph_path": plot_path,
            "metrics_displayed": ["loss", "bc/nll", "eval/nll_mean"],
        }
        write_summary(reports_dir, summary, filename="summary.json")
        self.logger.finish()
