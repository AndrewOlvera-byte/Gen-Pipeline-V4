from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any, Iterable
import torch
from .speed import apply_speed_flags
from .seed import set_seed_everywhere
from .logger_router import make_logger

class BaseTrainer:
    required_components: list[str] = []

    def __init__(self, cfg, **components):
        self.cfg = cfg
        # resolve accelerator preference
        accel_pref = str(getattr(cfg.speed, "accelerator", "auto")).lower() if hasattr(cfg, "speed") else "auto"
        if accel_pref == "cpu":
            use_cuda = False
        elif accel_pref == "gpu":
            use_cuda = torch.cuda.is_available()
        else:  # auto
            use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.components: Dict[str, Any] = components
        apply_speed_flags(cfg.speed)
        # resolve seed from flat or nested config
        def _maybe_get(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                return None
        seed_node = _maybe_get(cfg, "seed") or _maybe_get(_maybe_get(cfg, "exp"), "seed")
        seed_value = getattr(seed_node, "seed", 42) if seed_node is not None else 42
        try:
            seed_int = int(seed_value)
        except Exception:
            seed_int = 42
        set_seed_everywhere(seed_int)
        self.logger = make_logger(cfg)
        # resolve trainer cfg for checkpoint dir
        def _maybe_get(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                return None
        trainer_node = _maybe_get(cfg, "trainer") or _maybe_get(_maybe_get(cfg, "exp"), "trainer")
        ckpt_dir = getattr(trainer_node, "checkpoint_dir", "outputs/ckpts") if trainer_node is not None else "outputs/ckpts"
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def to_device(self, batch: Any) -> Any:
        """Recursively move tensors in a nested structure to device."""
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: self.to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(self.to_device(v) for v in batch)
        return batch

    def clip_grad(self, params, max_norm: float | None):
        if max_norm and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm)

    def save_checkpoint(self, step: int, extra: Dict[str, Any] | None = None):
        state = {"step": step}
        for name, obj in self.components.items():
            if hasattr(obj, "state_dict"):
                state[name] = obj.state_dict()
        if extra:
            state.update(extra)
        path = self.ckpt_dir / f"step_{step}.pt"
        torch.save(state, path)
        return str(path)

    def fit(self):
        """
        Subclasses:
         - build loaders/iterators from components
         - for step in range(max_steps): compute loss -> backward -> step
         - use logger.log() and save_checkpoint() as needed
        """
        raise NotImplementedError
