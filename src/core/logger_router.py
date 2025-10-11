from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any

class BaseLogger:
    def log(self, metrics: Dict[str, Any]): ...
    def artifact(self, path: str, name: str | None = None): ...
    def finish(self): ...
    def should_save(self) -> bool: return True

class WandbLogger(BaseLogger):
    def __init__(self, cfg):
        import wandb
        self._wandb = wandb
        name = cfg.logger.get("run_name") or cfg.exp.name or "run"
        self._wandb.init(
            project=cfg.logger.project,
            name=name,
            mode=cfg.logger.mode,
            tags=cfg.exp.get("tags", []),
            notes=cfg.exp.get("notes", "")
        )
        # save config snapshot
        try:
            self._wandb.config.update(dict(cfg), allow_val_change=True)
        except Exception:
            pass
        self._policy = cfg.mode if hasattr(cfg, "mode") else "train"

    def log(self, metrics: Dict[str, Any]):
        self._wandb.log(metrics)

    def artifact(self, path: str, name: str | None = None):
        art = self._wandb.Artifact(name or Path(path).name, type="artifact")
        art.add_file(path)
        self._wandb.log_artifact(art)

    def finish(self):
        self._wandb.finish()

    def should_save(self) -> bool:
        return self._policy == "train"

class OfflineLogger(BaseLogger):
    """Simple JSONL logger in outputs dir."""
    def __init__(self, cfg):
        outdir = Path("outputs") / (cfg.exp.name or "run")
        outdir.mkdir(parents=True, exist_ok=True)
        self._path = outdir / "metrics.jsonl"
        self._fh = open(self._path, "a", buffering=1)
        self._policy = cfg.mode if hasattr(cfg, "mode") else "train"

        # store a minimal config snapshot for reproducibility
        with open(outdir / "config_snapshot.json", "w") as f:
            try:
                json.dump(dict(cfg), f, indent=2, default=str)
            except Exception:
                f.write("{}")

    def log(self, metrics: Dict[str, Any]):
        self._fh.write(json.dumps(metrics) + "\n")

    def artifact(self, path: str, name: str | None = None):
        # no-op for simple offline logger
        return

    def finish(self):
        self._fh.close()

    def should_save(self) -> bool:
        return self._policy == "train"

def make_logger(cfg) -> BaseLogger:
    name = cfg.logger.name
    if name == "wandb":
        return WandbLogger(cfg)
    if name == "offline" or name == "csv":
        return OfflineLogger(cfg)
    # default: no-op logger
    return BaseLogger()
