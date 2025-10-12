from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from src.core.registry import register

class _MinariTorchDataset(Dataset):
    def __init__(self, ds, split_ratio=0.9, split="train"):
        # flatten episodes into (obs, act)
        obs_list, act_list = [], []
        for ep in ds.iterate_episodes():
            # align shapes: actions length == observations-1 usually
            obs = ep["observations"]
            acts = ep["actions"]
            T = min(len(acts), len(obs) - 1)
            if T <= 0: continue
            obs_list.append(obs[:T])
            act_list.append(acts[:T])

        obs = np.concatenate(obs_list, axis=0)
        acts = np.concatenate(act_list, axis=0)

        # split
        N = len(obs)
        idx = int(split_ratio * N)
        if split == "train":
            self.obs = torch.as_tensor(obs[:idx], dtype=torch.float32)
            self.acts = torch.as_tensor(acts[:idx], dtype=torch.float32)
        else:
            self.obs = torch.as_tensor(obs[idx:], dtype=torch.float32)
            self.acts = torch.as_tensor(acts[idx:], dtype=torch.float32)

        # expose dims for model build
        self.obs_dim = self.obs.shape[-1]
        self.act_dim = self.acts.shape[-1]

    def __len__(self): return self.obs.shape[0]
    def __getitem__(self, i): return {"obs": self.obs[i], "act": self.acts[i]}

@register("dataset", "minari")
class MinariDatasetFactory:
    def build(self, cfg_node, context):
        """
        cfg_node:
          name: "minari"
          path: str | None (dataset directory) or dataset id
          batch_size, num_workers (used later by datacoll/dataloader)
        """
        import minari
        # Resolve dataset path:
        # - If cfg_node.path points to a directory, load from path
        # - Else treat it as Minari dataset id
        import os
        from pathlib import Path
        try:
            from hydra.utils import get_original_cwd
            repo_root = Path(get_original_cwd())
        except Exception:
            repo_root = Path.cwd()

        if cfg_node.path:
            p = Path(str(cfg_node.path))
            if not p.is_absolute():
                p = repo_root / p
            if p.is_dir():
                ds = minari.load_dataset_from_path(str(p))
            else:
                ds = minari.load_dataset(str(p))
        else:
            raise ValueError("Minari path/ID required for dataset=minari")

        # save both splits for evaluator; trainer/datacoll will pick "train"
        train = _MinariTorchDataset(ds, split="train")
        valid = _MinariTorchDataset(ds, split="valid")
        return {"train": train, "valid": valid}
