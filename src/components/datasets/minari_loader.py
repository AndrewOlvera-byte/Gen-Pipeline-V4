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
        if cfg_node.path and ("/" in cfg_node.path or "\\" in cfg_node.path):
            ds = minari.load_dataset_from_path(cfg_node.path)
        else:
            # if path is an ID registered in ~/.minari/datasets
            ds = minari.load_dataset(cfg_node.path) if cfg_node.path else None
            if ds is None:
                raise ValueError("Minari path/ID required for dataset=minari")

        # save both splits for evaluator; trainer/datacoll will pick "train"
        train = _MinariTorchDataset(ds, split="train")
        valid = _MinariTorchDataset(ds, split="valid")
        return {"train": train, "valid": valid}
