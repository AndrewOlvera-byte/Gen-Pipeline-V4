from __future__ import annotations
from typing import Optional, Tuple
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


class _FlatTorchDataset(Dataset):
    def __init__(self, obs: np.ndarray, acts: np.ndarray, split_ratio: float = 0.9, split: str = "train"):
        n = min(len(obs), len(acts))
        obs = obs[:n]
        acts = acts[:n]
        idx = int(split_ratio * n)
        if split == "train":
            self.obs = torch.as_tensor(obs[:idx], dtype=torch.float32)
            self.acts = torch.as_tensor(acts[:idx], dtype=torch.float32)
        else:
            self.obs = torch.as_tensor(obs[idx:], dtype=torch.float32)
            self.acts = torch.as_tensor(acts[idx:], dtype=torch.float32)
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
        # - If cfg_node.path points to a directory, try local HDF5 or Minari path
        # - Else treat it as Minari dataset id or HDF5 file
        import os
        from pathlib import Path
        try:
            from hydra.utils import get_original_cwd
            repo_root = Path(get_original_cwd())
        except Exception:
            repo_root = Path.cwd()

        def _try_load_local_hdf5(base: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            import h5py
            candidates = [
                base / "data" / "main_data.hdf5",
                base / "main_data.hdf5",
                base if base.suffix.lower() in (".h5", ".hdf5") else None,
            ]
            candidates = [c for c in candidates if c is not None]
            for fp in candidates:
                if not fp.exists():
                    continue
                try:
                    with h5py.File(fp, "r") as f:
                        keys = list(f.keys())
                        # try common key names
                        obs_key = next((k for k in ["observations", "obs", "states"] if k in keys), None)
                        act_key = next((k for k in ["actions", "acts", "action"] if k in keys), None)
                        if obs_key is None or act_key is None:
                            # try nested under a group
                            for k in keys:
                                try:
                                    sub = f[k]
                                    if hasattr(sub, "keys"):
                                        skeys = list(sub.keys())
                                        o = next((kk for kk in ["observations", "obs", "states"] if kk in skeys), None)
                                        a = next((kk for kk in ["actions", "acts", "action"] if kk in skeys), None)
                                        if o and a:
                                            obs = sub[o][...]
                                            acts = sub[a][...]
                                            return obs, acts
                                except Exception:
                                    continue
                            continue
                        obs = f[obs_key][...]
                        acts = f[act_key][...]
                        return obs, acts
                except Exception:
                    continue
            return None

        if cfg_node.path:
            p = Path(str(cfg_node.path))
            if not p.is_absolute():
                p = repo_root / p
            # 1) Try direct local flat HDF5
            flat = _try_load_local_hdf5(p)
            if flat is not None:
                obs, acts = flat
                train = _FlatTorchDataset(obs, acts, split="train")
                valid = _FlatTorchDataset(obs, acts, split="valid")
                return {"train": train, "valid": valid}

            # 2) Else, try Minari loaders
            if p.is_dir():
                try:
                    ds = minari.load_dataset_from_path(str(p))  # type: ignore[attr-defined]
                except Exception:
                    ds = minari.load_dataset(str(p))
            else:
                ds = minari.load_dataset(str(p))
        else:
            raise ValueError("Minari path/ID required for dataset=minari")

        # save both splits for evaluator; trainer/datacoll will pick "train"
        train = _MinariTorchDataset(ds, split="train")
        valid = _MinariTorchDataset(ds, split="valid")
        return {"train": train, "valid": valid}
