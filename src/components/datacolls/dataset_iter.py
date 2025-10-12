from __future__ import annotations
from torch.utils.data import DataLoader
from src.core.registry import register

@register("datacoll", "dataset_iter")
class DatasetIteratorFactory:
    def build(self, cfg_node, context) -> dict:
        """
        Wraps a torch Dataset into a DataLoader iterator.
        Expects context["dataset"] == {"train": Dataset, "valid": Dataset}
        cfg_node is optional for BC; we take batch_size from cfg.dataset.
        """
        cfg = context["cfg"]
        train_ds = context["dataset"]["train"]
        valid_ds = context["dataset"]["valid"]

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=cfg.dataset.batch_size,
            shuffle=False,
            num_workers=cfg.dataset.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return {"train_loader": train_loader, "valid_loader": valid_loader}
