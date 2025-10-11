from __future__ import annotations
import torch

def apply_speed_flags(cfg_speed):
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg_speed.get("allow_tf32", True))
    torch.backends.cudnn.benchmark = bool(cfg_speed.get("cudnn_benchmark", True))
    torch.backends.cudnn.deterministic = bool(cfg_speed.get("cudnn_deterministic", False))
