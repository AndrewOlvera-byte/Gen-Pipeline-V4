from __future__ import annotations
import torch, torch.nn as nn
from torch.distributions import Normal, Independent, TransformedDistribution, constraints
from torch.distributions.transforms import TanhTransform
from src.core.registry import register

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.0, norm="layernorm"):
        super().__init__()
        layers, last = [], in_dim
        Norm = {"layernorm": nn.LayerNorm, "none": None}.get(norm, nn.LayerNorm)
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            if Norm: layers.append(Norm(h))
            if dropout > 0: layers.append(nn.Dropout(dropout))
            last = h
        self.net = nn.Sequential(*layers)
        self.out_dim = last
    def forward(self, x): return self.net(x)

class DiagGaussianTanh(nn.Module):
    def __init__(self, in_dim, act_dim, init_log_std=-0.5):
        super().__init__()
        self.mu = nn.Linear(in_dim, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * init_log_std)

    def forward(self, z):
        mu = self.mu(z)
        std = self.log_std.clamp(-5, 2).exp()
        base = Normal(mu, std)
        dist = TransformedDistribution(base, [TanhTransform(cache_size=1)])
        dist = Independent(dist, 1)
        # stash for logging
        dist._mu, dist._std = mu, std
        return dist

class PolicyModule(nn.Module):
    """Unified interface: policy(obs) -> dist"""
    def __init__(self, obs_dim, act_dim, hidden, init_log_std, dropout=0.0, norm="layernorm"):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden, dropout, norm)
        self.head = DiagGaussianTanh(self.backbone.out_dim, act_dim, init_log_std)
    def policy(self, obs):  # BC & PPO both call this
        z = self.backbone(obs)
        return self.head(z)

@register("model", "mlp_gauss")
class MLPDiagGaussFactory:
    def build(self, cfg_node, context):
        # infer dims from dataset if env is absent
        ds_bundle = context.get("dataset")
        if ds_bundle is not None:
            obs_dim = ds_bundle["train"].obs_dim
            act_dim = ds_bundle["train"].act_dim
        else:
            # (When used in RL, you’d pull dims from env/action_space here)
            raise ValueError("For BC, dataset must be present to infer dims.")
        model = PolicyModule(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=cfg_node.hidden_sizes,
            init_log_std=cfg_node.init_log_std,
            dropout=getattr(cfg_node, "dropout", 0.0),
            norm=getattr(cfg_node, "norm_layer", "layernorm"),
        )
        return model
