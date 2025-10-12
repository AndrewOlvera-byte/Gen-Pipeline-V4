from __future__ import annotations
import torch, torch.nn as nn
from torch.distributions import Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from src.core.registry import register

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.0):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
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
        dist._mu, dist._std = mu, std
        return dist

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, init_log_std, dropout=0.0):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden, dropout)
        self.policy_head = DiagGaussianTanh(self.backbone.out_dim, act_dim, init_log_std)
        self.value_head = nn.Linear(self.backbone.out_dim, 1)
    def _embed(self, obs):
        return self.backbone(obs)
    def policy(self, obs):
        z = self._embed(obs); return self.policy_head(z)
    def value(self, obs):
        z = self._embed(obs); return self.value_head(z).squeeze(-1)

@register("model", "mlp_gauss_value")
class ActorCriticFactory:
    def build(self, cfg_node, context):
        # infer dims: prefer env (PPO case), else dataset
        env_bundle = context.get("env")
        if env_bundle is not None:
            obs_dim, act_dim = env_bundle["obs_dim"], env_bundle["act_dim"]
        else:
            ds = context["dataset"]["train"]
            obs_dim, act_dim = ds.obs_dim, ds.act_dim
        model = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=list(cfg_node.hidden_sizes),
            init_log_std=cfg_node.init_log_std,
            dropout=getattr(cfg_node, "dropout", 0.0),
        )
        return model
