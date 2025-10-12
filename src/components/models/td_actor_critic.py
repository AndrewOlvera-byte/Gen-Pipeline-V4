from __future__ import annotations
import torch
import torch.nn as nn
from src.core.registry import register

from torchrl.modules import (
    MLP,
    NormalParamExtractor,
    ProbabilisticActor,
    ValueOperator,
    TanhNormal,
)
from torchrl.modules.tensordict_module import TensorDictModule

@register("model", "td_actor_critic")
class TDActorCriticFactory:
    def build(self, cfg_node, context):
        obs_dim = context["env"]["obs_dim"] if "env" in context else context["dataset"]["train"].obs_dim
        act_dim = context["env"]["act_dim"] if "env" in context else context["dataset"]["train"].act_dim

        hidden = list(cfg_node.hidden_sizes)
        init_log_std = float(getattr(cfg_node, "init_log_std", -0.5))

        # ----- Actor: backbone → (loc, scale) → ProbabilisticActor with TanhNormal -----
        actor_backbone = MLP(in_features=obs_dim, out_features=hidden[-1], num_cells=hidden[:-1], activation_class=nn.ReLU)
        mean_std_head = nn.Linear(hidden[-1], 2 * act_dim)
        actor_head = nn.Sequential(mean_std_head, NormalParamExtractor(init_std=init_log_std))
        # Wrap backbone+head in a TensorDictModule producing "loc","scale"
        actor_param = TensorDictModule(
            nn.Sequential(actor_backbone, actor_head),
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )
        actor = ProbabilisticActor(
            module=actor_param,
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={},
            return_log_prob=True,  # writes "sample_log_prob"
            # action is automatically squashed to [-1,1]; TorchRL will rescale if you add ActionSpecTransform later
        )

        # ----- Critic: value network over observation -----
        critic_net = MLP(in_features=obs_dim, out_features=1, num_cells=hidden, activation_class=nn.ReLU)
        critic = ValueOperator(
            module=critic_net,
            in_keys=["observation"],
            out_keys=["state_value"],
        )

        # Bundle for trainer & loss
        model = nn.ModuleDict({"actor": actor, "critic": critic})
        return model
