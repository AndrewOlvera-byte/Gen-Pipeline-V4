from __future__ import annotations
from src.core.registry import register
from torchrl.objectives import PPOLoss

@register("loss", "torchrl_ppo")
class TorchRLPPOFactory:
    def build(self, cfg_node, context):
        """
        Returns a configured PPOLoss module that:
          - reads "action", "sample_log_prob", "state_value"
          - writes standard PPO loss components
          - uses built-in value_estimator="gae" with (gamma, lmbda)
        """
        args = cfg_node.args
        clip_range = float(args.get("clip_range", 0.2))
        entropy_coef = float(args.get("entropy_coef", 0.0))
        value_coef = float(args.get("value_coef", 0.5))
        gamma = float(args.get("gamma", 0.99))
        lmbda = float(args.get("gae_lambda", 0.95))

        actor = context["model"]["actor"]
        critic = context["model"]["critic"]

        loss_module = PPOLoss(
            actor_network=actor,
            value_network=critic,
            clip_epsilon=clip_range,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            # default keys are fine: action="action", sample_log_prob="sample_log_prob", value="state_value"
        )
        # Use TorchRL's value estimator to compute advantage/returns in-place
        loss_module.set_keys(advantage_key="advantage", value_target_key="value_target")
        loss_module.gamma = gamma
        loss_module.lmbda = lmbda
        loss_module.value_estimator = "gae"  # enabling GAE inside loss module
        return loss_module
