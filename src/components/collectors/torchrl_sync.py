from __future__ import annotations
import torch
from src.core.registry import register
from torchrl.collectors import SyncDataCollector
from torchrl.envs.utils import ExplorationType, set_exploration_type

@register("collector", "torchrl_sync")
class TorchRLSyncCollectorFactory:
    def build(self, cfg_node, context):
        """
        Returns a TorchRL SyncDataCollector that yields TensorDict batches with
        keys: "observation","action","reward","done","next",... plus "state_value","sample_log_prob".
        """
        env = context["env"]["env"]
        actor = context["model"]["actor"]  # ProbabilisticActor (TensorDictModule)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor.to(device)

        # Ensure actor is in 'sampling' exploration mode during collection
        set_exploration_type(ExplorationType.RANDOM)

        collector = SyncDataCollector(
            env,
            policy=actor,
            frames_per_batch=int(cfg_node.frames_per_batch),
            total_frames=int(cfg_node.total_frames or 1_000_000),
            device=device,
            storing_device=device,  # keep on device for speed
        )
        return collector
