from __future__ import annotations
import torch
from src.core.registry import register, get
from torchrl.envs.utils import ExplorationType, set_exploration_type

try:
    # TorchRL async collector (0.6+)
    from torchrl.collectors import MultiaSyncDataCollector as _AsyncCollector
except Exception:
    _AsyncCollector = None

from torchrl.collectors import SyncDataCollector


@register("collector", "torchrl_async")
class TorchRLAsyncCollectorFactory:
    def build(self, cfg_node, context):
        """
        Asynchronous collector using TorchRL's MultiaSyncDataCollector when available.
        Fallbacks to SyncDataCollector if async is unavailable.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor = context["model"]["actor"].to(device)
        set_exploration_type(ExplorationType.RANDOM)

        frames_per_batch = int(getattr(cfg_node, "frames_per_batch", 32768))
        total_frames = int(getattr(cfg_node, "total_frames", 1_000_000))

        # If async collector is not present, fallback to sync on provided env
        if _AsyncCollector is None:
            env = context["env"]["env"]
            return SyncDataCollector(
                env,
                policy=actor,
                frames_per_batch=frames_per_batch,
                total_frames=total_frames,
                device=device,
                storing_device=device,
            )

        # Build create_env_fn from registry & cfg so each worker can spawn its own env
        cfg = context["cfg"]

        def _maybe_get(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                return None

        def _env_cfg_node(cfg_root):
            node = _maybe_get(cfg_root, "env")
            if node is not None:
                return node
            exp = _maybe_get(cfg_root, "exp")
            if exp is not None:
                node = _maybe_get(exp, "env")
                if node is not None:
                    return node
            return None

        env_node = _env_cfg_node(cfg)
        if env_node is None or getattr(env_node, "name", None) is None:
            # Fallback: use existing env from context
            env = context["env"]["env"]
            return SyncDataCollector(
                env,
                policy=actor,
                frames_per_batch=frames_per_batch,
                total_frames=total_frames,
                device=device,
                storing_device=device,
            )

        # Create a list of environment constructors for async workers
        num_workers = int(getattr(cfg_node, "num_workers", 2))
        env_factory = get("env", env_node.name)

        def make_env():
            bundle = env_factory().build(env_node, {"cfg": cfg})
            return bundle["env"]

        create_env_fns = [make_env for _ in range(num_workers)]

        collector = _AsyncCollector(
            create_env_fn=create_env_fns,
            policy=actor,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            device=device,
            storing_device=device,
        )
        return collector


