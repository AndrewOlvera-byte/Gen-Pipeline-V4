from __future__ import annotations
from typing import Any, Dict
from .registry import REG, get
from .trainer_base import BaseTrainer

def _cfg_node(cfg: Any, kind: str) -> Any:
    """
    Retrieve the config node for a given kind (e.g., "trainer", "dataset").
    Supports both flattened configs (cfg.trainer) and nested experiment shapes
    (cfg.exp.trainer or cfg.exp.trainer.trainer) to be resilient to packaging.
    """
    # First try flat shape: cfg.trainer
    node = getattr(cfg, kind, None)
    if getattr(node, "name", None) is not None:
        return node

    # Then try nested under exp: cfg.exp.trainer
    exp = getattr(cfg, "exp", None)
    if exp is not None:
        node = getattr(exp, kind, None)
        if getattr(node, "name", None) is not None:
            return node
        # Some configs may nest group name again, e.g., cfg.exp.trainer.trainer
        inner = getattr(node, kind, None)
        if getattr(inner, "name", None) is not None:
            return inner

    return None

def build_component(kind: str, cfg: Any, context: Dict[str, Any]) -> Any:
    node = _cfg_node(cfg, kind)
    if node is None or getattr(node, "name", None) is None:
        return None
    factory = get(kind, node.name)
    if hasattr(factory, "build"):
        return factory().build(node, context)
    return factory(node, context)

def build_trainer(cfg: Any) -> BaseTrainer:
    trainer_node = _cfg_node(cfg, "trainer")
    if trainer_node is None or getattr(trainer_node, "name", None) is None:
        raise KeyError("Trainer config not found. Expected 'trainer.name' in the composed config.")
    trainer_name = trainer_node.name
    TrainerCls = get("trainer", trainer_name)

    required = getattr(TrainerCls, "required_components", [])
    context: Dict[str, Any] = {"cfg": cfg}

    for kind in required:
        obj = build_component(kind, cfg, context)
        if obj is not None:
            context[kind] = obj

    kwargs = {k: v for k, v in context.items() if k in required}
    trainer = TrainerCls(cfg, **kwargs)
    return trainer
