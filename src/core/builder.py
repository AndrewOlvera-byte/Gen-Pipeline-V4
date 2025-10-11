from __future__ import annotations
from typing import Any, Dict
from .registry import REG, get
from .trainer_base import BaseTrainer

def _cfg_node(cfg: Any, kind: str) -> Any:
    return getattr(cfg, kind, None)

def build_component(kind: str, cfg: Any, context: Dict[str, Any]) -> Any:
    node = _cfg_node(cfg, kind)
    if node is None or getattr(node, "name", None) is None:
        return None
    factory = get(kind, node.name)
    if hasattr(factory, "build"):
        return factory().build(node, context)
    return factory(node, context)

def build_trainer(cfg: Any) -> BaseTrainer:
    trainer_name = cfg.trainer.name
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
