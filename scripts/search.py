import hydra
from omegaconf import DictConfig
from src.core.bootstrap import bootstrap
from src.core.builder import build_trainer

@hydra.main(config_path="../config", config_name="defaults", version_base=None)
def main(cfg: DictConfig) -> float:
    # auto-register all trainers/components
    bootstrap()
    # build and run
    trainer = build_trainer(cfg)
    # ask trainers to return final metric if possible; else evaluate once
    result = trainer.fit()  # may return a float (objective)
    if isinstance(result, (int, float)):
        return float(result)

    # Fallback: compute metric via evaluator
    # Both BC and PPO trainers already hold evaluator & model in components
    evaluator = trainer.components.get("evaluator", None)
    model = trainer.components.get("model", None)
    objective_key = getattr(getattr(cfg, "hpo", {}), "objective_key", "") or ""
    if evaluator and model and objective_key:
        metrics = evaluator(model) or {}
        if objective_key in metrics:
            return float(metrics[objective_key])
    # If no metric found, return 0.0 so the trial doesn't crash
    return 0.0

if __name__ == "__main__":
    main()
