import hydra
from omegaconf import DictConfig
from pathlib import Path
from src.core.bootstrap import bootstrap
from src.core.builder import build_trainer
from src.core.run_io import dump_full_config, write_summary


@hydra.main(config_path="../config", config_name="defaults", version_base=None)
def main(cfg: DictConfig) -> None:
    bootstrap()
    # Build trainer to get components (model, evaluator built via trainer.required_components)
    trainer = build_trainer(cfg)
    model = trainer.components.get("model", None)
    evaluator = trainer.components.get("evaluator", None)

    run_dir = Path.cwd()
    reports_dir = run_dir / "reports"
    dump_full_config(cfg, reports_dir)

    metrics = {}
    if evaluator and model:
        metrics = evaluator(model) or {}
    write_summary(reports_dir, metrics, filename="summary.json")


if __name__ == "__main__":
    main()


