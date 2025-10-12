import hydra
from omegaconf import DictConfig
from src.core.bootstrap import bootstrap
from src.core.builder import build_trainer

@hydra.main(config_path="../config", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    # auto-register all trainers/components
    bootstrap()
    # build the requested trainer from the registry and run
    trainer = build_trainer(cfg)
    trainer.fit()

if __name__ == "__main__":
    main()
