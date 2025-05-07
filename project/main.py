import os
import hydra

from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


@hydra.main(version_base=None, config_path="config", config_name="default")
def do_main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    trainer = Trainer(cfg)
    trainer.setup()
    trainer.load_model(".scratch/logs/only_AI/6/last_model.zip")
    trainer.train()


if __name__ == "__main__":
    do_main()
