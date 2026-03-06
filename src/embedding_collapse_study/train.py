"""
SimCLR training loop with geometry logging.
"""
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    # TODO: implement training loop
    pass


if __name__ == "__main__":
    main()
