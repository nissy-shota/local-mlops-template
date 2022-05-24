import logging
import os

import hydra
import mlflow
import torch
import torchvision
import torchvision.transforms as transforms
from dotenv import load_dotenv
from mlflow import log_artifacts, log_metric, log_param
from omegaconf import DictConfig

from notifications import notification


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    


if __name__ == "__main__":
    main()
