import logging
import os

import hydra
import mlflow
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dotenv import load_dotenv
from mlflow import log_artifacts, log_metric, log_param
from omegaconf import DictConfig

from notifications import notification
from models import simplenet


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    load_dotenv()

    notification_slack = notification.slack_notification(
        os.environ["SLACK_WEBHOOK_URL"]
    )
    mlflow.set_tracking_uri(hydra.utils.get_original_cwd() + "/mlruns")
    mlflow.set_experiment(cfg.experiment.name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define transform
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # create dataset and dataloader
    trainset = torchvision.datasets.CIFAR10(
        root=os.path.join(hydra.utils.get_original_cwd(), "data"),
        train=True,
        download=True,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=2
    )

    validationset = torchvision.datasets.CIFAR10(
        root=os.path.join(hydra.utils.get_original_cwd(), "data"),
        train=False,
        download=True,
        transform=transform,
    )
    validationloader = torch.utils.data.DataLoader(
        validationset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2
    )

    # cifar10 classes
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    model = simplenet.Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=cfg.train.learning_rate, momentum=cfg.train.momentum
    )

    with mlflow.start_run(run_name=cfg.experiment.run_name):
        mlflow.log_params(cfg.train)
        for epoch in range(1, cfg.train.epochs + 1):
            logger.info(f"Epoch: {epoch}")
            training_loss = 0.0
            for data in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # logger.info statistics
                training_loss += loss.item()

            logger.info(f"training loss: {training_loss / len(trainloader)}")
            mlflow.log_metric(
                key="train loss", value=training_loss / len(trainloader), step=epoch
            )

            validation_loss = 0.0
            with torch.no_grad():
                for data in validationloader:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()
                logger.info(
                    f"validation loss: {validation_loss / len(validationloader)}"
                )
                mlflow.log_metric(
                    key="valid loss",
                    value=validation_loss / len(validationloader),
                    step=epoch,
                )

        save_dir = os.path.join(hydra.utils.get_original_cwd(), "results")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "simple_model.hdf5")

        torch.save(model.state_dict(), save_path)

        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifacts(save_dir)

    if cfg.notification.is_notification:
        notification_slack.send_message("Experiment completed.")


if __name__ == "__main__":
    main()
