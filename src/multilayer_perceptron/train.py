import os
import typer
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from datetime import datetime
from litetorch.nn.sequential import Sequential
from litetorch.data.split import train_val_split
from litetorch.data.dataloader import DataLoader
from litetorch.training.trainer import Trainer
from litetorch.training.callbacks import EarlyStopCallback, TensorboardLoggerCallback
from litetorch.metrics import TrainAccuracy, ValAccuracy, TrainLoss, ValLoss, ConfusionMatrixImage
from .utils.preprocess import preprocess
from .utils.register import LOSS_REGISTRY, OPTIMIZER_REGISTRY
from .config import TARGET_FEATURE, DATASET_COLUMNS


train_cli = typer.Typer()

def trainer_fn(config: Dict) -> Tuple[float, float]:
    train_filepath = config["train_filepath"]
    model_path = config["model_path"]
    output_dir = config["output_dir"]
    epochs = config["epochs"]
    loss_fn = config["loss_fn"]
    optimizer_fn = config["optimizer_fn"]
    callbacks = config["callbacks"]

    # Load training data and split into train and validation sets
    data = np.load(train_filepath, allow_pickle=True)["data"]
    data = pd.DataFrame(data, columns=DATASET_COLUMNS)

    # Preprocess the data
    data = preprocess(data)
    train_data, val_data = train_val_split(data, val_size=0.2, shuffle=False)
    train_data = pd.DataFrame(train_data, columns=data.columns)
    val_data = pd.DataFrame(val_data, columns=data.columns)

    # Create the DataLoader for training and validation
    train_X: np.ndarray = train_data.drop(columns=[TARGET_FEATURE]).values
    train_y: np.ndarray = train_data[TARGET_FEATURE].values
    train_loader = DataLoader(
        train_X,
        train_y,
        batch_size=32,
        shuffle=True,
    )

    val_X: np.ndarray = val_data.drop(columns=[TARGET_FEATURE]).values
    val_y: np.ndarray = val_data[TARGET_FEATURE].values
    val_loader = DataLoader(
        val_X,
        val_y,
        batch_size=32,
        shuffle=False,
    )

    # Load the model and create the optimizer and loss function
    model = Sequential.load(model_path)
    optimizer = OPTIMIZER_REGISTRY[optimizer_fn](model.parameters())
    loss = LOSS_REGISTRY[loss_fn]()

    # Create the trainer and start training
    trainer = Trainer(
        model,
        optimizer,
        loss,
        train_loader,
        epochs,
        val_loader,
        callbacks=callbacks,
    )
    trainer.train()

    # Save the trained model
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(os.path.join(output_dir, f'{trainer}T{datetime.now().strftime("%Y%m%d%H%M%S")}.json'))
    return trainer.train_losses[-1], trainer.val_losses[-1]


@train_cli.command()
def start(
    model_path: str = "saved_models/defined/model.json",
    output_dir: str = "saved_models/trained",
    train_filepath: str = "data/split/train_data.npz",
    epochs: int = 1000,
    loss_fn: str = "binary_cross_entropy",
    optimizer_fn: str = "sgd",
):
    # Check the input parameters
    if not os.path.exists(train_filepath):
        raise FileNotFoundError(f"Train data {train_filepath} not found.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} not found.")
    if loss_fn not in LOSS_REGISTRY.keys():
        raise ValueError(f"Loss function {loss_fn} is not supported.")
    if optimizer_fn not in OPTIMIZER_REGISTRY.keys():
        raise ValueError(f"Optimizer {optimizer_fn} is not supported.")

    tensorboard_callback = TensorboardLoggerCallback(
        log_dir=f"logs/{datetime.now().strftime('%Y%m%d%H')}",
        metrics={
            "Loss/Train": TrainLoss(),
            "Loss/Val": ValLoss(),
            "Accuracy/Train": TrainAccuracy(),
            "Accuracy/Val": ValAccuracy(),
            "ConfusionMatrix": ConfusionMatrixImage(class_names=["B", "M"]),
        }
    )

    # Refactor the code to use the trainer function with the config dictionary
    # TODO: Add tuner to the config
    config = {
        "model_path": model_path,
        "output_dir": output_dir,
        "train_filepath": train_filepath,
        "epochs": epochs,
        "loss_fn": loss_fn,
        "optimizer_fn": optimizer_fn,
        "lr": 0.01,
        "batch_size": 32,
        "callbacks": [
            EarlyStopCallback(patience=10, monitor="val_losses", mode="min"),
            tensorboard_callback,
        ]
    }
    train_loss, val_loss = trainer_fn(config)
    print(f"Training completed. Train Loss: {train_loss}, Validation Loss: {val_loss}")


if __name__ == "__main__":
    train_cli()
