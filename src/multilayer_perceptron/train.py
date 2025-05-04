import os
import typer
import numpy as np
from typing import Optional
from datetime import datetime
from litetorch.nn.sequential import Sequential
from litetorch.data.dataloader import DataLoader
from litetorch.training.trainer import Trainer
from .utils.register import LOSS_REGISTRY, ACTICATION_REGISTRY, OPTIMIZER_REGISTRY


train_cli = typer.Typer()

def load_as_dataloader(file_path: str, target_idx: int):
    if not os.path.exists(file_path):
        return None
    data = np.load(file_path, allow_pickle=True)["data"]
    X = np.delete(data, target_idx, axis=1)
    y = data[:, target_idx]
    return DataLoader(X, y, shuffle=True)


@train_cli.command()
def start(
    train_filepath: str = "data/train_data.npz",
    val_filepath: Optional[str] = "data/val_data.npz",
    target_idx: int = 0,
    model_path: str = "saved_models/model.json",
    epochs: int = 10,
    loss_fn: str = "cross_entropy",
    optimizer_fn: str = "sgd"):
    """
    Start the training process.
    """

    train_loader = load_as_dataloader(train_filepath, target_idx)
    val_loader = load_as_dataloader(val_filepath, target_idx)

    model = Sequential.load(model_path)

    trainer = Trainer(
        model,
        OPTIMIZER_REGISTRY[optimizer_fn],
        LOSS_REGISTRY[loss_fn],
        train_loader,
        epochs,
        val_loader
    )
    trainer.train()
    trainer.save_model(os.path.join('saved_models', f'trained_model_{datetime.now()}'))

# @train_cli.command()
# def start(
#     train_data: str = "data/train_data.npz",
#     test_data: str = "data/test_data.npz",
#     val_data: str = "data/val_data.npz",
#     target_idx: int = 0,
#     model_path: str = "saved_models/model.json",
#     epochs: int = 10,
#     loss_function: str = "cross_entropy",
#     optimizer_function: str = "sgd"):
#     """
#     Start the training process.
#     """

#     load_to_dataloader()
#     # Load data from file path
#     train_data = np.load(train_data, allow_pickle=True)["data"]
#     test_data = np.load(test_data, allow_pickle=True)["data"]
#     val_data = np.load(val_data, allow_pickle=True)["data"]


#     # Separate the features and target
#     X_train = np.delete(train_data, target_idx, axis=1)
#     y_train = train_data[:, target_idx]
#     X_val = np.delete(val_data, target_idx, axis=1)
#     y_val = val_data[:, target_idx]

#     if loss_function not in LOSS_REGISTRY.keys():
#         raise ValueError(f"Loss function {loss_function} is not supported.")
    
#     if optimizer_function not in OPTIMIZER_REGISTRY.keys():
#         raise ValueError(f"Optimizer {optimizer_function} is not supported.") 

#     # Print the training and validation data shapes
#     print(f"Training data shape: {X_train.shape}")
#     print(f"Validation data shape: {X_val.shape}")
#     print(f"Training target shape: {y_train.shape}")
#     print(f"Validation target shape: {y_val.shape}")
#     print(f"Training with {epochs} epochs using {loss_function} loss and {optimizer} optimizer.")

#     model : Sequential = Sequential().load(model_path)
#     print("model loaded successfully.")
#     print(model)

#     optimizer = OPTIMIZER_REGISTRY[optimizer_function](model.parameters())
#     loss = LOSS_REGISTRY[loss_function]()

#     for epoch in range(epochs):
#         print(f"Epoch {epoch + 1}/{epochs}...")
#         y_pred = model(X_train)
#         loss(y_pred, )

#     model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#     model.save(os.path.join('saved_models', model_name))
#     print(f"Training completed, model saved to {os.path.join('saved_models', model_name)}")


if __name__ == "__main__":
    train_cli()
