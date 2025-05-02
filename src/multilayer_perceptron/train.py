import typer
import numpy as np
from litetorch.nn.sequential import Sequential


train_cli = typer.Typer()

@train_cli.command()
def start(
    train_data: str = "data/train_data.npy",
    val_data: str = "data/val_data.npy",
    target_name: str = "target",
    model_path: str = "model.json",
    epochs: int = 10,
    loss_function: str = "cross_entropy",
    optimizer: str = "adam"):
    """
    Start the training process.
    """
    # Load the training and validation data
    train_data = np.load(train_data)
    val_data = np.load(val_data)

    # Check if the target name is in the data
    if target_name not in train_data.dtype.names:
        raise ValueError(f"Target name {target_name} not found in the training data.")

    # Separate the features and target
    X_train = np.delete(train_data, np.where(train_data.dtype.names == target_name), axis=1)
    y_train = train_data[target_name]
    X_val = np.delete(val_data, np.where(val_data.dtype.names == target_name), axis=1)
    y_val = val_data[target_name]

    # Check if the loss function is valid
    if loss_function not in ["cross_entropy", "mean_squared_error"]:
        raise ValueError(f"Loss function {loss_function} is not supported.")

    # Check if the optimizer is valid
    if optimizer not in ["adam", "sgd"]:
        raise ValueError(f"Optimizer {optimizer} is not supported.")

    # Print the training and validation data shapes
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Validation target shape: {y_val.shape}")
    print(f"Training with {epochs} epochs using {loss_function} loss and {optimizer} optimizer.")

    model = Sequential().load(model_path)

    print("model loaded successfully.")
    print(model)

    for epoch in range(epochs):
        # Simulate training process
        print(f"Epoch {epoch + 1}/{epochs}...")
        # Here you would implement the actual training logic
        # For example, forward pass, backward pass, and weight updates
        # This is just a placeholder for demonstration purposes
        pass
    print("Training completed.")

