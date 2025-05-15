import os
import typer
import numpy as np
import pandas as pd
from datetime import datetime
from litetorch.nn.sequential import Sequential
from litetorch.nn.linear import Linear
from litetorch.nn.activation import ReLU
from litetorch.data.split import train_val_split
from litetorch.data.dataloader import DataLoader
from litetorch.training.trainer import Trainer
from sklite.preprocessing import LabelEncoder, StandardScaler
from .utils.register import LOSS_REGISTRY, ACTICATION_REGISTRY, OPTIMIZER_REGISTRY


train_cli = typer.Typer()

TARGET_FEATURE = "Diagnosis"
COLUMNS_NAME = [
    "ID",
    "Diagnosis",
    "Mean Radius",
    "Mean Texture",
    "Mean Perimeter",
    "Mean Area",
    "Mean Smoothness",
    "Mean Compactness",
    "Mean Concavity",
    "Mean Concave Points",
    "Mean Symmetry",
    "Mean Fractal Dimension",
    "Radius SE",
    "Texture SE",
    "Perimeter SE",
    "Area SE",
    "Smoothness SE",
    "Compactness SE",
    "Concavity SE",
    "Concave Points SE",
    "Symmetry SE",
    "Fractal Dimension SE",
    "Worst Radius",
    "Worst Texture",
    "Worst Perimeter",
    "Worst Area",
    "Worst Smoothness",
    "Worst Compactness",
    "Worst Concavity",
    "Worst Concave Points",
    "Worst Symmetry",
    "Worst Fractal Dimension",
]


def preprocess(data: pd.DataFrame, target_feature: str = TARGET_FEATURE) -> pd.DataFrame:
    # 1. Drop the ID column
    data = data.drop(columns=["ID"])

    # 2. Encode the target feature
    label_encoder = LabelEncoder(columns=[target_feature])
    label_encoder.fit(data)
    data = label_encoder.transform(data)

    # 3. Split the data into features and target
    X = data.drop(columns=[target_feature])
    y = data[target_feature]

    # 4. Normalize the features
    scaler = StandardScaler(columns=X.columns.tolist())
    scaler.fit(X)
    X = scaler.transform(X)

    # 5. Convert the data back to a DataFrame
    data = pd.DataFrame(X, columns=COLUMNS_NAME[2:])
    data[target_feature] = y
    data = data.astype({target_feature: "int"})
    return data


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

    # Load training data and split into train and validation sets
    data = np.load(train_filepath, allow_pickle=True)["data"]
    data = pd.DataFrame(data, columns=COLUMNS_NAME)

    # Preprocess the data
    data = preprocess(data, target_feature=TARGET_FEATURE)
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
        val_loader
    )
    trainer.train()

    # Save the trained model
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(os.path.join(output_dir, f'{trainer}T{datetime.now().strftime("%Y%m%d%H%M%S")}.json'))


if __name__ == "__main__":
    train_cli()
