import os
import typer
import numpy as np
import pandas as pd
from litetorch.data.split import train_val_test_split


split_cli = typer.Typer()

@split_cli.command()
def start(
        dataset: str = "dataset.csv",
        val_size: float = 0.0,
        test_size: float = 0.2,
        output_dir: str = "data"):
    """
    Start the split process.
    """
    # Check if the dataset exists
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Dataset {dataset} not found.")
    # Check if the output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Splitting the dataset {dataset} saving to {output_dir}...")
    df = pd.read_csv(dataset)
    data = df.values
    train_data, val_data, test_data = train_val_test_split(data, val_size, test_size, shuffle=False)

    # Save the train and validation data as npy files
    np.savez(
        os.path.join(output_dir, "train_data.npz"),
        data=train_data,
        columns=df.columns.to_numpy()
    )
    np.savez(
        os.path.join(output_dir, "val_data.npz"),
        data=val_data,
        columns=df.columns.to_numpy()
    )
    np.savez(
        os.path.join(output_dir, "test_data.npz"),
        data=test_data,
        columns=df.columns.to_numpy()
    )
    print(f"Train and validation data saved to {output_dir}.")
    print(f"Original data shape: {data.shape}")
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Train data output path: {os.path.join(output_dir, 'train_data.npz')}")
    print(f"Validation data output path: {os.path.join(output_dir, 'val_data.npz')}")
    print(f"Test data output path: {os.path.join(output_dir, 'test_data.npz')}")

if __name__ == "__main__":
    split_cli()
