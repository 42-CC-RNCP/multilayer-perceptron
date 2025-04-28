import os
import typer
import numpy as np
import pandas as pd
from litetorch.data.split import train_val_split


split_cli = typer.Typer()

@split_cli.command()
def start(dataset: str = "dataset.csv", split_ratio: float = 0.8, output_dir: str = "data"):
    """
    Start the split process.
    """
    # Check if the dataset exists
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Dataset {dataset} not found.")
    # Check if the output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Splitting the dataset {dataset} with a ratio of {split_ratio} and saving to {output_dir}...")
    df = pd.read_csv(dataset, header=None)
    data = df.values
    train_data, val_data = train_val_split(data, val_size=1 - split_ratio, shuffle=True)

    # Save the train and validation data as npy files
    np.save(os.path.join(output_dir, "train_data.npy"), train_data)
    np.save(os.path.join(output_dir, "val_data.npy"), val_data)
    print(f"Train and validation data saved to {output_dir}.")
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
