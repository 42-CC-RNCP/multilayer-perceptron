import os
import typer
import numpy as np
from litetorch.nn.sequential import Sequential
from litetorch.nn.linear import Linear
from litetorch.nn.activation import ReLU


model_cli = typer.Typer()

@model_cli.command()
def create(
        input_size: int,
        hidden_size: int = 2,
        output_size: int = 1,
        output_path: str = "model.json"):
    print(f"Creating a model with {input_size} input features, {hidden_size} hidden features, and {output_size} output features...")
    model = Sequential(
        Linear(input_size, hidden_size),
        ReLU(),
        Linear(hidden_size, output_size)
    )
    model.save(os.path.join('saved_models', output_path))

if __name__ == "__main__":
    model_cli()
