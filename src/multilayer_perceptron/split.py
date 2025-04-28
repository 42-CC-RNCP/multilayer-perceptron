import typer


split_cli = typer.Typer()

@split_cli.command()
def start(dataset: str = "dataset.csv", split_ratio: float = 0.8, output_dir: str = "output"):
    """
    Start the split process.
    """
    print(f"Splitting the dataset {dataset} with a ratio of {split_ratio} and saving to {output_dir}...")
