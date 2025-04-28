import typer


train_cli = typer.Typer()

@train_cli.command()
def start(epochs: int = 10):
    """
    Start the training process.
    """
    typer.echo(f"Starting the training process with {epochs} epochs...")
