import typer
from src.multilayer_perceptron.split import split_cli
from src.multilayer_perceptron.train import train_cli
from src.multilayer_perceptron.predict import predict_cli


app = typer.Typer()
app.add_typer(split_cli, name="split")
app.add_typer(train_cli, name="train")
app.add_typer(predict_cli, name="predict")

if __name__ == "__main__":
    app()
