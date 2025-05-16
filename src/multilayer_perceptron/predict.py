import os
import typer
import pandas as pd
import numpy as np
from litetorch.nn.sequential import Sequential
from sklite.metrics import AccuracyScore
from .config import TARGET_FEATURE, DATASET_COLUMNS
from .utils.preprocess import preprocess


predict_cli = typer.Typer()


def get_model_name(model_path: str) -> str:
    """
    Get the model name from the model path.
    """
    return os.path.basename(model_path).split(".")[0]


@predict_cli.command()
def run(
    model_path: str = typer.Option(..., help="Path to the trained model."),
    data_path: str = typer.Option(..., help="Path to the input data for prediction."),
    output_path: str = typer.Option("data/predicted", help="Path to save the prediction results."),
):
    data = np.load(data_path, allow_pickle=True)["data"]
    data = pd.DataFrame(data, columns=DATASET_COLUMNS)
    data = preprocess(data)

    X = data.drop(columns=[TARGET_FEATURE]).values
    y = data[TARGET_FEATURE].values

    model = Sequential.load(model_path)
    model.eval()
    y_pred : np.ndarray = model(X).data
    
    # Convert the probability predictions to binary labels
    y_pred = y_pred.flatten()
    y_pred_labels = (y_pred >= 0.5).astype(int)
    df_y_pred = pd.DataFrame({"probability": y_pred, "predict": y_pred_labels, "actual": y})
    
    metric = AccuracyScore()
    accuracy_score_value = metric(y, y_pred_labels)
    print(f"Accuracy score: {accuracy_score_value:.4f}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, f'{get_model_name(model_path)}_predictions.csv')
    df_y_pred.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}.")
