import os
import json
import pandas as pd
import click
import pickle
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


VALID_FILENAME = "valid.csv"
MODEL_FILENAME = "model.pkl"
METRICS_FILENAME = "metrics.json"

@click.command("validate")
@click.option("--input_dir")
@click.option("--models_dir")
def validate(input_dir: str, models_dir: str):
    path_to_valid_data = os.path.join(input_dir, VALID_FILENAME)
    path_to_model = os.path.join(models_dir, MODEL_FILENAME)
    path_to_metrics = os.path.join(models_dir, METRICS_FILENAME)

    valid_data = pd.read_csv(path_to_valid_data)
    with open(path_to_model, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(valid_data.drop(columns="target", axis=1))

    metrics = {
        "accuracy": accuracy_score(valid_data["target"], predictions),
        "f1_score": f1_score(valid_data["target"], predictions, average="macro"),
    }

    with open(path_to_metrics, "w") as fout:
        json.dump(metrics, fout)


if __name__ == "__main__":
    validate()
