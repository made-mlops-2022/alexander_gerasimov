import os
import pandas as pd
import pickle
import click


DATA_FILENAME = "data.csv"
MODEL_FILENAME = "model.pkl"
PREDICTIONS_FILENAME = "predictions.csv"
TARGET_COL = "target"

@click.command("predict")
@click.option("--input_dir", required=True)
@click.option("--models_dir", required=True)
@click.option("--output_dir", required=True)
def predict(input_dir: str, models_dir: str, output_dir: str):
    path_to_model = os.path.join(models_dir, MODEL_FILENAME)
    path_to_data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    path_to_predictions = os.path.join(output_dir, PREDICTIONS_FILENAME)

    with open(path_to_model, "rb") as fin:
        model = pickle.load(fin)

    df = pd.read_csv(path_to_data)
    predictions = pd.DataFrame(model.predict(df))

    os.makedirs(output_dir, exist_ok=True)
    predictions.to_csv(path_to_predictions, index=False)


if __name__ == '__main__':
    predict()
