import os
import pandas as pd
import click
import pickle
from sklearn.ensemble import RandomForestClassifier


TRAIN_FILENAME = "train.csv"
MODEL_FILENAME = "model.pkl"

@click.command("train")
@click.option("--input_dir")
@click.option("--models_dir")
@click.option("--random_state", default=42)
def train(input_dir: str, models_dir: str, random_state: int):
    path_to_train_data = os.path.join(input_dir, TRAIN_FILENAME)
    path_to_model = os.path.join(models_dir, MODEL_FILENAME)

    train_df = pd.read_csv(path_to_train_data)
    model = RandomForestClassifier(random_state=random_state)
    model.fit(train_df.drop(columns="target", axis=1), train_df["target"])

    os.makedirs(models_dir, exist_ok=True)
    with open(path_to_model, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train()
