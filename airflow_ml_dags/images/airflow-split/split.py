import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split


DATA_FILENAME = "data.csv"
TARGET_FILENAME = "target.csv"
TRAIN_DATA_FILENAME = "train.csv"
VALID_DATA_FILENAME = "valid.csv"

@click.command("split")
@click.option("--input_dir", required=True)
@click.option("--output_dir", required=True)
@click.option("--train_size", default=0.85)
@click.option("--random_state", default=42)
def split(input_dir: str, output_dir: str, train_size: float, random_state: int):
    path_to_data = os.path.join(input_dir, DATA_FILENAME)
    path_to_target = os.path.join(input_dir, TARGET_FILENAME)

    data = pd.read_csv(path_to_data)
    target = pd.read_csv(path_to_target)
    data["target"] = target.values
    train_data, valid_data = train_test_split(
        data, train_size=train_size, random_state=random_state,
    )

    os.makedirs(output_dir, exist_ok=True)
    path_to_train_data = os.path.join(output_dir, TRAIN_DATA_FILENAME)
    path_to_valid_data = os.path.join(output_dir, VALID_DATA_FILENAME)
    train_data.to_csv(path_to_train_data, index=False)
    valid_data.to_csv(path_to_valid_data, index=False)


if __name__ == "__main__":
    split()
