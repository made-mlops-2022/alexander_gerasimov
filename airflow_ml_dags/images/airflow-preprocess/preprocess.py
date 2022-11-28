import os
from shutil import copyfile
import click
import pandas as pd


DATA_FILENAME = "data.csv"
TARGET_FILENAME = "target.csv"

@click.command("preprocess")
@click.option("--input_dir")
@click.option("--output_dir")
def preprocess(input_dir: str, output_dir: str):
    path_to_raw_data = os.path.join(input_dir, DATA_FILENAME)
    path_to_raw_target = os.path.join(input_dir, TARGET_FILENAME)

    os.makedirs(output_dir, exist_ok=True)
    path_to_processed_data = os.path.join(output_dir, DATA_FILENAME)
    path_to_processed_target = os.path.join(output_dir, TARGET_FILENAME)

    train_df = pd.read_csv(path_to_raw_data)
    train_df.fillna(0)
    train_df.to_csv(path_to_processed_data, index=False)

    if os.path.isfile(path_to_raw_target):
        copyfile(path_to_raw_target, path_to_processed_target)


if __name__ == "__main__":
    preprocess()
