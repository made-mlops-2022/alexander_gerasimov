import argparse
from pathlib import Path
from typing import NoReturn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from project.entities.project_params import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    LABEL_COLUMN,
)
from project.utils import setup_logger

logger = setup_logger(path="logs/report.log")


def load_description(data: pd.DataFrame, output_dir: Path) -> NoReturn:
    description = data.describe()
    description = description.round(2)
    description.to_csv(output_dir / "description.csv")


def load_numerical_plots(data: pd.DataFrame, output_dir: Path) -> NoReturn:
    for column in NUMERICAL_COLUMNS:
        sns.boxplot(x=data[column], y=data[LABEL_COLUMN], orient="h")
        plt.savefig(output_dir / f"plot_num_{column}.png")
        plt.show()


def load_categorical_plots(data: pd.DataFrame, output_dir: Path) -> NoReturn:
    sns.countplot(x=data[LABEL_COLUMN])
    plt.savefig(output_dir / f"plot_{LABEL_COLUMN}.png")
    plt.show()

    for column in CATEGORICAL_COLUMNS:
        sns.countplot(x=data[column], hue=data[LABEL_COLUMN])
        plt.savefig(output_dir / f"plot_cat_{column}.png")
        plt.show()


def load_pairplot(data: pd.DataFrame, output_dir: Path) -> NoReturn:
    plt.figure(figsize=(20, 20))
    sns.pairplot(
        data[list(NUMERICAL_COLUMNS.keys()) + [LABEL_COLUMN]], hue=LABEL_COLUMN
    )
    plt.savefig(output_dir / "pairplot.png")


def load_heatmap(data: pd.DataFrame, output_dir: Path) -> NoReturn:
    plt.figure(figsize=(20, 20))
    sns.heatmap(data.corr(), annot=True)
    plt.savefig(output_dir / "heatmap.png")


def setup_parser(parser: argparse.ArgumentParser) -> NoReturn:
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="the path to the input data",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="the folder to save reports to",
        required=True,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="report-generator",
        description="A tool to generate reports on the data provided.",
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    output_dir = Path(arguments.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = list(CATEGORICAL_COLUMNS.keys())
    required_cols.extend(NUMERICAL_COLUMNS.keys())
    required_cols.append(LABEL_COLUMN)

    data = pd.read_csv(Path(arguments.input), usecols=required_cols)

    logger.info("Loading statictics...")
    load_description(data, output_dir)
    logger.info("Loading numerical plots...")
    load_numerical_plots(data, output_dir)
    logger.info("Loading categorical plots...")
    load_categorical_plots(data, output_dir)
    logger.info("Loading pairplot...")
    load_pairplot(data, output_dir)
    logger.info("Loading heatmap...")
    load_heatmap(data, output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
