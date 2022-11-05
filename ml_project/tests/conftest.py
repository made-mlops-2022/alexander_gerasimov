import os
from typing import List
import pytest
from tests.utils import generate_dataset


@pytest.fixture(scope="module")
def dataset_path():
    path = os.path.join(os.path.dirname(__file__), "train_data_sample.csv")
    data = generate_dataset()
    data.to_csv(path)
    return path


@pytest.fixture(scope="module")
def model_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "model.pkl")


@pytest.fixture(scope="module")
def output_data_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "prediction.csv")


@pytest.fixture(scope="module")
def target_col():
    return "target"


@pytest.fixture(scope="module")
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture(scope="module")
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture(scope="module")
def features_to_drop() -> List[str]:
    return []
