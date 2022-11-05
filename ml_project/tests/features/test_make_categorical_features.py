from typing import List
import numpy as np
import pandas as pd
import pytest
from project.features.build_features import process_categorical_features


@pytest.fixture(scope="function")
def categorical_feature() -> str:
    return "categorical_feature"


@pytest.fixture(scope="function")
def categorical_values() -> List[str]:
    return ["cat", "dog", "parrot"]


@pytest.fixture(scope="function")
def categorical_values_with_nan(categorical_values: List[str]) -> List[str]:
    return categorical_values + [np.nan]


@pytest.fixture(scope="function")
def fake_categorical_data(
    categorical_feature: str, categorical_values_with_nan: List[str]
) -> pd.DataFrame:
    return pd.DataFrame({categorical_feature: categorical_values_with_nan})


def test_process_categorical_features(
    fake_categorical_data: pd.DataFrame,
    categorical_feature: str,
    categorical_values: List[str],
):
    transformed: pd.DataFrame = process_categorical_features(fake_categorical_data)
    assert 3 == transformed.shape[1]
    assert 4 == transformed.sum().sum()
