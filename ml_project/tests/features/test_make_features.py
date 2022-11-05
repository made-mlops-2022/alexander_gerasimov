from typing import List
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from project.data.make_dataset import read_data
from project.entities.feature_params import FeatureParams
from project.features.build_features import build_transformer, make_features


@pytest.fixture(scope="function")
def feature_params(
    categorical_features: List[str],
    features_to_drop: List[str],
    numerical_features: List[str],
    target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col,
    )
    return params


def test_make_features(feature_params: FeatureParams, dataset_path: str):
    data = read_data(dataset_path)

    transformer = build_transformer(feature_params)
    transformer.fit(data)

    features, target = make_features(transformer, data, feature_params)
    assert not pd.isnull(features).any().any()
    assert all(x not in features.columns for x in feature_params.features_to_drop)
    assert_allclose(data[feature_params.target_col].to_numpy(), target.to_numpy())
