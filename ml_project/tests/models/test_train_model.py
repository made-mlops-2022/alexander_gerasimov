import os
import pickle
from typing import List, Tuple
import pandas as pd
import pytest
from py._path.local import LocalPath
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from project.data.make_dataset import read_data
from project.entities import TrainParams
from project.entities.feature_params import FeatureParams
from project.features.build_features import build_transformer, make_features
from project.models.train_model import serialize_model, train_model


@pytest.fixture(scope="function")
def features_and_target(
    dataset_path: str, categorical_features: List[str], numerical_features: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=[],
        target_col="target",
    )
    data = read_data(dataset_path)
    transformer = build_transformer(params)
    transformer.fit(data)
    features, target = make_features(transformer, data, params)
    return features, target


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainParams())
    assert isinstance(model, RandomForestClassifier)
    assert model.predict(features).shape[0] == target.shape[0]


def test_serialize_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    transformer = ColumnTransformer([])
    n_estimators = 10
    model = RandomForestClassifier(n_estimators=n_estimators)
    real_output = serialize_model(model, expected_output, transformer)
    assert real_output == expected_output
    assert os.path.exists(real_output)
    with open(real_output, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, Pipeline)
